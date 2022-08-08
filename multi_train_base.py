from transformers import RobertaTokenizer, AdamW, GPT2Tokenizer,BertTokenizer, BertConfig # ,  GPT2Tokenizer, GPT2LMHeadModel,Conv1D ,RobertaConfig # get_linear_schedule_with_warmup, GPT2Config
from tqdm import tqdm
import torch.nn.functional as F
from tools.common import seed_everything
import torch.nn as nn
import torch
import os
import importlib
from rouge import Rouge
from model_util import init_para_frompretrained, linear_schedule
import re
import math
import argparse
from data.util_rg import prepare_dataset
from new_model import MultiVAEModel
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import logging
import wandb
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.models.roberta.modeling_roberta import RobertaLMHead
from transformers import BertModel, BertConfig, RobertaConfig, \
    DebertaConfig, DebertaModel, GPT2Model, GPT2LMHeadModel, GPT2Config, DebertaTokenizer
from train import train_step, compute_loss, top_k_top_p_filtering
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

encoder_model_class = {
    'bert-base-cased': (BertConfig, BertModel, BertTokenizer),
    'roberta-base': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'microsoft/deberta-base': (DebertaConfig, DebertaModel, DebertaTokenizer)
}
decoder_model_class = {
    'bert-base-cased': (BertConfig, BertModel, BertOnlyMLMHead, BertTokenizer),
    'roberta-base': (RobertaConfig, RobertaModel, RobertaLMHead, RobertaTokenizer),
    'gpt2': (GPT2Config, GPT2Model, GPT2LMHeadModel, GPT2Tokenizer)
}

def sample_sequence(args, model, tokenizer, tokenizer_r, length, batch_size=None, x_mask=None, x_tokens=None,
                    y_mask=None, y_tokens=None, temperature=1, top_k=100, top_p=0.95, device='cuda', sample=True,
                    eos_token=None, model_type='cvae', decoder_x=None, decoder_x_mask=None):
    x_mask = x_mask.to(device)
    x_tokens = x_tokens.to(device)
    x_tokens_gpt = decoder_x.to(device)
    x_mask_gpt = decoder_x_mask.to(device)

    with torch.no_grad():
        prior = model.encoder_prior(input_ids=x_tokens, attention_mask=x_mask).last_hidden_state
        prior, _ = model.averageSelfAttention_prior(prior, x_mask.squeeze(1).squeeze(1))
        prior_mean = model.mean_prior(prior)
        z = prior_mean

        assert not torch.isnan(z).any(), 'sampling sequence get nan z'
        outputs = model.transformer(input_ids=x_tokens_gpt,
                                    past_key_values=None,
                                    attention_mask=x_mask_gpt,
                                    )
        if 'gpt2' in args.decoder_name:
            prev = x_tokens_gpt[:, -1].view(batch_size, -1)
        elif 'roberta' in args.decoder_name:
            prev = torch.LongTensor([0]).unsqueeze(0).to(args.device)
        elif 'bert-' in args.decoder_name and 'cased' in args.decoder_name:
            prev = torch.LongTensor([101]).unsqueeze(0).to(args.device)

        mem = outputs.past_key_values

        output = prev
        probability = torch.tensor([], dtype=z.dtype, device=device)
        if_end = torch.tensor([False] * batch_size, dtype=torch.bool, device=device)

        for i in range(length):# trange
            # logits, mem
            outputs = model.transformer(input_ids=prev, past_key_values=mem) # , representations=z
            mem = outputs.past_key_values

            input_proj = model.input_proj(z).unsqueeze(1)
            hidden = outputs.last_hidden_state + input_proj

            logits = model.lm_head(hidden)

            logits = logits[:, -1, :] / temperature
            logits = top_k_top_p_filtering(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            if sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                _, next_token = torch.topk(probs, k=1, dim=-1)
            probability = torch.cat((probability, probs.gather(1, next_token)), dim=1)
            output = torch.cat((output, next_token), dim=1)
            prev = next_token
            # early stopping if all sents have ended once
            if_end[next_token.view(-1).eq(eos_token)] = True
            if if_end.all():
                break
    return output, probability

def val_step(VAE, val_loader, tokenizer, args, device, loss_fn, endoftext, endoftext_id, max_val_batches, log=True):
    VAE.eval()
    n_words_bpe = 0
    n_words = 0
    logp_sum = 0.0
    kl_loss_sum = 0.0
    print("Validation loop.         Batches: %d" % len(val_loader))
    print("Validation loop. max_val_batches: %d" % max_val_batches)
    with tqdm(total=min(len(val_loader), max_val_batches)) as pbar:
        for i, (x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, decoder_x,
                decoder_x_mask) in enumerate(val_loader):
            with torch.no_grad():
                try:
                    input_tokens = input_tokens.view(-1, input_tokens.shape[-1])
                except:
                    target_tokens = y_tokens
                    input_tokens = y_tokens
                    mask = y_mask
                    print(input_tokens)
                loss, ce_loss, kl_loss = compute_loss(args, device, VAE, x_mask, x_tokens, y_mask, y_tokens,
                                                      input_tokens, target_tokens, mask, loss_fn, 1.0)
            if len(target_tokens.size()) == 1:
                target_tokens = target_tokens.unsqueeze(0)
            n, l = target_tokens.size()

            text = target_tokens[0, :].tolist()
            logprob = ce_loss.tolist()

            assert len(text) == len(logprob)

            if endoftext_id in text:
                idx = text.index(endoftext_id)
                text = text[:idx]
                logprob = logprob[:idx]

            logp_sum += sum(logprob)
            n_words_bpe += len(text)

            story = [tokenizer.decode(target_tokens[i, :]) for i in range(n)]
            assert len(story) == 1
            story = [s[:s.find(endoftext) + len(endoftext)] if endoftext in s else s for s in
                     story]
            all_words = [t for t in re.split('("|\'|!|\?|\.|,|:| |\n|’|“|”|;|\(|\)|`)', story[0])
                         if t != ' ' and t != '']
            words = len(all_words)
            n_words += words
            kl_loss_sum += kl_loss.item()
            if i > max_val_batches:
                break
            pbar.update(1)

    loss_bpe = logp_sum / n_words_bpe
    ppl_bpe = round(math.exp(min(logp_sum / n_words_bpe, 100)), 5)
    ppl_word = round(math.exp(min(logp_sum / n_words, 100)), 5)
    kl = kl_loss_sum / len(val_loader)

    if args.use_wandb and log:
        wandb.log({'eval ppl_bpe': ppl_bpe,  'eval ppl_word': ppl_word,'eval kl': kl, 'eval loss': loss_bpe})
    VAE.train()

def generate(VAE, args, test_loader, tokenizer, save_folder, tokenizer_r, device, endoftext, endoftext_id,
             num_iters, max_test_examples, log=True, print_generated=False):
    VAE.eval()
    n_samples = 0
    bleu4_sum = 0.0
    dic1 = {}
    dic2 = {}
    distinc1, distinc2 = 0, 0
    all1, all2 = 0, 0
    rouge_scores_values_sum = [0.0] * 3
    args.nsamples = 1
    args.batch_size = 1
    args.temperature = 1 # 0.95
    args.repetition_penalty = 1.0
    args.num_return_sequences = 1

    model_type = args.model_type
    samples_file = open(os.path.join(save_folder, 'generate-' + '%07d' % num_iters + args.note + '.txt'),
                        'w', encoding='utf8')
    print("***************test*******************")
    with tqdm(total=min(len(test_loader), max_test_examples)) as pbar:
        for i_test, (x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask,
                     decoder_x, decoder_x_mask) in enumerate(test_loader):
            if i_test >= max_test_examples:
                break
            if args.dataset in ['story', 'roc']:
                length = args.length
            elif args.dataset == 'wi':
                length = args.length
            else:
                raise(ValueError)
            args.generate_length = length
            eff_samples = []
            n, l = target_tokens.size()
            storys = [tokenizer.decode(target_tokens[i, :-1]) for i in range(n)] # [i, :-1] 去掉最后一位50256

            storys_str = [s[:s.find(endoftext) + len(endoftext)] if endoftext in s else s for s in
                          storys]
            for _ in range(args.nsamples // args.batch_size):
                out, _ = sample_sequence(
                    args,
                    model=VAE,
                    tokenizer=tokenizer,
                    tokenizer_r=tokenizer_r,
                    length=length,
                    batch_size=args.batch_size,
                    x_mask=x_mask,
                    x_tokens=x_tokens,
                    y_mask=y_mask,
                    y_tokens=y_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    device=device,
                    eos_token=endoftext_id,
                    model_type=model_type,
                    decoder_x=decoder_x,
                    decoder_x_mask=decoder_x_mask
                )
                out = out.tolist()
                # extract story, check metrics
                for i in range(len(out)):
                    text = out[i]
                    if endoftext in text:
                        idx = text.index(endoftext)
                        text = text[:idx]

                    text = tokenizer.decode(text).strip()
                    text = text.lstrip('<s>')
                    text = text.rstrip('</s>')
                    text = text.lstrip('[CLS]')
                    text = text.rstrip('[SEP]')

                    if print_generated:
                        print(text)

                    if args.dataset in ['story', 'roc']:
                        text_word = text.split()
                        for word in text_word:
                            all1 += 1
                            if word not in dic1:
                                dic1[word] = 1
                                distinc1 += 1
                        for m in range(0, len(text_word) - 1):
                            all2 += 1
                            if text_word[m] + " " + text_word[m + 1] not in dic2:
                                dic2[text_word[m] + " " + text_word[m + 1]] = 1
                                distinc2 += 1
                    try:
                        # check bleu
                        ref = storys_str[i].split()
                        ref[-1] = ref[-1].strip(endoftext)
                        text_word = text.split()

                        sf = SmoothingFunction()

                        bleu4 = sentence_bleu([ref], text_word, smoothing_function=sf.method7)
                        # check rouge
                        rouge = Rouge()
                        label = storys_str[i].strip(endoftext)
                        rouge_scores = rouge.get_scores(text, label) # 两个input都是str
                        rouge_scores_values = []
                        for i in rouge_scores[0].keys():
                            if i != 'rouge-l':
                                rouge_scores_values.append(rouge_scores[0][i]['r'])
                            else:
                                rouge_scores_values.append(rouge_scores[0][i]['f'])
                        bleu4_sum += bleu4
                        rouge_scores_values_sum = [v1 + v2 for v1, v2 in
                                                   zip(rouge_scores_values_sum, rouge_scores_values)]
                        n_samples += 1
                    except:
                        bleu4 = 0.0
                        rouge_scores = [{'rouge-1': {'r': 0.0},# , 'p': 0.0, 'r': 0.0
                                         'rouge-2': {'r': 0.0},
                                         'rouge-l': {'f': 0.0}}]
                    eff_samples.append((text, bleu4, rouge_scores))
                pbar.update(1)


            for i in range(len(eff_samples)):
                samples_file.write("=" * 50 + " SAMPLE " + str(i_test) + " " + "=" * 50)
                samples_file.write('\n' * 2)
                samples_file.write("=" * 40 + " Outlines  " + "=" * 40)
                samples_file.write('\n' * 2)

                if args.dataset not in ['story', 'roc']:
                    samples_file.write(tokenizer_r.decode(x_tokens[i, :][x_mask[i, :] == 1].tolist()))
                else:
                    samples_file.write(tokenizer_r.decode(x_tokens[i, :].tolist())) # 对于roc 将****也打印出来
                # samples_file.write('\n' * 2)
                # samples_file.write(tokenizer.decode(y_tokens[i, :][y_mask[i, :] == 1].tolist()))
                samples_file.write('\n' * 2)
                samples_file.write("=" * 40 + " Story " + "=" * 40)
                samples_file.write('\n' * 2)
                samples_file.write(storys_str[i])
                samples_file.write('\n' * 2)
                samples_file.write("=" * 40 + " Generated " + "=" * 40)
                samples_file.write('\n' * 2)
                samples_file.write(eff_samples[i][0])
                samples_file.write('\n' * 4)
                samples_file.flush()

    print('Test complete with %05d samples.' % n_samples)

    if n_samples != 0:
        bleu4 = round(bleu4_sum / n_samples, 5)
        rouge_scores_values = [round(r / n_samples, 5) for r in rouge_scores_values_sum]
    else:
        bleu4 = 0
        rouge_scores_values = [0 for r in rouge_scores_values_sum]

    print(' bleu-4:', float(bleu4*100))
    print(' rouge :', rouge_scores_values)

    if args.use_wandb and log:
        if args.dataset in ['story', 'roc']:
            wandb.log({'valid_samples': n_samples, 'bleu4': float(bleu4*100),
                       'rouge1 r': rouge_scores_values[0],
                       'rouge2 r': rouge_scores_values[1],
                       'rougel f': rouge_scores_values[2],
                       'distinc1': float(distinc1 / all1)*100,
                       'distinc2': float(distinc2 / all2)*100})
        else:
            wandb.log({'valid_samples': n_samples, 'bleu4': float(bleu4*100),
                       'rouge1 r': rouge_scores_values[0],
                       'rouge2 r': rouge_scores_values[1],
                       'rougel f': rouge_scores_values[2]})
    VAE.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='train')
    # Default parameters are set based on single GPU training
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--note', type=str, default='', help="used to record the purpose of each run")

    parser.add_argument('--encoder_name', type=str, default='roberta-base')
    parser.add_argument('--decoder_name', type=str, default='roberta-base')
    parser.add_argument('--few_shot', action="store_true", default=False)

    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--top_p', type=float, default=0.8)

    parser.add_argument('--length', type=int, default=32)
    parser.add_argument('--max_size', type=int, default=2000000)
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Whether to run wandb.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--data_type', type=str, default='t0', choices=['t' + str(i) for i in range(9)], help="t: type")
    parser.add_argument('--model_type', type=str, default='cvae', choices=['cvae'])
    parser.add_argument('--iterations', type=int, default=50000) #  wi 300001
    parser.add_argument('--dataset', type=str, default='roc', choices=['roc', 'story', 'wi'],
                        help="Dataset to use for training")
    parser.add_argument('--warmup', type=int, default=100,
                        help="Amount of iterations to warmup, then decay. (-1 for no warmup and decay)")
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[16],
                        help='batch size per GPU. Lists the schedule.')
    parser.add_argument('--seq-lens', nargs='+', type=int, default=[256],
                        help='seq length per sample. Lists the schedule.')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--out-dir', type=str, default='out_files')
    parser.add_argument('--load', type=str, help='path to load model from') # , default='out/test/'
    # use GPU
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--no_gpu', action="store_true")
    parser.add_argument('--learn_prior', action="store_true", default=True)
    parser.add_argument('--ed_share', action="store_true", default=False)


    args = parser.parse_args()
    args.learn_prior = True
    if args.ed_share:
        assert args.encoder_name == args.decoder_name

    args.group_name = args.encoder_name + '_' + args.decoder_name
    args.project_name = args.dataset
    if args.use_wandb:
        wandb.init(config=args, project=args.project_name, entity='entity name', group=args.group_name)
    args.experiment = args.project_name + args.group_name + str(args.lr)
    save_folder = os.path.join(args.out_dir,  args.experiment)
    os.makedirs(save_folder, exist_ok=True)

    args.gpu = 0
    device = torch.device(args.gpu)
    args.device = device

    # randomness
    seed_everything(args.seed)

    importlib.reload(logging)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    print('Loading models...')
    cache_dir = os.path.join(args.out_dir, 'model_cache')
    os.makedirs(cache_dir, exist_ok=True)

    tokenizer = decoder_model_class[args.decoder_name][-1].from_pretrained(args.decoder_name)
    tokenizer_r = encoder_model_class[args.encoder_name][-1].from_pretrained(args.encoder_name)

    print('Setup data...')
    assert len(args.batch_sizes) == len(args.seq_lens) == 1

    if args.dataset == 'roc':
        args.data_type = 't0'
        args.batch_sizes = [16]
        args.seq_lens = [256]
        args.iterations = 160000//args.batch_sizes[0]
        tuning_all_after_iters = args.iterations//100
        test_period = args.iterations//10
        eval_period = args.iterations//10
    elif args.dataset == 'story':
        args.data_type = 't0'
        args.batch_sizes = [16]
        args.seq_lens = [256]
        args.iterations = 80000//args.batch_sizes[0]
        tuning_all_after_iters = args.iterations//100
        test_period = args.iterations//20
        eval_period = args.iterations//20
    elif args.dataset == 'wi':
        args.data_type = 't0'
        args.batch_sizes = [16]
        args.seq_lens = [256]
        args.iterations = 200000//args.batch_sizes[0]
        tuning_all_after_iters = args.iterations//100
        test_period = args.iterations//10
        eval_period = args.iterations//10
    else:
        raise(NotImplementedError)

    args.warmup = args.iterations//100

    if args.decoder_name == 'gpt2':
        endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        endoftext = "<|endoftext|>"
        tokenizer.pad_token_id = [50256]
    else:
        endoftext_id = tokenizer.sep_token_id
        endoftext = tokenizer.sep_token
        tokenizer.eos_token = tokenizer.sep_token

    batch_schedule = list(zip(map(int, args.batch_sizes), map(int, args.seq_lens))) # [(16, 256)]
    assert len(batch_schedule) <= 2, 'Currently not supporting multiple schedule'

    encoder_special_id_list = []
    encoder_special_id_list.append(tokenizer_r.convert_tokens_to_ids(' *'))
    encoder_special_id_list.append(tokenizer_r.convert_tokens_to_ids('*'))
    encoder_special_id_list.append(tokenizer_r.convert_tokens_to_ids(tokenizer_r.tokenize(' *'))[0])
    encoder_special_id_list.append(tokenizer_r.convert_tokens_to_ids(tokenizer_r.tokenize('*'))[0])

    train_loader, val_loader, test_loader = prepare_dataset(args,
        data_dir=args.data_dir, dataset_name=args.dataset,
        encoder_special_id_list=encoder_special_id_list,
        tokenizer_encoder=tokenizer_r, tokenizer_decoder=tokenizer,
        train_bsz=batch_schedule[0][0], train_seq_len=batch_schedule[0][1],
        val_seq_len=batch_schedule[-1][1],
        test_seq_len=batch_schedule[-1][1],
        num_workers=1, make_train=True, make_val=True, make_test=True,
        data_type=args.data_type, max_size=args.max_size,
        )
    print('************ task name   ' + args.dataset + ' ****************')
    VAE = MultiVAEModel(encoder_model_class, decoder_model_class, args.encoder_name, args.decoder_name,
                        device=args.device,  ed_share=args.ed_share).to(device)
    e = 0  # number of epoch
    num_iters = 0
    beta = 1.00
    lr_schedule = linear_schedule(args)
    VAE.train()

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in VAE.named_parameters()],
            "lr":args.lr
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, correct_bias=False) # in BERT they use correct_bias=False,
    # can try True later, but probably won't make difference.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    print('Done.')
    optimizer.zero_grad()

    tuning_all = False
    if not tuning_all:
        if 'bert' in args.decoder_name:
            new_pars = ['c_z', 'attention_weights', 'mean', 'logvar', 'input_proj',
                        'attn_proj', 'Nu_fc1', 'Nu_fc2',
                        'lm_head_rep', # 'encoder', 'encoder_prior', 'transformer',
                        'averageSelfAttention', 'lm_head']
        else:
            new_pars = ['c_z', 'attention_weights', 'mean', 'logvar', 'input_proj',
                        'attn_proj', 'Nu_fc1', 'Nu_fc2',
                        'lm_head_rep', 'encoder', 'encoder_prior',
                        'averageSelfAttention']

        for name, parameter in VAE.named_parameters():
            if not any([True if n in name else False for n in new_pars]):
                parameter.requires_grad = False
        # print(VAE.lm_head.weight.requires_grad)

    print('first eval and generate before training...')

    val_step(VAE, val_loader, tokenizer, args, device, loss_fn, endoftext, endoftext_id, 10, log=False)

    generate(VAE, args, test_loader, tokenizer, save_folder, tokenizer_r,
             device, endoftext, endoftext_id, num_iters, 10, log=False, print_generated=True)

    VAE.train()
    while num_iters < args.iterations:
        # Run epoch Training
        print('Training loop. Batches:', len(train_loader))
        with tqdm(total=args.iterations) as pbar:
            for i, (x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask,
                    decoder_x, decoder_x_mask) in enumerate(train_loader):

                if not tuning_all and num_iters >= tuning_all_after_iters:
                    for name, parameter in VAE.named_parameters():
                        parameter.requires_grad = True
                    tuning_all = True
                    if args.use_wandb:
                        wandb.log({'start tuning all': num_iters})

                output = train_step(args, device, VAE, optimizer, x_mask, x_tokens, y_mask, y_tokens,
                                    input_tokens, target_tokens, mask, loss_fn, beta, args.model_type)
                loss, ce_loss, kl_loss = output[-1]
                if args.use_wandb:
                    wandb.log({'train loss': loss})

                end = num_iters >= args.iterations
                if args.warmup != -1:
                    scheduler.step()
                if end:
                    break
                num_iters += 1
                pbar.update(1)

                if (num_iters+1) % eval_period == 0:
                    val_step(VAE, val_loader, tokenizer, args, device, loss_fn,  endoftext,
                             endoftext_id, len(val_loader), log=True)
                if (num_iters+1) % test_period == 0:
                    generate(VAE, args, test_loader, tokenizer, save_folder, tokenizer_r, device,
                             endoftext, endoftext_id,
                             num_iters, len(test_loader), log=True)
        if not end:
            e += 1

    print('Training complete.')
    print('final eval and test.')

    val_step(VAE, val_loader, tokenizer, args, device, loss_fn,  endoftext, endoftext_id, len(val_loader)//10, log=True)
    generate(VAE, args, test_loader, tokenizer, save_folder, tokenizer_r, device, endoftext,endoftext_id, num_iters,
             len(test_loader), log=True)

if __name__ == "__main__":
    main()