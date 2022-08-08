from transformers import GPT2Tokenizer, GPT2LMHeadModel,GPT2Config,  AdamW,  Conv1D
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
import importlib
from rouge import Rouge
from model_util import *
from data.util import *
from model import *
import gc
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.manifold import TSNE
import matplotlib
import logging
import argparse
import wandb

matplotlib.use('Agg')
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # set visible devices for debug

def compute_loss(args, device, model, x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, loss_fn, beta):
    input_tokens = input_tokens.to(device)
    target_tokens = target_tokens.to(device)
    mask = mask.to(device)
    x_mask = x_mask.to(device)
    x_tokens = x_tokens.to(device)
    y_mask = y_mask.to(device)
    y_tokens = y_tokens.to(device)
    outputs = model(input_ids=input_tokens, attention_mask=mask, x_mask=x_mask, x_tokens=x_tokens, y_mask=y_mask,
                    y_tokens=y_tokens)
    logits = outputs[0]
    num_logits = logits.size(-1)
    # Perform masking
    if mask is not None:
        mask = mask.type(torch.bool)
        mask = mask.to(device)
        logits = logits.masked_select(mask.unsqueeze(-1))
        target_tokens = target_tokens.masked_select(mask)

    ce_loss = loss_fn(logits.view(-1, num_logits), target_tokens.view(-1))
    kl_loss = outputs[1]
    kl_loss = kl_loss.mean()
    loss = ce_loss.mean() + beta * kl_loss
    return loss, ce_loss, kl_loss


def train_step(args, device, model, optimizer, x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, loss_fn, beta, model_type):
    output = []
    loss, ce_loss, kl_loss = compute_loss(args, device, model, x_mask, x_tokens, y_mask, y_tokens, input_tokens,
                                          target_tokens, mask, loss_fn, beta)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)# max_grad_norm=1.0
    optimizer.step()
    optimizer.zero_grad()
    output.append((loss.item(), ce_loss.mean().item(), kl_loss.item()))
    return output

def top_k_top_p_filtering(logits, top_k=100, top_p=0.95, args = None, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits

def repeat_score(text, ngram=[3, 4, 5, 6]):
    ngram_list = []
    for ng in ngram:
        ngram_list.append([text[idx:idx + ng] for idx in range(len(text) - ng - 1)])

    max_occurs = []
    for ngrams in ngram_list:
        count_result = Counter([' '.join(n) for n in ngrams])
        try:
            max_occurs.append(
                max(count_result.values())
            )
        except:
            pass
    scores = [max_oc / ((len(text) / ngram[idx]) + ngram[idx]) for idx, max_oc in enumerate(max_occurs)]
    return max(scores) if len(scores) >= 1 else 1.0


def sample_sequence(args, model, tokenizer, length, batch_size=None, x_mask=None, x_tokens=None, y_mask=None, y_tokens=None,
                    temperature=1.0, top_k=100, top_p=0.95, device='cuda', sample=True, eos_token=None, model_type='cvae'):
    x_mask = x_mask.to(device)
    x_tokens = x_tokens.to(device)
    y_mask = y_mask.to(device)
    y_tokens = y_tokens.to(device)

    with torch.no_grad():
        if model_type == 'cvae':

            prior_mean, prior_logvar = model.encoder_prior(input_ids=x_tokens, attention_mask=x_mask)[:2]
            z = model.reparameterize(prior_mean, prior_logvar)

            assert not torch.isnan(z).any(), 'sampling sequence get nan z'


        _, mem = model.transformer(input_ids=x_tokens[:, :-1], past_key_values=None, attention_mask=x_mask[:, :-1], representations=z)
        prev = x_tokens[:, -1].view(batch_size, -1)

        output = prev
        probability = torch.tensor([], dtype=z.dtype, device=device)
        if_end = torch.tensor([False] * batch_size, dtype=torch.bool, device=device)

        for i in range(length):# trange
            logits, mem = model.transformer(input_ids=prev, past_key_values=mem, representations=z)

            logits = model.lm_head(logits)
            if model.add_softmax:
                logits_rep = model.lm_head_rep(z)
                logits = logits + logits_rep.unsqueeze(dim=1)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='train')
    # Default parameters are set based on single GPU training
    parser.add_argument('--lr', type=float, default=5e-5)

    parser.add_argument('--lrtimes', type=int, default=20)
    parser.add_argument('--top_k', type=int, default=20) # default = 100
    parser.add_argument('--max_size', type=int, default=200000)# max_eval_dataset_size, train=max_size*2, test=max_size//100
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Whether to run wandb.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--data_type', type=str, default='t0', choices=['t' + str(i) for i in range(9)], help="t: type")
    parser.add_argument('--model_type', type=str, default='cvae', choices=['cvae'])
    parser.add_argument('--iterations', type=int, default=850001)
    parser.add_argument('--dataset', type=str, default='story', choices=['ax', 'yp', 'wp', 'wi', 'roc', 'story'], help="Dataset to use for training")
    parser.add_argument('--warmup', type=int, default=100,
                        help="Amount of iterations to warmup, then decay. (-1 for no warmup and decay)")
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[2],
                        help='batch size per GPU. Lists the schedule.')
    parser.add_argument('--seq-lens', nargs='+', type=int, default=[768],# default=1024
                        help='seq length per sample. Lists the schedule.')
    parser.add_argument('--switch-time', type=float, default=0,
                        help="Percentage of iterations to spend on short sequence training.")
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--out-dir', type=str, default='out')
    parser.add_argument('--load', type=str, help='path to load model from')

    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--no_gpu', action="store_true")
    parser.add_argument('--fp16', action='store_true', help="Train using FP16?")
    parser.add_argument('--fp16_opt_level', default='O0', type=str, required=False)
    # KL cost annealing, increase beta from beta_0 to 1 in beta_warmup steps
    parser.add_argument('--beta_0', default=1.00, type=float)
    # parser.add_argument('--beta_warmup', type=int, default=50000)
    # cyc_vae parameters
    # parser.add_argument('--cycle', type=int, default=101640)
    parser.add_argument('--add_input', action="store_true", default=True)

    parser.add_argument('--add_attn', action="store_true", default=True)#, default=True
    parser.add_argument('--add_softmax', action="store_true")
    parser.add_argument('--attn_proj_vary', action="store_true")

    parser.add_argument('--learn_prior', action="store_true", default=True)

    args = parser.parse_args()# wi.12.proj_vary_beta_cvae
    if args.model_type == 'cvae':
        args.learn_prior = True

    args.group_name = 'origin'
    args.project_name = 'trial_' + args.dataset
    if args.use_wandb:
        wandb.init(config=args, project=args.project_name, entity='entity name', group=args.group_name)

    # GPU
    if not torch.cuda.is_available(): args.no_gpu = True
    gpu = not args.no_gpu
    args.gpu = 0
    device = torch.device(args.gpu if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    # prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu:
        torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # logging
    args.experiment = args.project_name + args.group_name + str(args.lr)
    save_folder = os.path.join(args.out_dir, args.experiment)
    os.makedirs(save_folder, exist_ok=True)
    # t_writer = SummaryWriter(os.path.join(save_folder, 'train'), flush_secs=5)
    # v_writer = SummaryWriter(os.path.join(save_folder, 'val'), flush_secs=5)
    importlib.reload(logging)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # logging.basicConfig(filename=os.path.join(save_folder, 'train.log'),
    #                     level=logging.INFO, format='%(asctime)s--- %(message)s')
    logger.info('\n*******************************************************************************\n')
    logger.info("the configuration:")
    logger.info(str(args).replace(',', '\n'))

    print('Loading models...')
    cache_dir = os.path.join(args.out_dir, 'model_cache')
    os.makedirs(cache_dir, exist_ok=True)
    # Load pre-trained teacher tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
    # Hack to allow tokenizing longer sequences.
    tokenizer.max_len = int(1024)# 1e12
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=cache_dir)
    print('gpt2_params:', num_params(gpt2_model))
    config = GPT2Config().from_pretrained('gpt2')

    if args.attn_proj_vary:
        args.add_attn = True

    VAE = VAEModel(config, add_input=args.add_input, add_attn=args.add_attn, add_softmax=args.add_softmax,
                   attn_proj_vary=args.attn_proj_vary, learn_prior=args.learn_prior)

    init_para_frompretrained(VAE.transformer, gpt2_model.transformer, share_para=True)
    init_para_frompretrained(VAE.encoder, gpt2_model.transformer, share_para=False)
    if args.learn_prior:# True
        init_para_frompretrained(VAE.encoder_prior, VAE.encoder, share_para=True)
        VAE.encoder_prior.averageSelfAttention.attention_weights = VAE.encoder.averageSelfAttention.attention_weights
    VAE.lm_head.weight = gpt2_model.lm_head.weight
    if VAE.add_softmax:
        VAE.lm_head_rep = Conv1D(*gpt2_model.lm_head.weight.size())

    print('VAE_params:', num_params(VAE))
    # 286694400
    if args.load:
        print('Loading model weights...')
        state = torch.load(os.path.join(args.load, 'model_latest.pt'))  # , map_location='cpu' model_latest.pt
        if 'module' in list(state.keys())[0]:  # model_path is data parallel model with attr 'module'
            state_copy = copy.copy(state)
            keys = state_copy.keys()
            for k in keys:
                state[k.replace('module.', '')] = state.pop(k)
        VAE.load_state_dict(state)
        gc.collect()
    print('Done.')

    # fix pre-trained parameters before certain iterations
    tuning_all = False
    for name, parameter in VAE.named_parameters():
        # print((name, parameter.requires_grad))
        new_pars = ['c_z', 'attention_weights', 'mean', 'logvar', 'input_proj', 'attn_proj', 'Nu_fc1', 'Nu_fc2',
                    'lm_head_rep', 'discriminator', 'generator']
        if not any([True if n in name else False for n in new_pars]):
           parameter.requires_grad = False
        else:
            print('train: ' + name)

    print('Setup data...')
    # Batch and sequence length schedule
    assert len(args.batch_sizes) == len(args.seq_lens) == 1

    if args.dataset == 'roc':
        args.data_type = 't0'
        args.batch_sizes = [36]
        args.seq_lens = [256]
        args.iterations = 240000//args.batch_sizes[0]

    elif args.dataset == 'story':

        args.data_type = 't0'
        args.batch_sizes = [16]
        args.seq_lens = [256]
        args.iterations = 120000//args.batch_sizes[0]
    else:
        raise(NotImplementedError)

    tuning_all_after_iters = args.iterations//100
    test_period = args.iterations//20
    eval_period = args.iterations//20
    args.warmup = args.iterations//100

    batch_schedule = list(zip(map(int, args.batch_sizes), map(int, args.seq_lens)))
    assert len(batch_schedule) <= 2, 'Currently not supporting multiple schedule'
    cur_b_schedule = len(batch_schedule) - 1 if args.switch_time == 0 else 0
    print('Batch schedule', batch_schedule)
    train_loader, val_loader, test_loader = prepare_dataset(
        args.data_dir, args.dataset, tokenizer,
        batch_schedule[cur_b_schedule][0], batch_schedule[cur_b_schedule][1],
        batch_schedule[-1][0], batch_schedule[-1][1], batch_schedule[-1][0], batch_schedule[-1][1],
        make_test=True, data_type=args.data_type, max_size=args.max_size,
        datatype=args.datatype, add_prompt=args.add_prompt) #num_workers=args.workers,

    print('Done.')
    # val_loader = test_loader
    ###
    print('Wrapping models and optimizers...')
    # Apply linear scaling rule to increase batch size for short sequence training.
    lr_schedule = switch_schedule(linear_schedule(args), batch_schedule[cur_b_schedule][0] / batch_schedule[-1][0],
                                  int(args.iterations * args.switch_time))
    VAE = VAE.to(device)
    VAE.train()

    args.learning_rate2 = args.lr/args.lrtimes

    optimizer_grouped_parameters = [
        {
            "params": [p for p in VAE.parameters()],
            "lr":args.lr
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, correct_bias=True)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    print('Done.')


    print('************ task name   ' + args.dataset + ' ****************')
    logger.info("Total iteration: %d" % args.iterations)
    e = 0  # number of epoch
    num_iters = 0
    optimizer.zero_grad()
    beta = args.beta_0
    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    def val_step(val_loader, max_val_batches, log=True):
        VAE.eval()
        n_words_bpe = 0
        n_words = 0
        logp_sum = 0.0
        kl_loss_sum = 0.0
        print("Validation loop.         Batches: %d" % len(val_loader))
        print("Validation loop. max_val_batches: %d" % max_val_batches)
        with tqdm(total=min(len(val_loader), max_val_batches)) as pbar:
            for i, (x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask) in enumerate(val_loader):
                with torch.no_grad():
                    loss, ce_loss, kl_loss = compute_loss(args, device, VAE, x_mask, x_tokens, y_mask, y_tokens,
                                                          input_tokens, target_tokens, mask, loss_fn, 1.0)

                if len(target_tokens.size()) == 1:
                    target_tokens = target_tokens.unsqueeze(0)
                n, l = target_tokens.size()

                text = target_tokens[0, :].tolist()
                logprob = ce_loss.tolist()
                assert len(text) == len(logprob) # assert bz=1

                if args.dataset not in ['roc', 'story']:
                    idx = text.index(endoftext)
                    text = text[idx + 1:]
                    logprob = logprob[idx + 1:]

                if endoftext in text:
                    idx = text.index(endoftext)
                    text = text[:idx]
                    logprob = logprob[:idx]

                logp_sum += sum(logprob)
                n_words_bpe += len(text)

                story = [tokenizer.decode(target_tokens[i, :]) for i in range(n)]
                if args.dataset not in ['roc', 'story']:
                    story = [s[s.find("<|endoftext|>") + len("<|endoftext|>"):] for s in story]
                story = [s[:s.find("<|endoftext|>") + len("<|endoftext|>")] if "<|endoftext|>" in s else s for s in
                         story]
                words = sum([len(
                    [t for t in re.split('("|\'|!|\?|\.|,|:| |\n|’|“|”|;|\(|\)|`)', s) if t != ' ' and t != '']) for
                    s in story])
                n_words += words
                kl_loss_sum += kl_loss.item()
                if i > max_val_batches:
                    break
                pbar.update(1)

        loss_bpe = logp_sum / n_words_bpe
        ppl_bpe = round(math.exp(min(logp_sum / n_words_bpe, 100)), 3)
        ppl_word = round(math.exp(min(logp_sum / n_words, 100)), 3)
        kl = kl_loss_sum / len(val_loader)

        logger.info('val loss    : %.4f' % loss_bpe)
        logger.info('val ppl_bpe : %.4f' % ppl_bpe)
        logger.info('val ppl_word: %.4f' % ppl_word)
        logger.info('val   kl    : %.4f' % kl)

        if args.use_wandb and log:
            wandb.log({'eval ppl_bpe': ppl_bpe,  'eval ppl_word': ppl_word,'eval kl': kl, 'eval loss': loss_bpe })
        VAE.train()

    def generate(test_loader, num_iters, max_test_examples, log=True):
        VAE.eval()
        n_samples = 0
        bleu4_sum = 0.0

        dic1 = {}
        dic2 = {}
        distinc1, distinc2 = 0, 0
        all1, all2 = 0, 0

        rouge_scores_values_sum = [0.0] * 3 # 9
        args.nsamples = 1
        args.batch_size = 1
        args.temperature = 0.95
        args.top_p = 0.95
        model_type = args.model_type
        # write samples to file
        samples_file = open(os.path.join(save_folder, 'generate-' + '%07d' % num_iters + '.txt'), 'w', encoding='utf8')

        print("***************test*******************")
        with tqdm(total=min(len(test_loader), max_test_examples)) as pbar:
            for i_test, (x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask) in enumerate(
                    test_loader):
                if i_test >= max_test_examples:
                    break
                length = -1
                if args.dataset in ['roc', 'story']:
                    length = 64
                elif args.dataset == 'wi':
                    length = 512
                elif args.dataset == 'wp':
                    length = 768
                args.generate_length = length

                eff_samples = []
                n, l = target_tokens.size()
                storys = [tokenizer.decode(target_tokens[i, :]) for i in range(n)]
                if args.dataset not in ['roc', 'story']:
                    storys = [s[s.find("<|endoftext|>") + len("<|endoftext|>"):] for s in storys]

                storys_str = [s[:s.find("<|endoftext|>") + len("<|endoftext|>")] if "<|endoftext|>" in s else s for s in
                              storys]

                for _ in range(args.nsamples // args.batch_size):
                    out, _ = sample_sequence(
                        args,
                        model=VAE,
                        tokenizer=tokenizer,
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
                        eos_token=tokenizer.encoder['<|endoftext|>'],
                        model_type=model_type
                    )
                    out = out.tolist()

                    # extract story, check metrics
                    for i in range(len(out)):
                        text = out[i]
                        text = text[text.index(endoftext) + 1:]
                        if endoftext in text:
                            idx = text.index(endoftext)
                            text = text[:idx]
                        text = tokenizer.decode(text).strip()
                        if args.dataset in ['roc', 'story']:
                            text_word = text.split(' ')
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
                            ref[-1] = ref[-1].strip('<|endoftext|>')
                            sf = SmoothingFunction()
                            bleu4 = sentence_bleu(ref, text.split(), smoothing_function=sf.method7)

                            rouge = Rouge()
                            rouge_scores = rouge.get_scores(text, storys_str[i].strip('<|endoftext|>'))
                            rouge_scores_values = []
                            for i in rouge_scores[0].keys():
                                if i != 'rouge-l':
                                    rouge_scores_values.append(rouge_scores[0][i]['r'])
                                else:
                                    rouge_scores_values.append(rouge_scores[0][i]['f'])
                            # rouge_scores_values = [rouge_scores[0][k]['r'] for k in rouge_scores[0].keys()]
                            bleu4_sum += bleu4
                            rouge_scores_values_sum = [v1 + v2 for v1, v2 in
                                                       zip(rouge_scores_values_sum, rouge_scores_values)]
                            n_samples += 1
                        except:
                            bleu4 = 0.0
                            rouge_scores = [{'rouge-1': {'r': 0.0},
                                             'rouge-2': {'r': 0.0},
                                             'rouge-l': {'f': 0.0}}]
                        eff_samples.append((text, bleu4, rouge_scores))
                    pbar.update(1)

                for i in range(len(eff_samples)):
                    samples_file.write("=" * 50 + " SAMPLE " + str(i_test) + " " + "=" * 50)
                    samples_file.write('\n' * 2)

                    samples_file.write("=" * 40 + " Outlines  " + "=" * 40)
                    samples_file.write('\n' * 2)
                    if args.dataset not in ['roc', 'story']:
                        samples_file.write(tokenizer.decode(x_tokens[i, :][x_mask[i, :] == 1].tolist()))
                    else:
                        samples_file.write(tokenizer.decode(x_tokens[i, :].tolist()))
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
        logger.info("Test complete with %05d samples.", n_samples)
        logger.info("Iteration completed: %d" % num_iters)

        bleu4 = round(bleu4_sum / n_samples, 5)
        rouge_scores_values = [round(r / n_samples, 3) for r in rouge_scores_values_sum]
        print(' bleu-4:', bleu4*100)
        print(' rouge :', rouge_scores_values)

        if args.use_wandb and log:
            if args.dataset in ['roc', 'story']:
                print("distinc1: %.5f" % float(distinc1 / all1)*100)
                print("distinc2: %.5f" % float(distinc2 / all2)*100)
                wandb.log({'valid_samples': n_samples, 'bleu4': bleu4*100, 'rouge1 r': rouge_scores_values[0], 'rouge2 r': rouge_scores_values[1],
                           'rougel f': rouge_scores_values[2], 'distinc1': float(distinc1 / all1)*100,
                           'distinc2': float(distinc2 / all2)*100})
            else:
                wandb.log({'bleu4': bleu4*100, 'rouge1 r': rouge_scores_values[0], 'rouge2 r': rouge_scores_values[1],
                           'rougel f': rouge_scores_values[2]})

        logger.info(' bleu-4: %f', bleu4)
        logger.info(' rouge : %s', str(rouge_scores_values))
        VAE.train()

    # test_plot(test_loader, num_iters)
    print('first eval and generate, before training...')
    val_step(val_loader, 10, False)
    generate(test_loader, num_iters, 5, False)
    # torch.save(VAE.state_dict(), os.path.join(save_folder, 'model_' + '{:07d}'.format(num_iters) + '.pt'))

    VAE.train()
    while num_iters < args.iterations:
        # Run epoch
        st = time.time()
        # Training
        print('Training loop. Batches:', len(train_loader))
        logger.info('\n----------------------------------------------------------------------')
        logger.info("Training loop.   Batches: %d" % len(train_loader))

        # train_iter = iter(train_loader); x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask = next(train_iter)
        with tqdm(total=args.iterations) as pbar:
            for i, (x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask) in enumerate(train_loader):
                # if num_iters % args.cycle >= args.cycle - args.beta_warmup:
                #     beta = min(1.0, beta + (1. - args.beta_0) / args.beta_warmup)
                if not tuning_all and num_iters >= tuning_all_after_iters:
                    for name, parameter in VAE.named_parameters():
                        # print((name, parameter.requires_grad))
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
                    # test_plot(test_loader, num_iters)
                    val_step(val_loader, len(val_loader))
                if (num_iters+1) % test_period == 0:
                    generate(test_loader, num_iters, len(test_loader))

        if not end:
            e += 1
            logger.info("Training loop. The ith epoch completed: %d" % e)
    # torch.save(VAE.state_dict(), os.path.join(save_folder, 'model_latest.pt'))
    print('Training complete.')
    logger.info("Training complete.")
    print('final eval and test.')
    val_step(val_loader, len(val_loader))
    generate(test_loader, num_iters, len(test_loader))

if __name__ == "__main__":
    main()




