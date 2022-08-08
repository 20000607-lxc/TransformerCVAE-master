import argparse

def get_argparse():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--use_wandb", action="store_true", default=False,
                        help="Whether to run wandb.")
    parser.add_argument("--generated_pipeline", action="store_true", default=False,
                        help="Whether to use generate function.")
    parser.add_argument("--model_type", default='no_type', type=str, #required=True,
                        help="Model type",
                        choices=['gpt2_loop', 'beam_search', 'bare_gpt2_loop',
                                 'gpt2_copy_loop', 'joint_train', 'init_four_prompt_encoder',
                                 'with_description', 'new_pipeline', 'with_filter', 'no_type'])
    parser.add_argument("--filling_value", type=int, default=-1000,
                        help="Log every X updates steps.",
                        choices=[0, -1000, -50])
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument('--cuda', type=int, default=3, help='Avaiable GPU ID')
    parser.add_argument("--remove_all_after_end", action="store_true", default=True,
                        help="Whether to remove all the tokens after 50256, if set it false, means only remove 50256 .")
    parser.add_argument("--negative_sample_step", type=int, default=1,
                        help="negative_sample_step.",
                        choices=[1])
    parser.add_argument("--output_file_name", default="demo.json", type=str, #required=True,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--group_name", default="lxc", type=str,
                        help="group name on wandb, can change into your name! ")
    parser.add_argument("--optimizer_choice", default="adamW", type=str,
                        help="optimizer choice!  ",
                        choices=['adafactor', 'adamW'])
    parser.add_argument("--tokenizer_name", default='gpt2_get_all_entity', type=str,
                        help=" 'gpt2_for_copy', Pretrained tokenizer name or path if not the same as model_name,",
                        choices=['gpt2_get_all_entity', 'gpt2_for_copy'])
    parser.add_argument("--use_extend_vocab", action="store_true", default=False,
                        help="Whether to add [BOS] token to gpt2 model  .")
    parser.add_argument("--use_discrete",  default=True, action="store_true",
                        help="Whether to use the discrete prompt .")
    parser.add_argument("--fine_tune", default=True, action="store_true",
                        help="Whether to perform fine tuning in gpt2.")
    parser.add_argument("--print_results", default=True, action="store_true",
                        help="Whether to print results during evaluation and test.")
    parser.add_argument("--output_file_dir", default='output_files/new_ontonote/', type=str, #required=True,
                        help="The output directory where the model predictions and checkpoints will be written,"
                             "choose from ['output_files/ontonote/', 'output_files/conll/'].",
                        choices=['output_files/new_ontonote/', 'output_files/new_conll/'])
    parser.add_argument("--save_model", default=False, action="store_true",
                        help="Whether to save the model checkpoints, currently,\ no need to save the checkpoints.")
    # the default numbers are for debug, do set your own limit!
    parser.add_argument("--train_limit", default=1000000, type=int,
                        help="the total lines load from train.text(notice not the number of examples)")
    parser.add_argument("--eval_limit", default=100000, type=int,
                        help="the total lines load from dev.text(notice not the number of examples)")
    parser.add_argument("--test_limit", default=100000, type=int,
                        help="the total lines load from test.text(notice not the number of examples)")
    parser.add_argument('--markup', default='bio', type=str,
                        choices=['bio'])
    parser.add_argument("--note", default='', type=str,
                        help="the implementation details to remind ")
    parser.add_argument("--max_len", default=32, type=int,
                        help="the max len of entities， in conll2003, average_len_of_entity < 8 ")
    parser.add_argument("--assume_true_length", default=1, type=int,
                        help="if the the same ids in pred entity and the true entity have the length of "
                             "assume_true_length, hit1=hit1+1 ")
    parser.add_argument("--template", default='1', type=str, #required=True,
                        help="prompt size, choose from the list:['1','2'] or you can modify the template in "
                             "run_ner_xxx.py by changing TEMPLATE_CLASSES ")
    parser.add_argument("--task_name", default='ontonote', type=str, #required=True,
                        help="The name of the task to train selected in the list: [conll2003', 'ontonote', ] ",
                        choices=['conll2003', 'ontonote', 'conll2003_mrc', 'ontonote4', 'genia'])
    parser.add_argument("--data_dir", default='datasets/ontonotes', type=str, #required=True,
                        help="The input data dir. choose from ['datasets/ontonotes', 'datasets/conll2003_bio'.",
                        choices=['datasets/ontonotes', 'datasets/conll2003_bio', 'datasets/conll03_mrc',
                                 'datasets/ontonote4', 'datasets/genia_mrc'])
    parser.add_argument("--model_name_or_path", default='gpt2', type=str, #required=True,
                        help="Path to pre-trained model or shortcut name , ""I only used: ['gpt2']" )
    parser.add_argument("--learning_rate", default=5e-5, type=float,#bert default = 5e-5
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,#bert default =  0.01
                        help="Weight decay if we apply some.")
    parser.add_argument("--output_dir", default='outputs/genia/gpt2', type=str, #required=True,
                        help="The output directory where the model predictions and checkpoints will be written,"
                             "choose from ['outputs/ontonote/gpt2', 'outputs/conll/gpt2'].", )
    parser.add_argument("--num_train_epochs", default=20, type=float,
                        help="Total number of training epochs to perform.")
    # Other parameters
    parser.add_argument('--loss_type', default='ce', type=str,
                        choices=['lsr', 'focal', 'ce'])
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3", )
    parser.add_argument("--train_max_seq_length", default=64, type=int,#default = 128,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--eval_max_seq_length", default=64, type=int,#default = 128,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--do_train", action="store_true", default=True,
                        help="Whether to run training.")
    parser.add_argument("--evaluate_and_test_during_training", action="store_true", default=True,
                        help="Whether to run evaluation during training at each logging step and run test at "
                             "the last epoch by each logging step .", )
    parser.add_argument("--do_eval_with_saved_model", action="store_true", default=False,
                        help="Whether to run eval on the dev set with saved models .")
    parser.add_argument("--do_predict_with_saved_model", action="store_true", default=False,
                        help="Whether to run predictions on the test set with saved models.")
    parser.add_argument("--do_lower_case", action="store_true", default=False,
                        help="Set this flag if you are using an uncased model.")
    # adversarial training
    parser.add_argument("--do_adv", action="store_true", default=False,
                        help="Whether to adversarial training.")
    parser.add_argument('--adv_epsilon', default=1.0, type=float,
                        help="Epsilon for adversarial.")
    parser.add_argument('--adv_name', default='word_embeddings', type=str,
                        help="name for adversarial layer.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linea learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true", default = False,
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number", )
    parser.add_argument("--predict_checkpoints", type=int, default=0,
                        help="predict checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true", default=True,
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true", default=True,
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html", )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    return parser

    # parser.add_argument("--extend_number", type=int, default=11000,
    #                     help="the max id of extend vocabulary for conll2003: it is 61157-50256.")
    # parser.add_argument("--use_filter",  default=False, action="store_true",
    #                     help="Whether to use the filter to limit the generated entity ids in the input sequence ids")
    # parser.add_argument("--hyper_param", default=10, type=int,
    #                     help="the hyper_param for filter, can choose any positive number  ")
    # parser.add_argument("--compute_copy_after_generation",  default=False, action="store_true",
    #                     help="Whether to compute the copy logits after generation ")
    # parser.add_argument("--compute_copy_every_step", default=False, action="store_true",
    #                     help="Whether to compute the copy logits every step.")
    # parser.add_argument("--use_encoder", default=False, action="store_true",
    #                     help="Whether to use the copy mechanism from Pointer.")
    # parser.add_argument('--former_label', action="store_true", default=False,
    #                     help='whether to use the former label')
    # parser.add_argument('--label_all_tokens', action="store_true", default=False,
    #                     help='whether to label all the tokens, otherwise will label split tokens, '
    #                     'choose label on!(label off is not right) ,(except for the first part in a word) with -100')