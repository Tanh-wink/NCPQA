import argparse


def getArgs():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default='../output_model', type=str, required=False,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--policies_file", default='../data/policies_context.pkl', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--train_file", default='../data/train_data.pkl', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default='../data/test_with_doc_top5.pkl', type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_answer_length", default=300, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_valid", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=True, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=64, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=512, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=100,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--n_best_size",
                        default=20, type=int,
                        help="Whether to evaluate dev set.")
    parser.add_argument("--null_score_diff_threshold",
                        default=0.1, type=float,
                        help="Whether to evaluate dev set.")
    parser.add_argument("--doc_stride",
                        default=128, type=int,
                        help="Whether to evaluate dev set.")
    parser.add_argument("--version_2_with_negative",
                        default=False,
                        action='store_true',
                        help="Whether to evaluate dev set.")
    parser.add_argument("--verbose_logging",
                        default=False,
                        action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                        "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument(
        "--model_type",
        default="albert",
        type=str,
        help="Model type selected in the list: bert, xlnet, albert"
    )
    parser.add_argument(
        "--model_name_or_path",
        default="../albert_pretrain_model/albert_chinese_base",
        type=str,
        help="Path to pre-trained model" ,
    )
    parser.add_argument(
        "--model_info",
        default="base_addBad",
        type=str,
        help="the detail of model",
    )
    parser.add_argument(
        "--our_pretrain_model",
        default="",
        type=str,
        help="continue pretrain model path",
    )
    parser.add_argument(
        "--valid_step",
        default=20,
        type=int,
        help="the frequence of doing valid",
    )

    # for predict
    parser.add_argument(
        "--do_bm25",
        default=False,
        action='store_true',
        help="use bm25 to filter sub_doc"
    )
    parser.add_argument(
        "--topRate",
        default=0.4,
        type=float,
        help="the model dir for prediction",
    )
    parser.add_argument(
        "--predict_model_path",
        default="/home/th/PycharmProjects/python3.5/NCPQA/output_model/albert_base_addBad/2020-03-22@11_57_57",
        type=str,
        help="the model dir for prediction",
    )



    # ../albert_pretrain_model/albert_chinese_base
    # ../chinese_wwm_pytorch
    return parser.parse_args()