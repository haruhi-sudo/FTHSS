from peft import (
    TaskType, 
    get_peft_model,
    PeftModel,
    PeftConfig,
    MultitaskPromptTuningConfig,
    MultitaskPromptTuningInit,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptTuningInit
)

def parse_peft_args(parser):
    # custom_peft的特殊参数
    parser.add_argument("--use_custom_peft", action="store_true", help="Use custom PEFT model")
    parser.add_argument("--output_order", type=str, default="A") # 要生成的内容，在串行任务中的顺序。按照A，B，C，D...的顺序
    parser.add_argument("--pre_prefix_path", type=str, nargs='+', default=[]) # 已经训练好的前置模型的路径。比如，output_order为C，则必须提供A和B的模型路径
    parser.add_argument("--mode", type=str, default="") 

    parser.add_argument("--peft_type", type=str, default="prompt_tuning")
    parser.add_argument("--num_virtual_tokens", type=int, default=50)
    parser.add_argument("--prompt_tuning_init_text", type=str, default="Sumarize the following text: ")
    # parser.add_argument("--modules_to_save", type=str, default="all")
    return parser

def get_peft_config(args):
    if args.peft_type == "prompt_tuning":
        peft_config = PromptTuningConfig(
            num_virtual_tokens=args.num_virtual_tokens,
            inference_mode=False, 
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            tokenizer_name_or_path=args.tokenizer_name if args.tokenizer_name is not None else args.model_name_or_path,
            prompt_tuning_init_text=args.prompt_tuning_init_text,
        )
    elif args.peft_type == "prefix_tuning":
        peft_config = PrefixTuningConfig(
            num_virtual_tokens=args.num_virtual_tokens,
            inference_mode=False, 
            task_type=TaskType.CAUSAL_LM,
            tokenizer_name_or_path=args.tokenizer_name if args.tokenizer_name is not None else args.model_name_or_path,
        )
    return peft_config
