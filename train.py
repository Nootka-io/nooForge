import logging
import torch
import json
import argparse
import transformers
from hf_args import ModelArguments, DataArguments, TrainingArguments
from transformers import (
    Trainer,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig, 
    get_peft_model
)
from datasets import load_from_disk
from torch.utils.data.dataloader import DataLoader
from utils import print_gpu_utilization
from utils.flash_attention_patch import replace_attn_with_flash_attn, forward
from utils.save_peft_callback import SavePeftModelCallback


logger = logging.getLogger(__name__)


def find_all_linear_names(args, model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def main():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments
    ))
    (
        model_args, 
        data_args, 
        training_args, 
        extra_args
    ) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    
    # replace attention
    if torch.cuda.get_device_capability()[0] >= 8:
        print("Using flash attention")
        replace_attn_with_flash_attn()
        use_flash_attention = True
    
    # get the model
    print(f'loading base model {args.model_name_or_path}...')        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        # load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map='auto'  # ToDo: set to none for deepspeed
    )
    
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=False,
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
        trust_remote_code=args.trust_remote_code,
    )
    
    model.resize_token_embeddings(len(tokenizer))  # ToDo: can this be moved to creation of the custom tokenizer?
    
    # model = prepare_model_for_kbit_training(model)

    # loads the LoraConfigs or checkpoints
    print(f'adding LoRA modules...')
    modules = find_all_linear_names(args, model)
    config = LoraConfig(
        r = args.lora_r,
        lora_alpha = args.lora_alpha,
        target_modules = modules,
        lora_dropout = args.lora_dropout,
        bias = "none",
        task_type = "CAUSAL_LM",
        modules_to_save=['embed_tokens', 'lm_head'],  # Test to see if it works and helps the embedding
    )
    model = get_peft_model(model, config)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    model.print_trainable_parameters()
    print_gpu_utilization()
    
    # load the dataset
    ds = load_from_disk(args.dataset)
    if training_args.do_eval:
        ds = ds.train_test_split(test_size=0.1)
    ds = ds.with_format('torch')
    
    training_args.bf16 = True
    
    # do the training
    trainer = Trainer(
        model = model, 
        train_dataset = ds['train'] if training_args.do_eval else ds,
        eval_dataset = ds['test'] if training_args.do_eval else None,
        args = training_args,        
    )
    
    # Callbacks
    trainer.add_callback(SavePeftModelCallback)
    
    print(f'beginning training...')
    result = trainer.train()
    trainer.save_state()
    

if __name__ == "__main__":
    main()