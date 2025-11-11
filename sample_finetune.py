import sys
import logging

import datasets
from datasets import load_dataset
import torch
import transformers
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

"""
A simple example on using SFTTrainer to finetune SlimMoE models. For
a more advanced example, please follow HF alignment-handbook/scripts/run_sft.py.
The script can be run on a single 80GB A100 or later generation GPU. Here are some suggestions on
further reducing memory consumption:
    - use deepspeed zero3
    - use gradient checkpointing
Please follow these steps to run the script:
1. Install dependencies: 
    conda install -c conda-forge accelerate
    pip3 install -i https://pypi.org/simple/ bitsandbytes
    pip3 install peft trl transformers datasets
    pip3 install einops flash_attn torchao
2. Run the code:
    python sample_finetune.py
"""

logger = logging.getLogger(__name__)


###################
# Hyper-parameters
###################
training_config = {
    "bf16": True,
    "do_eval": False,
    "optim": "adamw_torch_8bit",
    "learning_rate": 5.0e-06,
    "log_level": "info",
    "logging_steps": 20,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 1,
    "max_steps": -1,
    "output_dir": "./checkpoint_dir",
    "overwrite_output_dir": True,
    "per_device_eval_batch_size": 1,
    "per_device_train_batch_size": 1,
    "remove_unused_columns": True,
    "save_steps": 100,
    "save_total_limit": 1,
    "seed": 0,
    "gradient_checkpointing": False,
    # "gradient_checkpointing_kwargs":{"use_reentrant": False},
    "gradient_accumulation_steps": 1,
    "warmup_ratio": 0.2,
    "max_length": 4096,
    "dataset_text_field": "text",
    "packing": True,
    }

train_conf = SFTConfig(**training_config)

###############
# Setup logging
###############
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = train_conf.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process a small summary
logger.warning(
    f"Process rank: {train_conf.local_rank}, device: {train_conf.device}, n_gpu: {train_conf.n_gpu}"
    + f" distributed training: {bool(train_conf.local_rank != -1)}, 16-bits training: {train_conf.fp16}"
)
logger.info(f"Training/evaluation parameters {train_conf}")


################
# Model Loading
################
checkpoint_path = "microsoft/Phi-tiny-MoE-instruct"
model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
    torch_dtype=torch.bfloat16,
    device_map=None
)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'


##################
# Data Processing
##################
def apply_chat_template(
    example,
    tokenizer,
):
    messages = example["messages"]
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    return example

raw_dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
train_dataset = raw_dataset["train_sft"]
test_dataset = raw_dataset["test_sft"]
column_names = list(train_dataset.features)

processed_train_dataset = train_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=10,
    remove_columns=column_names,
    desc="Applying chat template to train_sft",
)

processed_test_dataset = test_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=10,
    remove_columns=column_names,
    desc="Applying chat template to test_sft",
)


###########
# Training
###########
trainer = SFTTrainer(
    model=model,
    args=train_conf,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_test_dataset,
    processing_class=tokenizer,
)
train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()


#############
# Evaluation
#############
tokenizer.padding_side = 'left'
metrics = trainer.evaluate()
metrics["eval_samples"] = len(processed_test_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


# ############
# # Save model
# ############
trainer.save_model(train_conf.output_dir)