import logging
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from trl import ModelConfig, ScriptArguments, TrlParser
from trl.rewards import accuracy_reward, think_format_reward
from trl.trainer import GRPOConfig, GRPOTrainer


NUM_PROC = 100
DEBUG = False
if DEBUG:
    logging.basicConfig(level=logging.INFO)


@dataclass
class TokenizerConfig:
    tokenizer_name_or_path: str | None = None


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig, TokenizerConfig))
    script_args, training_args, model_args, tokenizer_args = parser.parse_args_and_config()
    ################
    # Model & Processor
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        trust_remote_code=True,
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    tokenizer_name_or_path = (
        tokenizer_args.tokenizer_name_or_path
        if tokenizer_args.tokenizer_name_or_path
        else model_args.model_name_or_path
    )
    processor = AutoTokenizer.from_pretrained(tokenizer_name_or_path, padding_side="left", trust_remote_code=True)

    ################
    # Dataset
    ################
    print("Begin processing dataset")
    train_dataset = load_dataset(
        "json",
        data_files=script_args.dataset_name,
        split="train",
        streaming=False,
        num_proc=NUM_PROC,
    )

    SYSTEM_PROMPT = (
        "A conversation between user and assistant. The user asks a question, and the assistant solves it. The "
        "assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think></think> tags, i.e., <think>\nThis is my "
        "reasoning.\n</think>\nThis is my answer."
    )

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
        }

    train_dataset = train_dataset.map(make_conversation, num_proc=NUM_PROC)

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        processing_class=processor,
        args=training_args,
        reward_funcs=[think_format_reward, accuracy_reward],
        train_dataset=train_dataset,
        peft_config=None,
    )
    # validate_accelerator_config(trainer.accelerator)
    trainer.accelerator.print(f"Begin training {trainer._name} for model `{model_args.model_name_or_path}`")
    resume_from_checkpoint = training_args.resume_from_checkpoint if training_args.resume_from_checkpoint else None
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint at '{resume_from_checkpoint}'")
    trainer.train(resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
