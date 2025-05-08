
import os

import torch
from datasets import load_dataset
from transformers import (
    ByT5Tokenizer,
    T5ForConditionalGeneration,
    T5Config,
    TrainingArguments,
    Trainer,
)

from t5_data_collator import DataCollatorForT5MLM, compute_t5_input_and_target_lengths

MAX_SEQ_LENGTH = 1024
NOISE_DENSITY = 0.15
MEAN_NOISE_SPAN_LENGTH = 20.0

def train_byt5_on_mc4():
    """
    This function outlines the steps to train a ByT5 model on the
    English portion of the mC4 corpus, configured for multi-GPU training
    on the full dataset.
    """

    expanded_tokenizer_length, computed_target_length = compute_t5_input_and_target_lengths(
        inputs_length=MAX_SEQ_LENGTH,
        noise_density=NOISE_DENSITY,
        mean_noise_span_length=MEAN_NOISE_SPAN_LENGTH
    )

    config = T5Config(
        vocab_size=384,
        d_model=768,
        d_ff=2048,
        num_layers=6,
        num_decoder_layers=2,
        num_heads=8,
        d_kv=64,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="gated-gelu",
        tie_word_embeddings=True,
        is_encoder_decoder=True,
        is_gated_act=True,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        use_cache=False
    )

    # 1. Define Model and Tokenizer
    model_name = "google/byt5-small"
    tokenizer = ByT5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Verify model configuration
    print(f"Loaded model config for {model_name}:")
    print(f"  Encoder layers: {config.num_layers}")
    print(f"  Decoder layers: {config.num_decoder_layers}")
    print(f"  d_model: {config.d_model}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  Target params (approx): {total_params} and Trainable params (approx): {trainable_params}")

    # Gradient Checkpointing to save memory for larger models/batches
    model.gradient_checkpointing_enable()

    def preprocess_function(examples):
        tokenized_output = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=expanded_tokenizer_length,
            return_attention_mask=True,
        )
        return tokenized_output

    # 2. Load the mC4 Dataset (English portion - Full Stream)
    print("Loading mC4 dataset (English) for streaming...")
    try:
        dataset = load_dataset("allenai/c4", "en", streaming=True, trust_remote_code=True)
        eng_ds = dataset["train"].map(preprocess_function, batched=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Dataset stream obtained.")


    # 4. Data Collator
    data_collator = DataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=NOISE_DENSITY,
        mean_noise_span_length=MEAN_NOISE_SPAN_LENGTH,
        input_length=MAX_SEQ_LENGTH,
        target_length=computed_target_length,
        pad_token_id=model.config.pad_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # 5. Set up Training Arguments for Multi-GPU
    output_dir = "./byt5_mc4_english_full_pretrain_1024"

    num_cpus = os.cpu_count()
    dataloader_workers = min(max(1, (num_cpus // torch.cuda.device_count()) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else num_cpus // 2 if num_cpus else 4), 32)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        # --- Batch Size & Epochs ---
        per_device_train_batch_size=90,
        max_steps=1_000_000,
        # --- Logging, Saving & Checkpointing ---
        logging_dir='./logs_full_pretrain_1024',
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=3,
        # --- Optimizer & Scheduler ---
        learning_rate=1e-4,
        lr_scheduler_type="linear",
        warmup_ratio=0.01,
        weight_decay=0.01,
        # --- Performance & Hardware ---
        bf16=True,
        dataloader_num_workers=dataloader_workers,
        dataloader_pin_memory=True,
        report_to=["wandb"],
    )

    print("Training arguments configured for multi-GPU full dataset training.")
    print(f"  Model: {model_name}")
    print(f"  Max Sequence Length: {MAX_SEQ_LENGTH}")
    print(f"  Max Training Steps: {training_args.max_steps}")

    current_global_batch_tokens = (torch.cuda.device_count() if torch.cuda.is_available() else 1) * \
                                  training_args.per_device_train_batch_size * \
                                  (
                                      training_args.gradient_accumulation_steps if training_args.gradient_accumulation_steps else 1) * \
                                  MAX_SEQ_LENGTH
    print(f"  Current effective global batch size: {current_global_batch_tokens} tokens/bytes")
    print(f"  ByT5 paper pre-training global batch size: {2 ** 20} tokens/bytes (target for full replication)")
    print(f"  Dataloader workers per process: {training_args.dataloader_num_workers}")

    # 6. Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=eng_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 7. Start Training
    print("Starting full-scale training on mC4 English...")
    try:
        trainer.train()
    except Exception as e:
        print(f"An error occurred during training: {e}")

    print("Training finished.")

    # 8. Save the Final Model and Tokenizer
    print(f"Saving final model and tokenizer to {output_dir}_final...")
    final_save_path = f"{output_dir}_final"
    os.makedirs(final_save_path, exist_ok=True)
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    config.save_pretrained(final_save_path)
    print(f"Model and tokenizer saved to {final_save_path}")

    print("Script execution finished.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        exit()

    print(
        f"CUDA is available. Number of GPUs visible to this process: {torch.cuda.device_count()}")

    train_byt5_on_mc4()
