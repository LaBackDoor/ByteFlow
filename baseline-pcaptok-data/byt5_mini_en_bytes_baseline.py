import os
import torch
from datasets import load_dataset, Dataset, IterableDataset, interleave_datasets
from transformers import (
    T5ForConditionalGeneration,
    T5Config,
    TrainingArguments,
    Trainer,
)
from hybrid_byt5_tokenizer import HybridByT5PCAPTokenizer
from t5_data_collator import DataCollatorForT5MLM, compute_t5_input_and_target_lengths
import itertools
import random

# Constants
MAX_SEQ_LENGTH = 1024
NOISE_DENSITY = 0.15
MEAN_NOISE_SPAN_LENGTH = 20.0


def train_byt5_on_mc4_and_pcap():
    # [existing code for calculating lengths, initializing model and tokenizer]

    # Calculate input and target lengths based on the denoising objective
    expanded_tokenizer_length, computed_target_length = compute_t5_input_and_target_lengths(
        inputs_length=MAX_SEQ_LENGTH,
        noise_density=NOISE_DENSITY,
        mean_noise_span_length=MEAN_NOISE_SPAN_LENGTH
    )

    print(f"Input length: {expanded_tokenizer_length}, Target length: {computed_target_length}")

    # [config, model initialization, etc.]
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

    # Initialize our enhanced PCAP tokenizer
    tokenizer = HybridByT5PCAPTokenizer(pcap_vocab_size=280)

    # Initialize the model and resize token embeddings to match tokenizer
    model = T5ForConditionalGeneration(config)
    model.resize_token_embeddings(len(tokenizer))

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # Create preprocessing function for text data
    def preprocess_text(examples):
        tokenized_output = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=expanded_tokenizer_length,
            return_attention_mask=True,
        )
        return tokenized_output

    # Load MC4 English dataset (streaming)
    print("Loading MC4 English dataset (streaming)...")
    eng_dataset = load_dataset("allenai/c4", "en", streaming=True, trust_remote_code=True)
    eng_ds = eng_dataset["train"].map(preprocess_text, batched=True)
    print("MC4 English dataset loaded successfully")

    # Create a streaming PCAP dataset
    pcap_dir = "../data/flows"  # Update this path to your PCAP directory

    # Function to iterate through PCAP files and yield them one by one
    def pcap_iterator():
        pcap_files = [os.path.join(pcap_dir, f) for f in os.listdir(pcap_dir) if f.endswith(".pcap")]
        # Shuffle the files to avoid any potential bias
        random.shuffle(pcap_files)

        for i, pcap_path in enumerate(pcap_files):
            try:
                # Process the PCAP file with the tokenizer
                input_ids = tokenizer.tokenize_text_with_pcap("", pcap_path)

                # Truncate if needed
                if len(input_ids) > expanded_tokenizer_length:
                    input_ids = input_ids[:expanded_tokenizer_length]

                # Create attention mask
                attention_mask = [1] * len(input_ids)

                # Pad if needed
                padding_length = expanded_tokenizer_length - len(input_ids)
                if padding_length > 0:
                    input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
                    attention_mask = attention_mask + [0] * padding_length

                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }

                # Print progress occasionally
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1} PCAP files")

            except Exception as e:
                print(f"Error processing {pcap_path}: {e}")
                continue

    # Create an IterableDataset for PCAP files
    pcap_ds = IterableDataset.from_generator(pcap_iterator)

    # Interleave the two streaming datasets
    print("Creating interleaved dataset...")
    train_dataset = interleave_datasets(
        [eng_ds, pcap_ds],
        probabilities=[0.8, 0.2],
        stopping_strategy="first_exhausted"
    )

    # Initialize the data collator
    data_collator = DataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=NOISE_DENSITY,
        mean_noise_span_length=MEAN_NOISE_SPAN_LENGTH,
        input_length=MAX_SEQ_LENGTH,
        target_length=computed_target_length,
        pad_token_id=model.config.pad_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )


    # Configure output directory
    output_dir = "./byt5_mini_mc4_pcap_1024"

    # Calculate optimal number of dataloader workers
    num_cpus = os.cpu_count()
    dataloader_workers = min(
        max(1, (
                    num_cpus // torch.cuda.device_count()) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else num_cpus // 2 if num_cpus else 4),
        32
    )

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        # Batch size and training length
        per_device_train_batch_size=90,
        max_steps=1_000_000,
        # Logging and saving
        logging_dir='./logs_byt5_mini_mc4_pcap',
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=3,
        # Optimizer and learning rate
        learning_rate=1e-4,
        lr_scheduler_type="linear",
        warmup_ratio=0.01,
        weight_decay=0.01,
        # Hardware optimization
        bf16=True,
        dataloader_num_workers=dataloader_workers,
        dataloader_pin_memory=True,
        # Reporting
        report_to=["wandb"],
    )

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Start training
    print("Starting training...")
    try:
        trainer.train()
        print("Training completed successfully.")
    except Exception as e:
        print(f"Error during training: {e}")

    # Save the final model and tokenizer
    final_save_path = f"{output_dir}_final"
    print(f"Saving final model and tokenizer to {final_save_path}")
    os.makedirs(final_save_path, exist_ok=True)
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    config.save_pretrained(final_save_path)
    print(f"Model and tokenizer saved to {final_save_path}")


if __name__ == "__main__":
    # Check for CUDA
    if not torch.cuda.is_available():
        print("CUDA is not available. This training script requires GPU acceleration.")
        exit()

    print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
    print(f"CUDA devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

    # Run the training
    train_byt5_on_mc4_and_pcap()