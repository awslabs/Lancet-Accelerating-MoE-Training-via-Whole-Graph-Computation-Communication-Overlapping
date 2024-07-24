import torch
import os

def load_wikitext2(batch_size, seq_len):
    """Download (if needed) and load WikiText-2 dataset."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    max_len = seq_len
    print("Loading WikiText-2...")
    data_dir_from_env = os.environ.get("RAF_DATASET_DIR")
    if data_dir_from_env:
        data_dir = data_dir_from_env
    else:
        data_dir = "/root/wikitext"
        if not os.path.exists(data_dir):
            # download from huggingface
            data_dir = "wikitext"
    datasets = load_dataset(data_dir, "wikitext-2-raw-v1")

    tokenizer_dir_from_env = os.environ.get("RAF_TOKENIZER_DIR")
    if tokenizer_dir_from_env:
        tokenizer_dir = tokenizer_dir_from_env
    else:
        tokenizer_dir = "/root/gpt2_tokenizer"
        if not os.path.exists(tokenizer_dir):
            # download from huggingface
            tokenizer_dir = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = datasets.map(
        tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported
        # it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // max_len) * max_len
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_len] for i in range(0, total_length, max_len)]
            for k, t in concatenated_examples.items()
        }
        result["input_ids"] = torch.LongTensor(result["input_ids"])
        result["labels"] = result["input_ids"].clone().numpy()
        if result["input_ids"].shape[0] == 0:
            return result

        # create random array of floats in equal dimension to input_ids
        rand = torch.rand(result["input_ids"].shape)
        # where the random array is less than 0.15, we set true
        mask_arr = (rand < 0.15) * (result["input_ids"] != 101) * (result["input_ids"] != 102)
        # create selection from mask_arr
        selection = torch.flatten((mask_arr[0]).nonzero()).tolist()
        # apply selection index to inputs.input_ids, adding MASK tokens
        result["input_ids"][:, selection] = 103

        result["input_ids"] = result["input_ids"].numpy()

        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=batch_size,
        num_proc=1,
    )

    lm_datasets["train"].set_format(type="torch", columns=["input_ids", "labels"])
    # lm_datasets["validation"].set_format(
    #     type="torch", columns=["input_ids", "labels"]
    # )
    train_loader = torch.utils.data.DataLoader(lm_datasets["train"], batch_size=batch_size, drop_last=True)
    # val_loader = torch.utils.data.DataLoader(lm_datasets["validation"], batch_size=batch_size)

    # return train_loader, val_loader
    return train_loader

def transform_batch(batch):
    if isinstance(batch, dict):
        inputs, labels = batch["input_ids"], batch["labels"]
    else:
        inputs, labels = batch
    return inputs, labels.reshape(-1)