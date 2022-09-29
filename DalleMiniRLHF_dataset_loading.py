# %%
from datasets import load_dataset, Dataset, concatenate_datasets

# %%
PPO_TRUNCATED_DATASET_SIZE = 80000
PPO_DATASET_SIZE = 1600
# %%
def load_ppo_dataset():
    print("LOADING DATASET")
    full_dataset = load_dataset("laion/laion400m", split="train")
    assert isinstance(full_dataset, Dataset)
    truncated_dataset = full_dataset.shuffle(seed=0).select(
        range(PPO_TRUNCATED_DATASET_SIZE)
    )
    ppo_dataset_nsfw = (
        truncated_dataset.filter(lambda example: example["NSFW"] == "NSFW")
        .shuffle(seed=0)
        .select(range(PPO_DATASET_SIZE // 2))
    )
    ppo_dataset_not_nsfw = (
        truncated_dataset.filter(lambda example: example["NSFW"] != "NSFW")
        .shuffle(seed=0)
        .select(range(PPO_DATASET_SIZE // 2))
    )
    ppo_dataset = concatenate_datasets([ppo_dataset_nsfw, ppo_dataset_not_nsfw])

    def preprocess_caption_text_function(examples):
        examples["prompt"] = [prompt for prompt in examples["TEXT"]]
        return examples

    ppo_dataset = ppo_dataset.map(
        preprocess_caption_text_function,
        batched=True,
        remove_columns=[
            "SAMPLE_ID",
            "URL",
            "TEXT",
            "HEIGHT",
            "WIDTH",
            "LICENSE",
            "NSFW",
            "similarity",
        ],
    )
    print("FINISHED PREPROCESSING DATASET")
    return ppo_dataset
