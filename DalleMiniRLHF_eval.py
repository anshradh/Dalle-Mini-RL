# %%
import torch
from .DalleMiniRLHF_dataset_loading import load_ppo_dataset
from datasets import Dataset
from .DalleMiniRLHF_training import rewards_from_image_list
from IPython.display import display
import pandas as pd
from w3d5_part1_clip_solution import (
    CLIPConfig,
    load_trained_model,
    CLIPTextConfig,
    CLIPVisionConfig,
)
from .DalleMiniPytorch import DalleMiniWithValueHead
from .load_nsfw_model import load_pytorch_and_keras_nsfw_models

MAIN = __name__ == "__main__"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
if MAIN:
    # Load models and dataset
    ppo_dataset = load_ppo_dataset()
    policy_model_ref = DalleMiniWithValueHead(is_mega=False, dtype=torch.float16)
    policy_model = DalleMiniWithValueHead(is_mega=False, dtype=torch.float16)
    clip_config = CLIPConfig(CLIPVisionConfig(), CLIPTextConfig())
    clip = load_trained_model(clip_config).to(device)
    nsfw, nsfw_keras_model = load_pytorch_and_keras_nsfw_models(device)
# %%
if MAIN:
    # Load trained model weights
    policy_model.load_state_dict(
        torch.load("new_fully_trained_policy_model_state_dict.pt")
    )
# %%
if MAIN:
    #### get a batch from the dataset
    bs = 50
    game_data = dict()
    assert isinstance(ppo_dataset, Dataset)
    df_batch = ppo_dataset.shuffle(seed=22).select(range(bs))
    game_data["query"] = list(df_batch["prompt"])

    #### get response from trained and reference policies
    game_data["response (before)"] = [
        policy_model_ref.generate_image(df_batch["prompt"][i], seed=1, grid_size=1)
        for i in range(bs)
    ]

    game_data["response (after)"] = [
        policy_model.generate_image(df_batch["prompt"][i], seed=1, grid_size=1)
        for i in range(bs)
    ]

    #### nsfw analysis of query/response pairs before/after
    rewards = rewards_from_image_list(
        df_batch["prompt"],
        game_data["response (before)"],
        device,
        clip,
        nsfw,
        nsfw_keras_model,
    )
    game_data["rewards (before)"] = rewards.cpu().numpy()

    rewards = rewards_from_image_list(
        df_batch["prompt"],
        game_data["response (after)"],
        device,
        clip,
        nsfw,
        nsfw_keras_model,
    )
    game_data["rewards (after)"] = rewards.cpu().numpy()

    # store results in a dataframe
    df_results = pd.DataFrame(game_data)
# %%
if MAIN:
    print()
    print("describe:")
    display(df_results.describe())
    print()
    print("conditioning on reward below 0 (likely NSFW)")
    display(
        df_results[df_results["rewards (before)"] <= 0]["rewards (before)"].describe()
    )
    display(
        df_results[df_results["rewards (after)"] <= 0]["rewards (after)"].describe()
    )
# %%
if MAIN:
    for i in range(len(df_results)):
        print("Prompt #" + str(i + 1) + "\n" + df_results["query"][i])
        print("Reference Response #" + str(i + 1) + "\n")
        display(df_results["response (before)"][i])
        print("Trained Response #" + str(i + 1) + "\n")
        display(df_results["response (after)"][i])
# %%
