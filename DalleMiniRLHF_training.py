# %%
from typing import cast, Callable
import time
import torch
from tqdm.auto import tqdm
import wandb
import numpy as np
from PIL import Image
from torchvision import transforms
from w3d5_part1_clip_solution import (
    CLIPConfig,
    load_trained_model,
    CLIPTextConfig,
    CLIPVisionConfig,
)
from w3d5_part1_clip_solution import tokenize as clip_text_tokenize
from .DalleMiniPPOTrainer import PPOTrainer
import torch
import torch.nn as nn
from .load_nsfw_model import (
    nsfw_normalize,
    load_pytorch_and_keras_nsfw_models,
)
from .DalleMiniPytorch import DalleMiniWithValueHead
import wandb
from .DalleMiniRLHF_dataset_loading import load_ppo_dataset

MAIN = __name__ == "__main__"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
# Load all the models
clip_config = CLIPConfig(CLIPVisionConfig(), CLIPTextConfig())
if MAIN:
    policy_model = DalleMiniWithValueHead(is_mega=False, dtype=torch.float16)
    policy_model_ref = DalleMiniWithValueHead(is_mega=False, dtype=torch.float16)
    value_model = DalleMiniWithValueHead(is_mega=False, dtype=torch.float16)
    nsfw, nsfw_keras_model = load_pytorch_and_keras_nsfw_models(device)
    clip = load_trained_model(clip_config).to(device)
# %%
# Load the dataset
ppo_dataset = load_ppo_dataset()
# %%
@torch.no_grad()
def rewards_from_image_list(
    prompts, image_list, device, clip, nsfw_pytorch, nsfw_keras
):
    """
    Helper function to create a list of rewards for images, utilizing the trained NSFW image classifier and feeding CLIP embeddings into it.
    """
    rewards_list = []
    for i, im in enumerate(image_list):
        full_im = im
        preprocess = cast(
            Callable[[Image.Image], torch.Tensor],
            transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(
                        (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.Normalize(
                        (0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            ),
        )
        clip_tokenized_prompt = clip_text_tokenize([prompts[i][:200]])
        clip_input_ids = clip_tokenized_prompt["input_ids"]
        if clip_input_ids.shape[1] > clip_config.text_config.max_position_embeddings:
            clip_input_ids = clip_input_ids[
                :, clip_config.text_config.max_position_embeddings
            ]
        clip_attention_mask = clip_tokenized_prompt["attention_mask"]
        if (
            clip_attention_mask.shape[1]
            > clip_config.text_config.max_position_embeddings
        ):
            clip_attention_mask = clip_attention_mask[
                :, clip_config.text_config.max_position_embeddings
            ]
        clip_out = clip(
            clip_input_ids.to(device),
            clip_attention_mask.to(device),
            preprocess(full_im).unsqueeze(0).to(device),
        )
        nsfw_im_emb_arr = nsfw_normalize(
            clip_out.image_embeds.cpu().detach().numpy(), nsfw_keras
        )
        nsfw_score = nsfw_pytorch(torch.from_numpy(nsfw_im_emb_arr).to(device))
        nsfw_reward = (
            (-30)
            if nsfw_score > 0.4
            else (0)
            if nsfw_score < 0.1
            else 10
            if nsfw_score < 0.01
            else (-10)
        )
        clip_sim = clip_out.image_embeds @ clip_out.text_embeds.T
        clip_reward = 20.0 * clip_sim.squeeze(-1)
        reward = (
            clip_reward + nsfw_reward
            if nsfw_reward >= 0
            else nsfw_reward + torch.zeros_like(clip_reward)
        )
        rewards_list.append(reward)
    return torch.cat(rewards_list, dim=0)  # type: ignore


# %%
config = {
    "total_steps": 1600,
    "batch_size": 4,
    "forward_batch_size": 1,
    "ppo_epochs": 2,
    "start_lr": 1e-5,
    "end_lr": 0.0,
    "init_kl_coef": 0.2,
    "target": 6,
    "horizon": 10000,
    "gamma": 1,
    "lam": 0.95,
    "cliprange": 0.2,
    "cliprange_value": 0.2,
    "vf_coef": 0.1,
    "max_grad_norm": 0.25,
    "adam_eps": 1e-4,
}
if MAIN:
    ppo_trainer = PPOTrainer(policy_model, policy_model_ref, value_model, **config)
# %%
if MAIN:
    """
    Main Training loop, iterating over batches sampled from the dataset and passing them to the PPOTrainer.
    """
    wandb.init(project="MLAB_final_project", config=config)
    print(f"Training with config: {config}")
    for epoch in tqdm(
        range(int(np.ceil(config["total_steps"] / config["batch_size"])))
    ):
        torch.cuda.empty_cache()
        logs = dict()
        game_data = dict()
        timing = dict()
        t0 = time.time()

        #### get a batch from the dataset
        df_batch = ppo_dataset.shuffle().select(range(config["batch_size"]))  # type: ignore
        game_data["query"] = list(df_batch["prompt"])

        #### get response from policy model/roll-out phase
        t = time.time()
        queries = []
        responses = []
        with torch.no_grad():
            for i in range(config["batch_size"]):
                query = df_batch["prompt"][i]
                queries.append(query)
                response = policy_model.generate_image(
                    df_batch["prompt"][i], seed=1, grid_size=1
                )
                responses.append(response)
        game_data["response"] = responses
        timing["time/get_response"] = time.time() - t

        torch.cuda.empty_cache()

        #### get rewards
        t = time.time()
        rewards = rewards_from_image_list(
            list(df_batch["prompt"]), responses, device, clip, nsfw, nsfw_keras_model
        )
        timing["time/get_rewards"] = time.time() - t

        torch.cuda.empty_cache()

        #### Run PPO training
        t = time.time()
        stats = ppo_trainer.step(queries, rewards)
        timing["time/optimization"] = time.time() - t

        #### Log everything
        timing["time/epoch"] = time.time() - t0
        table_rows = [
            list(r)
            for r in zip(
                game_data["query"], game_data["response"], rewards.cpu().tolist()
            )
        ]
        wandb.log(stats, step=(epoch) * config["batch_size"])
        torch.save(
            policy_model.state_dict(), f"new_policy_model_state_dict_after_{epoch}.pt"
        )
        ppo_trainer.scheduler.step()
    wandb.finish()
    torch.save(
        policy_model.state_dict(), "new_fully_trained_policy_model_state_dict.pt"
    )
# %%
