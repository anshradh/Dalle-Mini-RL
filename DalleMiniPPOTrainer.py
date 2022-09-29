# %%
import collections.abc
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
from einops import repeat
from fancy_einsum import einsum

# %%
def flatten_dict(nested, sep="/"):
    """Flatten dictionary and concatenate nested keys with separator."""

    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, collections.abc.Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v

    flat = {}
    rec(nested, "", flat)
    return flat


def stack_dicts(stats_dicts):
    """Stack the values of a dict."""
    results = dict()
    for k in stats_dicts[0]:
        stats_list = [torch.flatten(d[k]) for d in stats_dicts]
        results[k] = torch.stack(stats_list)
    return results


def logprobs_from_logits(logits, labels):
    """
    Calculates logprobs from supplied logits and corresponding labels.
    """
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def normalize(values, shift_mean=True):
    """Normalize values."""
    mean, var = torch.mean(values), torch.var(values)
    normalized = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        normalized += mean
    return normalized


def entropy_from_logits(logits):
    """Calculate entropy from logits."""
    pd = F.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd * logits, axis=-1)  # type: ignore
    return entropy


def stats_to_np(stats_dict):
    """Cast all tensors in dict to numpy arrays."""
    new_dict = dict()
    for k, v in stats_dict.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.detach().cpu().numpy()
        else:
            new_dict[k] = v
        if np.isscalar(new_dict[k]):
            new_dict[k] = float(new_dict[k])
    return new_dict


# %%
class PPOScheduler:
    """Implements a linear learning rate decay scheduler for PPO."""

    def __init__(self, optimizer, initial_lr: float, end_lr: float, num_updates: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.num_updates = num_updates
        self.n_step_calls = 0

    def step(self):
        """Implement linear learning rate decay so that after num_updates calls to step, the learning rate is end_lr."""
        self.n_step_calls += 1
        frac = 1.0 - (self.n_step_calls - 1.0) / self.num_updates
        lr_now = frac * self.initial_lr + (1 - frac) * self.end_lr
        self.optimizer.param_groups[0]["lr"] = lr_now


# %%
class AdaptiveKLController:
    """
    Adaptive KL controller described in the first RLHF on language models paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


# %%
class PPOTrainer:
    """
    The PPOTrainer uses Proximal Policy Optimization to optimise language models.
    """

    default_params = {
        "total_steps": 500000,
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
        "ent_coef": 0.01,
        "batch_size": 256,
        "forward_batch_size": 16,
        "ppo_epochs": 4,
        "max_grad_norm": 0.5,
        "adam_eps": 1e-4,
    }

    def __init__(self, policy_model, ref_policy_model, value_model, **ppo_params):
        """
        Initialize PPOTrainer. Modified from https://github.com/lvwerra/trl/blob/master/trl/ppo.py

        Args:
            policy_model (torch.model): Actor Image Generation Model (in this case, Dalle-Mega)
            ref_policy_model (torch.model): Image Generation reference model used for KL penalty (not modified during training)
            value_model: Critic Image Generation Model (updated separately to prevent updates to one model interfering with the other)
            tokenizer (tokenizer): Tokenizer for Image Generation Models
            ppo_params (dict or None): PPO parameters for training. Can include following keys:
                'total_steps' (int): Total number of training steps
                'start_lr' (float): Initial Adam learning rate, default: 1e-5
                'end_lr' (float): Ending Adam learning rate, default: 0.
                'batch_size' (int): Number of samples per optimisation step, default: 256
                'forward_batch_size' (int): Number of samples forward passed through model at a time, default: 16
                'ppo_epochs' (int): Number of optimisation epochs per batch of samples, default: 4
                'gamma' (float)): Gamma parameter for advantage calculation, default: 1.
                'lam' (float): Lambda parameter for advantage calcualation, default: 0.95
                'cliprange_value' (float): Range for clipping values in loss calculation, default: 0.2
                'cliprange' (float): Range for clipping in PPO policy gradient loss, default: 0.2
                'vf_coef' (float): Scaling factor for value loss, default: 0.1
                'adap_kl_ctrl' (bool): Use adaptive KL control, otherwise linear, default: True
                'init_kl_coef' (float): Initial KL penalty coefficient (used for adaptive and linear control), default: 0.2
                'target' (float): Target KL value for adaptive KL control, default: 6.0
                'horizon' (float): Horizon for adaptive KL control, default: 10000
                'max_grad_norm' (float): Used for clipping of gradients

        """
        self.ppo_params = self.default_params
        self.ppo_params.update(ppo_params)
        self.policy_model = policy_model
        self.ref_policy_model = ref_policy_model
        self.value_model = value_model

        optimizer_params = []
        for name, p in self.policy_model.named_parameters():
            if "v_head" not in name:
                p.requires_grad_()
                optimizer_params.append(p)
        for p in self.value_model.parameters():
            p.requires_grad_()
            optimizer_params.append(p)

        self.optimizer = torch.optim.Adam(
            params=optimizer_params,
            lr=self.ppo_params["start_lr"],
            eps=self.ppo_params["adam_eps"],
        )

        self.scheduler = PPOScheduler(
            self.optimizer,
            self.ppo_params["start_lr"],
            self.ppo_params["end_lr"],
            self.ppo_params["total_steps"] // self.ppo_params["batch_size"],
        )

        self.kl_ctl = AdaptiveKLController(
            self.ppo_params["init_kl_coef"],
            self.ppo_params["target"],
            self.ppo_params["horizon"],
        )

    def step(self, queries, scores):
        """
        Run a PPO optimisation step.

        args:
            queries (List): List of strings containing the queries, shape [query_length]
            scores (List): tensor containing the scores, shape [batch_size]

        returns:
            train_stats (dict): a summary of the training statistics
        """

        bs = self.ppo_params["batch_size"]
        assert bs == len(
            queries
        ), f"Batch size ({bs}) does not match number of examples ({len(queries)})"

        timing = dict()
        t0 = time.time()

        t = time.time()
        logprobs, ref_logprobs, values = self.batched_forward_pass(queries)
        timing["time/ppo/forward_pass"] = time.time() - t

        t = time.time()
        rewards, non_score_reward = self.compute_rewards(scores, logprobs, ref_logprobs)
        timing["time/ppo/compute_rewards"] = time.time() - t

        t = time.time()
        all_stats = []
        idxs = list(range(bs))
        for _ in range(self.ppo_params["ppo_epochs"]):
            np.random.shuffle(idxs)
            for i in range(bs):
                idx = idxs[i]
                train_stats = self.train_minibatch(
                    logprobs[idx].unsqueeze(0),
                    values[idx].unsqueeze(0),
                    rewards[idx].unsqueeze(0),
                    queries[idx],
                )
                all_stats.append(train_stats)
        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(
            train_stats["policy/advantages"]
        ).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(
            train_stats["policy/advantages"], -1
        )
        train_stats["policy/ratio"] = torch.flatten(
            train_stats["policy/ratio"]
        ).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
        )
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t

        self.kl_ctl.update(stats["objective/kl"], self.ppo_params["batch_size"])

        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)
        return stats

    @torch.no_grad()
    def batched_forward_pass(self, queries):
        """Calculate model outputs in multiple batches (Replay rollout)."""
        bs = self.ppo_params["batch_size"]
        fbs = self.ppo_params["forward_batch_size"]
        all_logprobs = []
        all_ref_logprobs = []
        all_values = []

        for i in range(int(bs / fbs)):
            query_batch = queries[i * fbs : (i + 1) * fbs]
            input_ids = self.policy_model.tokenize_for_forward(query_batch)
            logits, _, policy_output_ids = self.policy_model(input_ids, use_grad=False)
            ref_logits, _, policy_ref_output_ids = self.ref_policy_model(
                input_ids, use_grad=False
            )
            _, values, _ = self.value_model(input_ids, use_grad=False)
            logprobs = logprobs_from_logits(logits, policy_output_ids)
            ref_logprobs = logprobs_from_logits(ref_logits, policy_ref_output_ids)
            for j in range(fbs):
                all_values.append(values[j])
                all_logprobs.append(logprobs[j])
                all_ref_logprobs.append(ref_logprobs[j])
        return all_logprobs, all_ref_logprobs, all_values

    # @torch.set_grad_enabled(True)
    def train_minibatch(self, logprobs, values, rewards, query):
        """
        Train one PPO minibatch.
        """
        torch.cuda.empty_cache()
        loss_p, loss_v, loss_e, train_stats = self.loss(
            logprobs, values, rewards, query
        )
        loss = loss_p + loss_v + loss_e
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.optimizer.param_groups[0]["params"], self.ppo_params["max_grad_norm"]
        )
        self.optimizer.step()
        return train_stats

    def compute_rewards(self, scores, logprobs, ref_logprobs):
        """
        Compute per token rewards from scores and KL-penalty for deviation from reference policy.
        """
        rewards, non_score_rewards = [], []
        for score, logprob, ref_logprob in zip(scores, logprobs, ref_logprobs):
            kl = logprob - ref_logprob
            non_score_reward = -self.kl_ctl.value * kl
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()
            reward[-1] += score
            rewards.append(reward)
        return rewards, non_score_rewards

    @torch.no_grad()
    def calc_advantages(self, values, rewards):
        """
        Use GAE to calculate advantages from values and rewards. Implements a vectorized form of GAE.
        """
        full_values = torch.cat(
            (
                values,
                torch.zeros(
                    (values.shape[0], 1), device=values.device, dtype=values.dtype
                ),
            ),
            dim=1,
        )
        delta = (rewards + self.ppo_params["gamma"] * full_values[:, 1:]) - full_values[
            :, :-1
        ]
        gl = self.ppo_params["gamma"] * self.ppo_params["lambda"]
        gl_by_t = torch.exp(
            torch.arange(delta.shape[1], device=values.device)
        ) * np.log(gl)
        gl_by_t_mat = repeat(
            torch.outer(gl_by_t ** -1, gl_by_t).triu(),
            "t1 t2 -> b t1 t2",
            b=values.shape[0],
        )
        advantages = einsum("b t1 t2, b t2 -> b t1", gl_by_t_mat, delta)

        return advantages

    def calc_policy_loss(self, normalized_advantages, new_logprobs, old_logprobs):
        """
        Calculate the Clipped-PPO Objective Policy Loss.
        """
        ratio = torch.exp(new_logprobs - old_logprobs)

        pg_losses = -normalized_advantages * ratio
        pg_losses2 = -normalized_advantages * torch.clamp(
            ratio,
            1.0 - self.ppo_params["cliprange"],
            1.0 + self.ppo_params["cliprange"],
        )

        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
        pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())

        return pg_loss, pg_clipfrac, ratio

    def calc_value_loss(self, vpred, values, returns):
        """
        Calculate the Value-component of the PPO Loss for the critic.
        """
        vpredclipped = torch.clamp(
            vpred,
            values - self.ppo_params["cliprange_value"],
            values + self.ppo_params["cliprange_value"],
        )

        vf_losses1 = (vpred - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * torch.mean(torch.max(vf_losses1, vf_losses2))
        vf_clipfrac = torch.mean(torch.gt(vf_losses2, vf_losses1).double())
        vf_loss = self.ppo_params["vf_coef"] * vf_loss

        return vf_loss, vf_clipfrac

    def calc_entropy_loss(self, logits):
        """
        Return the entropy loss term for PPO.
        """
        entropy = torch.mean(entropy_from_logits(logits))
        e_loss = -entropy * self.ppo_params["ent_coef"]

        return e_loss, entropy

    def loss(self, old_logprobs, values, rewards, query):
        """
        Collect and return advantage, loss, and various training stat calculations.
        """
        advantages = self.calc_advantages(values, rewards)

        returns = advantages + values
        advantages = normalize(advantages)
        advantages = advantages.detach()

        query_input_ids = self.policy_model.tokenize_for_forward(query)

        logits, _, policy_output_ids = self.policy_model(query_input_ids)

        (
            _,
            vpred,
            _,
        ) = self.value_model(query_input_ids)

        logprob = logprobs_from_logits(logits, policy_output_ids)

        pg_loss, pg_clipfrac, ratio = self.calc_policy_loss(
            advantages, logprob, old_logprobs
        )

        vf_loss, vf_clipfrac = self.calc_value_loss(vpred, values, returns)

        e_loss, entropy = self.calc_entropy_loss(logits)

        loss = pg_loss + vf_loss + e_loss

        approxkl = 0.5 * torch.mean((logprob - old_logprobs) ** 2)
        policykl = torch.mean(logprob - old_logprobs)
        return_mean, return_var = torch.mean(returns), torch.var(returns)
        value_mean, value_var = torch.mean(values), torch.var(values)

        stats = dict(
            loss=dict(policy=pg_loss, value=vf_loss, entropy=e_loss, total=loss),
            policy=dict(
                entropy=entropy,
                approxkl=approxkl,
                policykl=policykl,
                clipfrac=pg_clipfrac,
                advantages=advantages,
                advantages_mean=torch.mean(advantages),
                ratio=ratio,
            ),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(
                vpred=torch.mean(vpred),
                error=torch.mean((vpred - returns) ** 2),
                clipfrac=vf_clipfrac,
                mean=value_mean,
                var=value_var,
            ),
        )
        return pg_loss, vf_loss, e_loss, flatten_dict(stats)

    def record_step_stats(self, kl_coef, **data):
        """Record training step statistics."""
        kl_list = [
            logprobs - ref_logprobs
            for logprobs, ref_logprobs in zip(data["logprobs"], data["ref_logprobs"])
        ]
        mean_kl = torch.mean(torch.stack([torch.sum(kl) for kl in kl_list]))
        mean_entropy = torch.mean(
            torch.stack([torch.sum(-log_probs) for log_probs in data["logprobs"]])
        )
        mean_non_score_reward = torch.mean(
            torch.stack(
                [
                    torch.sum(non_score_reward)
                    for non_score_reward in data["non_score_reward"]
                ]
            )
        )
        stats = {
            "objective/kl": mean_kl,
            "objective/kl_dist": kl_list,
            "objective/logprobs": data["logprobs"],
            "objective/ref_logprobs": data["ref_logprobs"],
            "objective/kl_coef": kl_coef,
            "objective/entropy": mean_entropy,
            "ppo/mean_non_score_reward": mean_non_score_reward,
        }

        for k, v in data["train_stats"].items():
            stats[f"ppo/{k}"] = torch.mean(v, axis=0)  # type: ignore
        stats["ppo/val/var_explained"] = (
            1 - stats["ppo/val/error"] / stats["ppo/returns/var"]
        )
        return stats
