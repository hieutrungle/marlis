import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
from typing import Tuple, Callable, Dict, Optional, Union
import math
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import wandb
import torchinfo
import importlib.resources
import copy
import pyrallis

from torchrl.data import ReplayBuffer, LazyMemmapStorage
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule, TensorDictModule

import traceback
import multiprocessing as mp
import marlis
from marlis.utils import utils, pytorch_utils, running_mean
from marlis.drl.agents import sac
import matplotlib.pyplot as plt
from marlis.drl.envs import register_envs

register_envs()
torch.set_float32_matmul_precision("high")


@dataclass
class TrainConfig:

    # General arguments
    command: str = "train"  # the command to run
    load_model: str = "-1"  # Model load file name for resume training, "-1" doesn't load
    load_eval_model: str = "-1"  # Model load file name for evaluation, "-1" doesn't load
    checkpoint_dir: str = "-1"  # the path to save the model
    replay_buffer_dir: str = "-1"  # the path to save the replay buffer
    load_replay_buffer: str = "-1"  # the path to load the replay buffer
    verbose: bool = False  # whether to log to console
    seed: int = 1  # seed of the experiment
    eval_seed: int = 111  # seed of the evaluation
    save_interval: int = 100  # the interval to save the model
    start_step: int = 0  # the starting step of the experiment

    # Environment specific arguments
    env_id: str = "wireless-sigmap-v0"  # the environment id of the task
    sionna_config_file: str = "-1"  # Sionna config file
    num_envs: int = 8  # the number of parallel environments
    ep_len: int = 75  # the maximum length of an episode
    eval_ep_len: int = 45  # the maximum length of an episode

    # Network specific arguments
    ff_dim: int = 256  # the hidden dimension of the feedforward networks

    # Algorithm specific arguments
    total_timesteps: int = 10_001  # total timesteps of the experiments
    n_updates: int = 20  # the number of updates per step
    buffer_size: int = int(80_000)  # the replay memory buffer size
    gamma: float = 0.985  # the discount factor gamma
    tau: float = 0.015  # target smoothing coefficient (default: 0.005)
    batch_size: int = 256  # the batch size of sample from the reply memory
    learning_starts: int = 601  # the timestep to start learning
    policy_lr: float = 3e-4  # the learning rate of the policy network optimizer
    q_lr: float = 1e-3  # the learning rate of the q network optimizer
    warmup_steps: int = 500  # the number of warmup steps
    policy_frequency: int = 2  # the frequency of training policy (delayed)
    target_network_frequency: int = 2  # the frequency of updates for the target nerworks
    alpha: float = 0.2  # Entropy regularization coefficient

    # Wandb logging
    wandb_mode: str = "online"  # wandb mode
    project: str = "SARIS"  # wandb project name
    group: str = "TQC"  # wandb group name
    name: str = "Reward_split"  # wandb run name

    def __post_init__(self):
        lib_dir = importlib.resources.files(marlis)
        source_dir = os.path.dirname(lib_dir)
        self.source_dir = source_dir

        if self.checkpoint_dir == "-1":
            raise ValueError("Checkpoints dir is required for training")
        if self.sionna_config_file == "-1":
            raise ValueError("Sionna config file is required for training")
        if self.command.lower() == "train" and self.replay_buffer_dir == "-1":
            raise ValueError("Replay buffer dir is required for training")
        if self.command.lower() == "eval" and self.load_eval_model == "-1":
            raise ValueError("Load eval model is required for evaluation")

        device = pytorch_utils.init_gpu()
        self.device = device


def wandb_init(config: TrainConfig) -> None:
    key_filename = os.path.join(config.source_dir, "tmp_wandb_api_key.txt")
    with open(key_filename, "r") as f:
        key_api = f.read().strip()
    wandb.login(relogin=True, key=key_api, host="https://api.wandb.ai")
    wandb.init(
        config=config,
        dir=config.checkpoint_dir,
        project=config.project,
        group=config.group,
        name=config.name,
        mode=config.wandb_mode,
    )


def make_env(config: TrainConfig, idx: int, eval_mode: bool) -> Callable:

    def thunk() -> gym.Env:

        seed = config.seed if not eval_mode else config.eval_seed
        max_episode_steps = config.ep_len if not eval_mode else config.eval_ep_len
        seed += idx
        env = gym.make(
            config.env_id,
            idx=idx,
            sionna_config_file=config.sionna_config_file,
            log_string=config.name,
            eval_mode=eval_mode,
            seed=seed,
            max_episode_steps=max_episode_steps,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env = gym.wrappers.FlattenObservation(env)
        env.action_space.seed(config.seed)
        env.observation_space.seed(config.seed)

        return env

    return thunk


def normalize_obs(
    flat_obs: torch.Tensor,
    rms: running_mean.RunningMeanStd,
    envs: gym.vector.VectorEnv,
    epsilon: float = 1e-9,
):

    channel_space = envs.get_attr("channel_space")
    angle_space = envs.get_attr("angle_space")
    # position_space = envs.get_attr("position_space")

    # channels
    channel_len = math.prod(channel_space[0].shape)
    channels = flat_obs[..., :channel_len]
    channels = torch.div(
        torch.sub(channels, rms.mean.to(device=channels.device, dtype=channels.dtype)),
        torch.sqrt(rms.var.to(device=channels.device, dtype=channels.dtype)) + epsilon,
    )

    # angles
    angle_len = math.prod(angle_space[0].shape)
    angles = flat_obs[..., channel_len : channel_len + angle_len]
    init_angles = [math.radians(135.0)] + [math.radians(90.0)] * 7
    init_angles = np.concatenate([init_angles] * 9)
    # offset
    angles = torch.sub(angles, torch.tensor(init_angles, device=angles.device, dtype=angles.dtype))
    # # normalize
    # angles = torch.div(torch.rad2deg(angles), 45.0)

    pos = flat_obs[..., channel_len + angle_len :]
    flat_obs = torch.cat([channels, angles, pos], dim=-1)
    return flat_obs.float()


def update_channel_rmss(
    flat_obs: torch.Tensor,
    channel_rms: running_mean.RunningMeanStd,
):
    channel_len = np.prod(channel_rms.mean.shape)
    channel_rms.update(flat_obs[..., :channel_len])


def create_scheduler(optimizer, warmup_steps, num_train_steps, lr):
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0 / 10.0, total_iters=warmup_steps
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, num_train_steps - warmup_steps, eta_min=lr / 5.0
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, cosine_scheduler], [warmup_steps]
    )
    return scheduler


@pyrallis.wrap()
def main(config: TrainConfig):

    torch.compiler.reset()
    sionna_config = utils.load_config(config.sionna_config_file)
    # set random seeds
    pytorch_utils.init_seed(config.seed)
    if config.verbose:
        utils.log_args(config)
        utils.log_config(sionna_config)

    # env setup
    if config.command.lower() == "train":
        envs = gym.vector.AsyncVectorEnv(
            [make_env(config, i, eval_mode=False) for i in range(config.num_envs)],
            context="spawn",
        )
        # envs = gym.vector.SyncVectorEnv(
        #     [make_env(config, i, eval_mode=False) for i in range(config.num_envs)],
        #     # context="spawn",
        # )
    elif config.command.lower() == "eval":
        envs = gym.vector.AsyncVectorEnv(
            [make_env(config, i, eval_mode=True) for i in range(config.num_envs)],
            context="spawn",
        )
    else:
        raise ValueError(f"Invalid command: {config.command}, available commands: train, eval")

    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"
    print(f"Observation space: {envs.single_observation_space}")
    print(f"Action space: {envs.single_action_space}\n")
    ob_space = envs.single_observation_space
    ac_space = envs.single_action_space

    # Create running meanstd for normalization
    channel_space = envs.get_attr("channel_space")
    channel_rms = running_mean.RunningMeanStd(
        shape=(math.prod(channel_space[0].shape)),
    )

    # ##################
    # # TESTING
    # obs, _ = envs.reset(options={"start_init": True})
    # # actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    # # next_obs, rewards, terminations, truncations, infos = envs.step(actions)

    # # TRY NOT TO MODIFY: record rewards for plotting purposes
    # # if "final_info" in infos:

    # #     # get path gains
    # #     prev_path_gains = [info["prev_path_gains"] for info in infos["final_info"]]
    # #     path_gains = [info["path_gains"] for info in infos["final_info"]]
    # # else:
    # #     prev_path_gains = infos["prev_path_gains"]
    # #     path_gains = infos["path_gains"]
    # # prev_path_gains = np.stack(prev_path_gains)
    # # path_gains = np.stack(path_gains)
    # # prev_path_gains = torch.as_tensor(prev_path_gains, dtype=torch.float)
    # # path_gains = torch.as_tensor(path_gains, dtype=torch.float)

    # # # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
    # # real_next_obs = list(copy.deepcopy(next_obs))
    # # for idx, trunc in enumerate(truncations):
    # #     if trunc:
    # #         real_next_obs[idx] = infos["final_observation"][idx]

    # exit(0)
    # ##################

    # Init checkpoints
    print(f"Checkpoints dir: {config.checkpoint_dir}")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    with open(os.path.join(config.checkpoint_dir, "train_config.yaml"), "w") as f:
        pyrallis.dump(config, f)

    # Load models
    checkpoint = None
    if config.command.lower() == "eval":
        print(f"Loading model from {config.load_eval_model}")
        checkpoint = torch.load(config.load_eval_model, weights_only=False)
    else:
        if config.load_model != "-1":
            print(f"Loading model from {config.load_model}")
            checkpoint = torch.load(config.load_model, weights_only=False)

    # Actor-Critic setup
    actor = sac.Actor(envs=envs, ff_dim=config.ff_dim, device=config.device)
    actor_detach = sac.Actor(envs=envs, ff_dim=config.ff_dim, device=config.device)
    if checkpoint != None:
        actor.load_state_dict(checkpoint["actor"])
    from_module(actor).to_module(actor_detach)
    policy = TensorDictModule(
        actor_detach.get_action, in_keys=["observation"], out_keys=["action", "log_pi", "mean"]
    )

    qf1 = sac.SoftQNetwork(envs=envs, ff_dim=config.ff_dim, device=config.device)
    qf2 = sac.SoftQNetwork(envs=envs, ff_dim=config.ff_dim, device=config.device)

    tmp_obs = torch.randn((1, *ob_space.shape), device=config.device)
    torchinfo.summary(
        actor,
        input_data=tmp_obs,
        col_names=["input_size", "output_size", "num_params"],
    )
    tmp_ac = torch.randn(1, *ac_space.shape, device=config.device)
    torchinfo.summary(
        qf1,
        input_data=[tmp_obs, tmp_ac],
        col_names=["input_size", "output_size", "num_params"],
    )

    # Target networks
    qf1_target = sac.SoftQNetwork(envs=envs, ff_dim=config.ff_dim, device=config.device)
    qf2_target = sac.SoftQNetwork(envs=envs, ff_dim=config.ff_dim, device=config.device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    # Automatic entropy tuning
    target_entropy = -torch.prod(
        torch.Tensor(envs.single_action_space.shape).to(config.device)
    ).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=config.device)

    # Optimzier setup
    a_optimizer = optim.AdamW([log_alpha], lr=torch.tensor(config.q_lr))
    q_optimizer = optim.AdamW(
        list(qf1.parameters()) + list(qf2.parameters()), lr=torch.tensor(config.q_lr)
    )
    actor_optimizer = optim.AdamW(list(actor.parameters()), lr=torch.tensor(config.policy_lr))

    if checkpoint != None:
        print(f"Loading models and optimizers from checkpoint!")
        actor.load_state_dict(checkpoint["actor"])
        actor_optimizer = optim.AdamW(list(actor.parameters()), lr=torch.tensor(config.policy_lr))
        actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])

        if config.load_eval_model == "-1":
            log_alpha = checkpoint["log_alpha"].clone().detach().requires_grad_(True)
            a_optimizer = optim.AdamW([log_alpha], lr=torch.tensor(config.q_lr))
            a_optimizer.load_state_dict(checkpoint["a_optimizer"])

            qf1.load_state_dict(checkpoint["qf1"])
            qf2.load_state_dict(checkpoint["qf2"])
            qf1_target.load_state_dict(checkpoint["qf1_target"])
            qf2_target.load_state_dict(checkpoint["qf2_target"])
            q_optimizer = optim.AdamW(
                list(qf1.parameters()) + list(qf2.parameters()), lr=torch.tensor(config.q_lr)
            )
            q_optimizer.load_state_dict(checkpoint["q_optimizer"])

        channel_rms = checkpoint["channel_rms"]

    # replay buffer setup
    rb_dir = config.replay_buffer_dir
    rb = ReplayBuffer(
        storage=LazyMemmapStorage(config.buffer_size, scratch_dir=rb_dir),
        batch_size=config.batch_size,
    )
    if config.load_replay_buffer != "-1":
        print(f"Loading replay buffer from {config.load_replay_buffer}")
        rb.loads(config.load_replay_buffer)
        print(f"Replay buffer loaded with {len(rb)} samples")

    envs.single_observation_space.dtype = np.float32

    if config.command.lower() == "train":
        try:
            train_agent(
                config,
                envs,
                channel_rms,
                actor,
                actor_detach,
                policy,
                qf1,
                qf2,
                qf1_target,
                qf2_target,
                target_entropy,
                log_alpha,
                a_optimizer,
                q_optimizer,
                # q_scheduler,
                actor_optimizer,
                # actor_scheduler,
                rb,
            )
        except Exception as e:
            traceback.print_exc()
            raise e
        finally:
            rb.dump(config.replay_buffer_dir)
            wandb.finish()
            envs.close()
            envs.close_extras()
    elif config.command.lower() == "eval":
        eval(
            config,
            envs,
            channel_rms,
            actor,
        )
        envs.close()
        envs.close_extras()
    else:
        raise ValueError(f"Invalid command: {config.command}, available commands: train, eval")


def train_agent(
    config: TrainConfig,
    envs: gym.vector.AsyncVectorEnv,
    channel_rms: running_mean.RunningMeanStd,
    actor: sac.Actor,
    actor_detach: sac.Actor,
    policy: TensorDictModule,
    qf1: sac.SoftQNetwork,
    qf2: sac.SoftQNetwork,
    qf1_target: sac.SoftQNetwork,
    qf2_target: sac.SoftQNetwork,
    target_entropy: float,
    log_alpha: torch.Tensor,
    a_optimizer: torch.optim.Optimizer,
    q_optimizer: torch.optim.Optimizer,
    # q_scheduler: torch.optim.lr_scheduler._LRScheduler,
    actor_optimizer: torch.optim.Optimizer,
    # actor_scheduler: torch.optim.lr_scheduler._LRScheduler,
    rb: ReplayBuffer,
):
    wandb_init(config)

    alpha = log_alpha.detach().exp()

    def update_critic(data):
        # optimize the model
        q_optimizer.zero_grad()
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = actor.get_action(data["next_observations"])
            qf1_next_target = qf1_target(data["next_observations"], next_state_actions)
            qf2_next_target = qf2_target(data["next_observations"], next_state_actions)
            min_qf_next_target = torch.minimum(qf1_next_target, qf2_next_target)
            min_qf_next_target = min_qf_next_target - alpha * next_state_log_pi
            next_q_value = data["rewards"].flatten() + (
                1.0 - data["terminations"].float().flatten()
            ) * config.gamma * min_qf_next_target.view(-1)

        qf1_values = qf1(data["observations"], data["actions"])
        qf1_loss = F.mse_loss(qf1_values.view(-1), next_q_value)

        qf2_values = qf2(data["observations"], data["actions"])
        qf2_loss = F.mse_loss(qf2_values.view(-1), next_q_value)

        qf_loss = qf1_loss + qf2_loss

        qf_loss.backward()
        q_optimizer.step()
        return TensorDict(qf_loss=qf_loss.detach())

    def update_pol(data):
        actor_optimizer.zero_grad()
        pi, log_pi, _ = actor.get_action(data["observations"])
        qf1_pi = qf1(data["observations"], pi)
        qf2_pi = qf2(data["observations"], pi)
        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)
        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

        actor_loss.backward()
        actor_optimizer.step()

        a_optimizer.zero_grad()
        with torch.no_grad():
            _, log_pi, _ = actor.get_action(data["observations"])
        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

        alpha_loss.backward()
        a_optimizer.step()

        return TensorDict(
            actor_loss=actor_loss.detach(),
            qf_pi=min_qf_pi.detach(),
            log_pi=log_pi.detach(),
            alpha=alpha.detach(),
            alpha_loss=alpha_loss.detach(),
        )

    update_critic = torch.compile(update_critic)
    update_pol = torch.compile(update_pol)
    policy = torch.compile(policy)

    # update_critic = CudaGraphModule(update_critic, in_keys=[], out_keys=[], warmup=3)
    # update_pol = CudaGraphModule(update_pol, in_keys=[], out_keys=[], warmup=3)

    # TRY NOT TO MODIFY: start the game
    stored_obs = []
    if config.start_step == 0:
        obs, _ = envs.reset(options={"start_init": True})
    else:
        obs, _ = envs.reset()
    stored_obs.append(obs)
    last_step = config.start_step + config.total_timesteps
    pbar = tqdm.tqdm(
        range(config.start_step, config.start_step + config.total_timesteps + 1),
        dynamic_ncols=True,
        initial=config.start_step,
        total=config.start_step + config.total_timesteps + 1,
    )
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=envs.num_envs)
    desc = ""

    for global_step in pbar:
        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            if config.start_step == 0 and global_step < config.learning_starts * 9 / 10:
                actions = np.array(
                    [envs.single_action_space.sample() for _ in range(envs.num_envs)]
                )
            else:
                torch_obs = torch.Tensor(copy.deepcopy(obs)).float().to(config.device)
                torch_obs = normalize_obs(torch_obs, channel_rms, envs)
                actions, _, _ = policy(torch_obs.to(config.device))
                actions = actions.detach().cpu().numpy()

        if (
            global_step == config.learning_starts
            and len(stored_obs) > 0
            and config.load_model == "-1"
        ):
            # update channel rms normalization
            stored_obs = np.concatenate(stored_obs, axis=0)
            update_channel_rmss(torch.tensor(stored_obs), channel_rms)
            print(f"Updated channel rms: {channel_rms}")
            torch.save(
                {"channel_rms": channel_rms}, os.path.join(config.checkpoint_dir, "channel_rms.pth")
            )
            stored_obs = []

        # TRY NOT TO MODIFY: execute the game and log data.
        try:
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        except Exception as e:
            traceback.print_exc()
            # Close any multiprocesses from mp queue if present
            for proc in mp.active_children():
                # proc.terminate()
                proc.join(timeout=60)
            time.sleep(2)
            obs, _ = envs.reset(seed=config.seed)
            continue

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            # log episodic returns
            for info in infos["final_info"]:
                r = float(info["episode"]["r"][0])
                max_ep_ret = max(max_ep_ret, r)
                avg_returns.append(r)

            avg_ret = torch.tensor(avg_returns).mean()
            std_ret = torch.tensor(avg_returns).std()
            log_dict = {"episodic_return": avg_ret, "episodic_return_std": std_ret}

            desc = f"global_step={global_step}, episodic_return={avg_ret: 4.2f} (max={max_ep_ret: 4.2f})"
            wandb.log(log_dict, step=global_step)

            # update channel rms normalization
            if (
                global_step < config.learning_starts
                and len(stored_obs) > 0
                and config.load_model == "-1"
            ):
                stored_obs = np.concatenate(stored_obs, axis=0)
                update_channel_rmss(torch.tensor(stored_obs), channel_rms)
                print(f"Updated channel rms: {channel_rms}")
                torch.save(
                    {"channel_rms": channel_rms},
                    os.path.join(config.checkpoint_dir, "channel_rms.pth"),
                )
            stored_obs = []

            # get path gains
            prev_path_gains = [info["prev_path_gains"] for info in infos["final_info"]]
            path_gains = [info["path_gains"] for info in infos["final_info"]]
        else:
            prev_path_gains = infos["prev_path_gains"]
            path_gains = infos["path_gains"]
        prev_path_gains = np.stack(prev_path_gains)
        path_gains = np.stack(path_gains)
        prev_path_gains = torch.as_tensor(prev_path_gains, dtype=torch.float)
        path_gains = torch.as_tensor(path_gains, dtype=torch.float)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = list(copy.deepcopy(next_obs))
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        if global_step == last_step - 1:
            truncations = [True] * len(truncations)
        rewards = np.asarray(rewards, dtype=np.float32)[..., None]
        terminations = np.asarray(terminations, dtype=np.float32)[..., None]
        truncations = np.asarray(truncations, dtype=np.float32)[..., None]
        transition = TensorDict(
            observations=obs,
            next_observations=real_next_obs,
            actions=actions,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            path_gains=path_gains,
            prev_path_gain=prev_path_gains,
            batch_size=obs.shape[0],
        )
        rb.extend(transition)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        # if "final_info" in infos and global_step < config.learning_starts * 9 / 10:
        if (
            global_step % (config.ep_len // 4) == 0
            and global_step < config.learning_starts * 9 / 10
            and global_step != 0
            and config.start_step == 0
        ):
            obs, _ = envs.reset(options={"start_init": True})
        stored_obs.append(obs)

        # ALGO LOGIC: training.
        if global_step > config.learning_starts:
            log_infos = {}
            for j in range(config.n_updates):
                # Get data
                data = rb.sample()
                data = {
                    k: torch.as_tensor(v, device=config.device, dtype=torch.float)
                    for k, v in data.items()
                }
                data["observations"] = normalize_obs(data["observations"], channel_rms, envs)
                data["next_observations"] = normalize_obs(
                    data["next_observations"], channel_rms, envs
                )
                data = TensorDict(data)

                # Update Q networks
                log_infos.update(update_critic(data))
                # q_scheduler.step()

                if j % config.policy_frequency == 1:  # TD 3 Delayed update support
                    for _ in range(config.policy_frequency):
                        # compensate for the delay by doing 'actor_update_interval' instead of 1
                        log_infos.update(update_pol(data))
                        # actor_scheduler.step()

                        with torch.no_grad():
                            log_alpha.clamp_(-5.0, 1.0)
                            alpha.copy_(log_alpha.detach().exp())
                            alpha = torch.clamp(alpha, 0.1, 0.95)

                # update the target networks
                if global_step % config.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(
                            config.tau * param.data + (1 - config.tau) * target_param.data
                        )
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(
                            config.tau * param.data + (1 - config.tau) * target_param.data
                        )

            if global_step > config.learning_starts:
                with torch.no_grad():
                    q_lr = q_optimizer.param_groups[0]["lr"]
                    a_lr = actor_optimizer.param_groups[0]["lr"]
                    logs = {
                        "train/path_gain": path_gains.mean(),
                        "train/path_gain_std": path_gains.std(),
                        "train/path_gain_diff": (path_gains - prev_path_gains).mean(),
                        "train/reward_mean": rewards.mean(),
                        "train/reward_std": rewards.std(),
                        "train/actor_loss": log_infos["actor_loss"].mean().item(),
                        "train/alpha_loss": log_infos["alpha_loss"].mean().item(),
                        "train/qf_loss": log_infos["qf_loss"].mean().item(),
                        "train/alpha": alpha.item(),
                        "train/log_alpha": log_alpha.clone().detach().item(),
                        "train/actor_entropy": (-log_infos["log_pi"]).mean().item(),
                        "train/actor_min_q": log_infos["qf_pi"].mean().item(),
                        # "train/q_lr": q_lr,
                        # "train/a_lr": a_lr,
                    }

                wandb.log({**logs}, step=global_step)
                pbar.set_description(
                    desc
                    + f" | actor_loss={logs['train/actor_loss']: 4.3f} | qf_loss={logs['train/qf_loss']: 4.3f}"
                )

            if global_step % config.save_interval == 0 or global_step == last_step - 1:
                saved_dict = {
                    "actor": actor.state_dict(),
                    "qf1": qf1.state_dict(),
                    "qf2": qf2.state_dict(),
                    "qf1_target": qf1_target.state_dict(),
                    "qf2_target": qf2_target.state_dict(),
                    "log_alpha": log_alpha,
                    "channel_rms": channel_rms,
                    "q_optimizer": q_optimizer.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "a_optimizer": a_optimizer.state_dict(),
                }
                torch.save(
                    saved_dict,
                    os.path.join(config.checkpoint_dir, f"model_{global_step}.pth"),
                )
                torch.save(
                    saved_dict,
                    os.path.join(config.checkpoint_dir, f"model.pth"),
                )

    #         if (
    #             global_step % int(1.5 * config.save_interval) == 0
    #             or global_step == config.total_timesteps - 1
    #         ):
    #             # evaluate the model
    #             eval_episodic_rets = eval(config, eval_envs, obs_rmss, actor, is_plot=False)
    #             avg_ret = torch.tensor(eval_episodic_rets).mean()
    #             std_ret = torch.tensor(eval_episodic_rets).std()
    #             log_dict = {"eval/episodic_return": avg_ret, "eval/episodic_return_std": std_ret}
    #             wandb.log(log_dict, step=global_step)

    # eval_envs.close()
    # eval_envs.close_extras()


def eval(
    config: TrainConfig,
    envs: gym.vector.AsyncVectorEnv,
    channel_rms: running_mean.RunningMeanStd,
    actor: sac.Actor,
    is_plot: bool = True,
):

    # print(obs_rmss)
    mode = "default"
    # policy = TensorDictModule(
    #     actor.get_action, in_keys=["observation"], out_keys=["action", "log_prob", "mean"]
    # )
    # policy = torch.compile(policy, mode=mode)
    # policy = CudaGraphModule(policy)

    policy = torch.compile(actor.get_action, mode=mode)

    all_rewards = np.empty((config.eval_ep_len, envs.num_envs))
    all_path_gains = np.empty((config.eval_ep_len, envs.num_envs, 3))
    episodic_returns = np.zeros((envs.num_envs,))

    obs, _ = envs.reset(seed=config.seed)

    for global_step in range(config.eval_ep_len):
        # print(f"\nSTEP: {global_step}")
        torch_obs = torch.tensor(copy.deepcopy(obs), dtype=torch.float, device=config.device)
        torch_obs = normalize_obs(torch_obs, channel_rms, envs)

        with torch.no_grad():
            # actions = actor(obs=normalized_flat_obs)
            _, _, actions = policy(torch_obs.to(config.device))
            actions = actions.detach().cpu().numpy()

            # actions, _, _ = actor.get_action(torch_obs.to(config.device))
            # actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        rewards = np.asarray(rewards, dtype=np.float32)
        episodic_returns += rewards

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            # get path gains
            prev_path_gains = [info["prev_path_gains"] for info in infos["final_info"]]
            path_gains = [info["path_gains"] for info in infos["final_info"]]
        else:
            prev_path_gains = infos["prev_path_gains"]
            path_gains = infos["path_gains"]
        prev_path_gains = np.stack(prev_path_gains)
        path_gains = np.stack(path_gains)
        prev_path_gains = torch.as_tensor(prev_path_gains, dtype=torch.float)
        path_gains = torch.as_tensor(path_gains, dtype=torch.float)

        all_rewards[global_step, :] = rewards
        all_path_gains[global_step, ...] = path_gains

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = list(copy.deepcopy(next_obs))
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

    if is_plot:
        record_path_gain_statistics(config, envs, all_rewards, all_path_gains)

    return episodic_returns


def record_path_gain_statistics(config, envs, all_rewards, all_path_gains):

    # plot path gains
    linear_path_gains = 10 ** (all_path_gains / 10)
    sum_path_gains = np.sum(linear_path_gains, axis=-1)
    db_sum_path_gains = 10 * np.log10(sum_path_gains)
    mean_path_gains = np.mean(db_sum_path_gains, axis=1)
    std_path_gains = np.std(db_sum_path_gains, axis=1)
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.plot(mean_path_gains)
    ax.fill_between(
        range(config.eval_ep_len),
        mean_path_gains - std_path_gains,
        mean_path_gains + std_path_gains,
        alpha=0.2,
    )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Path Gain")
    ax.set_title("Path Gain")
    ax.grid()
    plt.savefig(os.path.join(config.checkpoint_dir, "path_gain.png"))

    # Plot each of dm_sum_path_gains
    fig, ax = plt.subplots(figsize=(16, 12))
    for i in range(envs.num_envs):
        ax.plot(db_sum_path_gains[:, i])
    ax.set_xlabel("Steps")
    ax.set_ylabel("Path Gain")
    ax.set_title("Path Gain")
    labels = ["env" + str(i + config.eval_seed) for i in range(envs.num_envs)]
    ax.legend(labels)
    ax.grid()
    plt.savefig(os.path.join(config.checkpoint_dir, "all_path_gain.png"))

    # plot rewards
    mean_rewards = np.mean(all_rewards, axis=1)
    std_rewards = np.std(all_rewards, axis=1)
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.plot(mean_rewards)
    ax.fill_between(
        range(config.eval_ep_len), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2
    )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Reward")
    ax.set_title("Rewards")
    ax.grid()
    plt.savefig(os.path.join(config.checkpoint_dir, "rewards.png"))

    # Save all_rewards and all_path_gains
    np.save(os.path.join(config.checkpoint_dir, "all_rewards.npy"), all_rewards)
    np.save(os.path.join(config.checkpoint_dir, "all_path_gains.npy"), all_path_gains)


if __name__ == "__main__":
    main()
