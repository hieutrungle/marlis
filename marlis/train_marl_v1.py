import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
from typing import List, Callable, Dict, Optional, Union
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
from marlis.drl.agents.marl_sac_v1 import Actor, SoftQNetwork
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

    use_compile: bool = False  # whether to use torch.dynamo compiler

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
    actor_frequency: int = 2  # the frequency of training policy (delayed)
    target_network_frequency: int = 2  # the frequency of updates for the target nerworks

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


def normalize_obs(
    flat_obs: torch.Tensor,
    envs: gym.vector.VectorEnv,
    epsilon: float = 1e-9,
    ob_type: int = -1,
):
    """
    ob_type: int = -1
        -1: global
        0: local 0
        1: local 1
    """
    if ob_type == -1:
        ac_space = envs.single_action_space
        angle_space = envs.get_attr("global_angle_space")[0]
        position_space = envs.get_attr("global_position_space")[0]
    else:
        ac_space = envs.get_attr("local_action_spaces")[0][ob_type]
        angle_space = envs.get_attr("angle_spaces")[0][ob_type]
        position_space = envs.get_attr("position_spaces")[0][ob_type]

    # angles
    angle_high = torch.tensor(angle_space.high, device=flat_obs.device, dtype=flat_obs.dtype)
    angle_low = torch.tensor(angle_space.low, device=flat_obs.device, dtype=flat_obs.dtype)
    angle_range = angle_high - angle_low
    angle_len = math.prod(angle_space.shape)
    angles = flat_obs[..., :angle_len]
    angles = (angles - angle_low) / angle_range

    pos = flat_obs[..., angle_len:]
    flat_obs = torch.cat([angles, pos], dim=-1)
    return flat_obs.float()


def make_env(config: TrainConfig, idx: int) -> Callable:

    def thunk() -> gym.Env:

        seed = config.seed
        max_episode_steps = (
            config.ep_len if config.command.lower() == "train" else config.eval_ep_len
        )
        seed += idx
        env = gym.make(
            config.env_id,
            idx=idx,
            sionna_config_file=config.sionna_config_file,
            log_string=config.name,
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
    envs = gym.vector.AsyncVectorEnv(
        [make_env(config, i) for i in range(config.num_envs)], context="spawn"
    )
    # envs = gym.vector.SyncVectorEnv([make_env(config, i) for i in range(config.num_envs)])

    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"
    ob_space = envs.single_observation_space
    ac_space = envs.single_action_space

    # Local actors
    actors = [Actor(envs=envs, ff_dim=config.ff_dim, device=config.device, idx=i) for i in range(2)]

    local_ob_spaces = envs.get_attr("local_observation_spaces")[0]
    local_ac_spaces = envs.get_attr("local_action_spaces")[0]

    tmp_local_obs = torch.randn((1, *local_ob_spaces[0].shape), device=config.device)
    torchinfo.summary(
        actors[0],
        input_data=tmp_local_obs,
        col_names=["input_size", "output_size", "num_params"],
    )

    # Local critics
    qfs = []
    qfs_target = []
    for i in range(2):
        qfs.append(
            [
                SoftQNetwork(envs=envs, ff_dim=config.ff_dim, device=config.device, idx=i)
                for _ in range(2)
            ]
        )
        qfs_target.append(
            [
                SoftQNetwork(envs=envs, ff_dim=config.ff_dim, device=config.device, idx=i)
                for _ in range(2)
            ]
        )
        for j in range(2):
            qfs_target[i][j].load_state_dict(qfs[i][j].state_dict())

    tmp_local_ac = torch.randn(1, *local_ac_spaces[0].shape, device=config.device)
    torchinfo.summary(
        qfs[0][0],
        input_data=[tmp_local_obs, tmp_local_ac],
        col_names=["input_size", "output_size", "num_params"],
    )

    # Global critic networks
    gqfs = [
        SoftQNetwork(envs=envs, ff_dim=2 * config.ff_dim, device=config.device) for _ in range(2)
    ]
    gqfs_target = [
        SoftQNetwork(envs=envs, ff_dim=2 * config.ff_dim, device=config.device) for _ in range(2)
    ]
    for i in range(2):
        gqfs_target[i].load_state_dict(gqfs[i].state_dict())

    tmp_obs = torch.randn((1, *ob_space.shape), device=config.device)
    tmp_ac = torch.randn((1, *ac_space.shape), device=config.device)
    torchinfo.summary(
        gqfs[0],
        input_data=[tmp_obs, tmp_ac],
        col_names=["input_size", "output_size", "num_params"],
    )

    # Automatic entropy tuning
    log_alphas = [torch.zeros(1, requires_grad=True, device=config.device) for _ in range(2)]

    # Optimzier setup
    alphas_optimizer = optim.AdamW(list(log_alphas), lr=config.q_lr)
    actors_optimizer = optim.AdamW(
        list(actors[0].parameters()) + list(actors[1].parameters()), lr=config.policy_lr
    )
    qfs_optimizer = optim.AdamW(
        list(qfs[0][0].parameters())
        + list(qfs[0][1].parameters())
        + list(qfs[1][0].parameters())
        + list(qfs[1][1].parameters()),
        lr=config.q_lr,
    )
    gqfs_optimizer = optim.AdamW(
        list(gqfs[0].parameters()) + list(gqfs[1].parameters()), lr=torch.tensor(config.q_lr)
    )

    # Init checkpoints
    print(f"Checkpoints dir: {config.checkpoint_dir}")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    with open(os.path.join(config.checkpoint_dir, "train_config.yaml"), "w") as f:
        pyrallis.dump(config, f)

    # Load models
    checkpoint = None
    if config.command.lower() == "eval":
        print(f"Evalation:: Loading model from {config.load_eval_model}")
        checkpoint = torch.load(config.load_eval_model, weights_only=False)
    else:
        if config.load_model != "-1":
            print(f"Resume Training:: Loading model from {config.load_model}")
            checkpoint = torch.load(config.load_model, weights_only=False)

    if checkpoint != None:
        print(f"Loading models and optimizers from checkpoint!")
        actors_states = checkpoint["actors_states"]
        for i in range(2):
            actors[i].load_state_dict(actors_states[i])

        if config.load_eval_model == "-1":
            # only load optimizers, target networks and alpha if not in eval mode
            qfs_states = checkpoint["qfs_states"]
            qfs_target_states = checkpoint["qfs_target_states"]
            gqfs_states = checkpoint["gqfs_states"]
            gqfs_target_states = checkpoint["gqfs_target_states"]
            log_alphas_states = checkpoint["log_alphas_states"]

            for i in range(2):
                for j in range(2):
                    qfs[i][j].load_state_dict(qfs_states[i][j])
                    qfs_target[i][j].load_state_dict(qfs_target_states[i][j])
                gqfs[i].load_state_dict(gqfs_states[i])
                gqfs_target[i].load_state_dict(gqfs_target_states[i])
                log_alphas[i] = log_alphas_states[i].clone().detach().requires_grad_(True)

            actors_optimizer.load_state_dict(checkpoint["actors_optimizer"])
            qfs_optimizer.load_state_dict(checkpoint["qfs_optimizer"])
            gqfs_optimizer.load_state_dict(checkpoint["gqfs_optimizer"])
            alphas_optimizer = optim.AdamW(list(log_alphas), lr=config.q_lr)
            alphas_optimizer.load_state_dict(checkpoint["alphas_optimizer"])

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

    optimizers = [actors_optimizer, qfs_optimizer, gqfs_optimizer, alphas_optimizer]
    critics = [qfs, qfs_target, gqfs, gqfs_target]
    try:
        train_agent(config, envs, actors, critics, log_alphas, optimizers, rb)
    except Exception as e:
        traceback.print_exc()
        raise e
    finally:
        rb.dump(config.replay_buffer_dir)
        wandb.finish()
        envs.close()
        envs.close_extras()


def train_agent(
    config: TrainConfig,
    envs: gym.vector.VectorEnv,
    actors: List[Actor],
    critics: List[List[SoftQNetwork]],
    log_alphas: List[torch.Tensor],
    optimizers: List[optim.Optimizer],
    rb: ReplayBuffer,
):

    wandb_init(config)

    qfs, qfs_target, gqfs, gqfs_target = critics
    actors_optimizer, qfs_optimizer, gqfs_optimizer, alphas_optimizer = optimizers

    # Entropy target
    target_entropys = [
        -torch.prod(
            torch.Tensor(envs.get_attr("local_action_spaces")[0][i].shape).to(config.device)
        )
        for i in range(2)
    ]
    alphas = [log_alpha.detach().exp() for log_alpha in log_alphas]

    def update_critics(data):
        qfs_optimizer.zero_grad()
        gqfs_optimizer.zero_grad()

        with torch.no_grad():
            next_actions0, next_log_pi0, _ = actors[0].get_action(data["local0_obs"])
            qf_next_target00 = qfs_target[0][0](data["next_local0_obs"], next_actions0)
            qf_next_target01 = qfs_target[0][1](data["next_local0_obs"], next_actions0)
            min_qf_next_target0 = torch.minimum(qf_next_target00, qf_next_target01)
            min_qf_next_target0 = min_qf_next_target0 - alphas[0] * next_log_pi0
            next_q_value0 = data["rewards"].flatten() + config.gamma * (
                1.0 - data["terminations"].float().flatten()
            ) * min_qf_next_target0.view(-1)

            next_actions1, next_log_pi1, _ = actors[1].get_action(data["local1_obs"])
            qf_next_target10 = qfs_target[1][0](data["next_local1_obs"], next_actions1)
            qf_next_target11 = qfs_target[1][1](data["next_local1_obs"], next_actions1)
            min_qf_next_target1 = torch.minimum(qf_next_target10, qf_next_target11)
            min_qf_next_target1 = min_qf_next_target1 - alphas[1] * next_log_pi1
            next_q_value1 = data["rewards"].flatten() + config.gamma * (
                1.0 - data["terminations"].float().flatten()
            ) * min_qf_next_target1.view(-1)

            next_actions = torch.cat([next_actions0, next_actions1], dim=-1)
            gqf_next_target0 = gqfs_target[0](data["next_global_obs"], next_actions)
            gqf_next_target1 = gqfs_target[1](data["next_global_obs"], next_actions)
            min_gqf_next_target = torch.minimum(gqf_next_target0, gqf_next_target1)
            min_gqf_next_target = (
                min_gqf_next_target - alphas[0] * next_log_pi0 - alphas[1] * next_log_pi1
            )
            next_gq_value = data["rewards"].flatten() + config.gamma * (
                1.0 - data["terminations"].float().flatten()
            ) * min_gqf_next_target.view(-1)

        qf_values00 = qfs[0][0](data["local0_obs"], data["local0_actions"])
        qf_loss00 = F.mse_loss(qf_values00.view(-1), next_q_value0)

        qf_values01 = qfs[0][1](data["local0_obs"], data["local0_actions"])
        qf_loss01 = F.mse_loss(qf_values01.view(-1), next_q_value0)

        qf_values10 = qfs[1][0](data["local1_obs"], data["local1_actions"])
        qf_loss10 = F.mse_loss(qf_values10.view(-1), next_q_value1)

        qf_values11 = qfs[1][1](data["local1_obs"], data["local1_actions"])
        qf_loss11 = F.mse_loss(qf_values11.view(-1), next_q_value1)

        qf_loss = qf_loss00 + qf_loss01 + qf_loss10 + qf_loss11
        qf_loss.backward()
        qfs_optimizer.step()

        gqf_values0 = gqfs[0](data["global_obs"], data["global_actions"])
        gqf_loss0 = F.mse_loss(gqf_values0.view(-1), next_gq_value)
        gqf_values1 = gqfs[1](data["global_obs"], data["global_actions"])
        gqf_loss1 = F.mse_loss(gqf_values1.view(-1), next_gq_value)
        gqf_loss = gqf_loss0 + gqf_loss1
        gqf_loss.backward()
        gqfs_optimizer.step()

        return TensorDict(
            qf_loss=qf_loss.detach(),
            gqf_loss=gqf_loss.detach(),
        )

    def update_actors(data):
        actors_optimizer.zero_grad()
        pi0, log_pi0, _ = actors[0].get_action(data["local0_obs"])
        qf_values00 = qfs[0][0](data["local0_obs"], pi0)
        qf_values01 = qfs[0][1](data["local0_obs"], pi0)
        min_qf_values0 = torch.minimum(qf_values00, qf_values01)
        actor_loss0 = (alphas[0] * log_pi0 - min_qf_values0).mean()

        pi1, log_pi1, _ = actors[1].get_action(data["local1_obs"])
        qf_values10 = qfs[1][0](data["local1_obs"], pi1)
        qf_values11 = qfs[1][1](data["local1_obs"], pi1)
        min_qf_values1 = torch.minimum(qf_values10, qf_values11)
        actor_loss1 = (alphas[1] * log_pi1 - min_qf_values1).mean()

        # Add global critic loss
        pi = torch.cat([pi0, pi1], dim=-1)
        gqf_values0 = gqfs[0](data["global_obs"], pi)
        gqf_values1 = gqfs[1](data["global_obs"], pi)
        gqf_values = torch.minimum(gqf_values0, gqf_values1)

        actor_loss = actor_loss0 + actor_loss1 - gqf_values.mean()

        actor_loss.backward()
        actors_optimizer.step()

        alphas_optimizer.zero_grad()
        with torch.no_grad():
            _, log_pi0, _ = actors[0].get_action(data["local0_obs"])
            _, log_pi1, _ = actors[1].get_action(data["local1_obs"])
        alpha_loss0 = (log_alphas[0] * (-log_pi0 - target_entropys[0])).mean()
        alpha_loss1 = (log_alphas[1] * (-log_pi1 - target_entropys[1])).mean()
        alpha_loss = alpha_loss0 + alpha_loss1
        alpha_loss.backward()
        alphas_optimizer.step()

        return TensorDict(
            actor_loss=actor_loss.detach(),
            alpha_loss=alpha_loss.detach(),
            qf0_val=min_qf_values0.detach(),
            qf1_val=min_qf_values1.detach(),
            gqf_val=gqf_values.detach(),
            log_pi0=log_pi0.detach(),
            log_pi1=log_pi1.detach(),
        )

    if config.use_compile:
        update_critics = torch.compile(update_critics, dynamic=False)
        update_actors = torch.compile(update_actors, dynamic=False)
    # update_critics = torch.compile(update_critics, dynamic=False, fullgraph=True)
    # update_actors = torch.compile(update_actors, dynamic=False, fullgraph=True)

    # training loop
    last_step = config.start_step + config.total_timesteps
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=envs.num_envs)

    if config.command.lower() == "train":
        pbar = tqdm.tqdm(
            range(config.start_step, config.start_step + config.total_timesteps + 1),
            dynamic_ncols=True,
            initial=config.start_step,
            total=config.start_step + config.total_timesteps + 1,
        )
        # info: {'env_idx': [num_envs, local_obs]}
        obs, infos = envs.reset(options={"start_init": True, "eval_mode": False})
    else:
        pbar = tqdm.tqdm(range(config.eval_ep_len), dynamic_ncols=True)
        obs, infos = envs.reset(options={"start_init": True, "eval_mode": True})

    # shape: [num_envs, [num_reflectors, local_ob_dim]]
    local_obs = np.array([ob for ob in infos["reset_local_obs"]], dtype=np.float32)
    # shape: [num_reflector, num_envs, local_ob_dim]
    local_obs = np.transpose(local_obs, (1, 0, 2))

    desc = ""
    metric_desc = ""

    for global_step in pbar:
        # ALGO LOGIC: action logic
        if config.start_step == 0 and global_step < config.learning_starts:
            # [num_envs, global_ac_dim], global_ac_dim = num_reflectors * local_ac_dim
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            # ` TODO: put local action here
            # [num_envs, global_ac_dim], global_ac_dim = num_reflectors * local_ac_dim
            actions = []
            # shape: [num_reflector, num_envs, local_ob_dim]
            torch_local_obs = torch.Tensor(local_obs.copy()).float().to(config.device)
            torch_local_obs[0] = normalize_obs(torch_local_obs[0], envs, ob_type=0)
            torch_local_obs[1] = normalize_obs(torch_local_obs[1], envs, ob_type=1)
            for i in range(2):
                # [num_envs, local_ac_dim]
                with torch.no_grad():
                    local_actions, _, _ = actors[i].get_action(torch_local_obs[i].to(config.device))
                    actions.append(local_actions.detach().cpu().numpy())
            # [num_reflector, num_envs, local_ac_dim] -> [num_envs, num_reflector, local_ac_dim]
            actions = np.transpose(np.array(actions), (1, 0, 2))
            # action shape: [num_envs, num_reflector * local_ac_dim]
            actions = np.reshape(actions, (envs.num_envs, -1))
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

        # ENV: handle `final_observation`
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

            # get path gains
            prev_path_gains = [info["prev_path_gains"] for info in infos["final_info"]]
            path_gains = [info["path_gains"] for info in infos["final_info"]]
            next_local_obs = np.array([ob for ob in infos["reset_local_obs"]], dtype=np.float32)

        else:
            prev_path_gains = infos["prev_path_gains"]
            path_gains = infos["path_gains"]
            next_local_obs = np.array([ob for ob in infos["next_local_obs"]], dtype=np.float32)

        prev_path_gains = np.stack(prev_path_gains)
        path_gains = np.stack(path_gains)
        prev_path_gains = torch.as_tensor(prev_path_gains, dtype=torch.float)
        path_gains = torch.as_tensor(path_gains, dtype=torch.float)

        real_next_obs = list(copy.deepcopy(next_obs))
        real_next_local_obs = list(copy.deepcopy(next_local_obs))
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
                real_next_local_obs[idx] = np.array(
                    [ob for ob in infos["final_info"][idx]["next_local_obs"]], dtype=np.float32
                )
        real_next_obs = np.array(real_next_obs)
        real_next_local_obs = np.array(real_next_local_obs)

        # LOCAL OBS: Reshape real_next_local_obs
        # shape: [num_envs, [num_reflectors, local_ob_dim]]
        real_next_local_obs = np.array([ob for ob in real_next_local_obs], dtype=np.float32)
        # shape: [num_reflector, num_envs, local_ob_dim]
        real_next_local_obs = np.transpose(real_next_local_obs, (1, 0, 2))

        rewards = np.asarray(rewards, dtype=np.float32)[..., None]
        terminations = np.asarray(terminations, dtype=np.float32)[..., None]
        truncations = np.asarray(truncations, dtype=np.float32)[..., None]
        local_actions = np.reshape(actions, (envs.num_envs, 2, -1))
        local_actions = np.transpose(local_actions, (1, 0, 2))

        transition = TensorDict(
            global_obs=obs,
            global_actions=actions,
            local0_obs=local_obs[0],
            local1_obs=local_obs[1],
            local0_actions=local_actions[0],
            local1_actions=local_actions[1],
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            prev_path_gains=prev_path_gains,
            path_gains=path_gains,
            next_global_obs=real_next_obs,
            next_local0_obs=real_next_local_obs[0],
            next_local1_obs=real_next_local_obs[1],
            batch_size=obs.shape[0],
        )
        rb.extend(transition)

        # ENV: store transition
        obs = next_obs
        local_obs = next_local_obs
        local_obs = np.array([ob for ob in local_obs], dtype=np.float32)
        # shape: [num_reflector, num_envs, local_ob_dim]
        local_obs = np.transpose(local_obs, (1, 0, 2))

        # ALGO LOGIC: training.
        if global_step > config.learning_starts and config.load_eval_model == "-1":
            # ALGO LOGIC: update critics and actors
            log_infos = {}
            timer = time.time()
            for j in range(config.n_updates):
                # Normalize data
                data = rb.sample()
                data = {
                    k: torch.as_tensor(v, device=config.device, dtype=torch.float)
                    for k, v in data.items()
                }
                data["global_obs"] = normalize_obs(data["global_obs"], envs, ob_type=-1)
                data["local0_obs"] = normalize_obs(data["local0_obs"], envs, ob_type=0)
                data["local1_obs"] = normalize_obs(data["local1_obs"], envs, ob_type=1)
                data["next_global_obs"] = normalize_obs(data["next_global_obs"], envs, ob_type=-1)
                data["next_local0_obs"] = normalize_obs(data["next_local0_obs"], envs, ob_type=0)
                data["next_local1_obs"] = normalize_obs(data["next_local1_obs"], envs, ob_type=1)
                data = TensorDict(data)

                # Update critics
                critic_infos = update_critics(data)
                log_infos.update(critic_infos)

                if j % config.actor_frequency == 1:  # TD 3 Delayed update support
                    for _ in range(config.actor_frequency):
                        # compensate for the delay by doing 'actor_update_interval' instead of 1
                        log_infos.update(update_actors(data))

                        for k in range(len(log_alphas)):
                            with torch.no_grad():
                                log_alphas[k].clamp_(-5.0, 1.0)
                                alphas[k].copy_(log_alphas[k].detach().exp())
                                alphas[k] = torch.clamp(alphas[k], 0.1, 0.95)

                # update the target networks
                if global_step % config.target_network_frequency == 0:
                    for k in range(2):
                        for qf, qf_target in zip(qfs[k], qfs_target[k]):
                            update_target_network(qf, qf_target, config)
                    for gqf, gqf_target in zip(gqfs, gqfs_target):
                        update_target_network(gqf, gqf_target, config)

            train_time = time.time() - timer

            with torch.no_grad():
                logs = {
                    "train_reward/path_gain": path_gains.mean(),
                    "train_reward/path_gain_std": path_gains.std(),
                    "train_reward/path_gain_diff": (path_gains - prev_path_gains).mean(),
                    "train_reward/reward_mean": rewards.mean(),
                    "train_reward/reward_std": rewards.std(),
                    "train/actor_loss": log_infos["actor_loss"].mean().item(),
                    "train/alpha_loss": log_infos["alpha_loss"].mean().item(),
                    "train/qf_loss": log_infos["qf_loss"].mean().item(),
                    "train/gqf_loss": log_infos["gqf_loss"].mean().item(),
                    "train/qf0": log_infos["qf0_val"].mean().item(),
                    "train/qf1": log_infos["qf1_val"].mean().item(),
                    "train/gqf": log_infos["gqf_val"].mean().item(),
                    "train/alpha0": alphas[0].item(),
                    "train/alpha1": alphas[1].item(),
                    "train/log_alpha0": log_alphas[0].clone().detach().item(),
                    "train/log_alpha1": log_alphas[1].clone().detach().item(),
                    "train/actor0_entropy": (-log_infos["log_pi0"]).mean().item(),
                    "train/actor1_entropy": (-log_infos["log_pi1"]).mean().item(),
                    "train/train_time": train_time,
                }

            wandb.log({**logs}, step=global_step)
            metric_desc = f" | actor_loss={logs['train/actor_loss']: 4.3f} | qf_loss={logs['train/qf_loss']: 4.3f}"

            # MODULE: Save modules
            if global_step % config.save_interval == 0 or global_step == last_step - 1:
                actors_states = [actor.state_dict() for actor in actors]
                qfs_states = [[qf.state_dict() for qf in qfs[i]] for i in range(2)]
                qfs_target_states = [[qf.state_dict() for qf in qfs_target[i]] for i in range(2)]
                gqfs_states = [gqf.state_dict() for gqf in gqfs]
                gqfs_target_states = [gqf.state_dict() for gqf in gqfs_target]
                log_alphas_states = [log_alpha.clone().detach() for log_alpha in log_alphas]

                saved_dict = {
                    "actors_states": actors_states,
                    "qfs_states": qfs_states,
                    "qfs_target_states": qfs_target_states,
                    "gqfs_states": gqfs_states,
                    "gqfs_target_states": gqfs_target_states,
                    "log_alphas_states": log_alphas_states,
                    "actors_optimizer": actors_optimizer.state_dict(),
                    "qfs_optimizer": qfs_optimizer.state_dict(),
                    "gqfs_optimizer": gqfs_optimizer.state_dict(),
                    "alphas_optimizer": alphas_optimizer.state_dict(),
                }
                torch.save(
                    saved_dict,
                    os.path.join(config.checkpoint_dir, f"model_{global_step}.pth"),
                )
                torch.save(
                    saved_dict,
                    os.path.join(config.checkpoint_dir, f"model.pth"),
                )

        pbar.set_description(desc + metric_desc)


def update_target_network(qf, qf_target, config):
    for param, target_param in zip(qf.parameters(), qf_target.parameters()):
        target_param.data.copy_(config.tau * param.data + (1 - config.tau) * target_param.data)


if __name__ == "__main__":
    main()
