import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation

from typing import Tuple, Optional
from collections import OrderedDict
import subprocess
import time
import numpy as np
from gymnasium import Env, spaces
import pickle
import glob
import math
import copy
from marlis.utils import utils
from marlis.blender_script import shared_utils
from marlis import sigmap
from sionna.channel import (
    cir_to_ofdm_channel,
    subcarrier_frequencies,
    time_lag_discrete_time_channel,
    cir_to_time_channel,
    time_to_ofdm_channel,
)
import tensorflow as tf


class SharedAPV1(Env):

    def __init__(
        self,
        idx: int,
        sionna_config_file: str,
        log_string: str = "SharedAPV1",
        seed: int = 0,
        **kwargs,
    ):
        super(SharedAPV1, self).__init__()

        self.idx = idx
        self.log_string = log_string
        self.seed = seed + idx
        self.np_rng = np.random.default_rng(self.seed)

        tf.random.set_seed(self.seed)
        print(f"using GPU: {tf.config.experimental.list_physical_devices('GPU')}")

        self.sionna_config = utils.load_config(sionna_config_file)

        # positions
        self.rx_pos = np.array(self.sionna_config["rx_positions"], dtype=np.float32)
        self.rt_pos = np.array(self.sionna_config["rt_positions"], dtype=np.float32)
        self.tx_pos = np.array(self.sionna_config["tx_positions"], dtype=np.float32)

        # orient the tx
        tx_orientations = []
        for i in range(len(self.rt_pos)):
            r, theta, phi = compute_rot_angle(self.tx_pos[i], self.rt_pos[i])
            tx_orientations.append([phi, theta - math.pi / 2, 0.0])
        self.sionna_config["tx_orientations"] = tx_orientations

        # Set up logging
        self.current_time = "_" + time.strftime("%d-%m-%Y_%H-%M-%S")

        # Set up action and observation space
        # Load the reflector configuration, angle is in radians
        reflector_configs = shared_utils.get_config_shared_ap()
        self.theta_configs = reflector_configs[0]
        self.phi_configs = reflector_configs[1]
        self.num_groups = reflector_configs[2]
        self.num_elements_per_group = reflector_configs[3]

        # angles = [theta, phi] for each tile
        # theta: zenith angle, phi: azimuth angle
        self.init_thetas = []
        self.init_phis = []
        self.angle_spaces = []
        theta_highs = []
        phi_highs = []
        theta_lows = []
        phi_lows = []

        for i in range(len(self.rt_pos)):
            init_theta = self.theta_configs[0][i]
            init_phi = self.phi_configs[0][i]
            # init_per_group = [init_phi] + [init_theta] * self.num_elements_per_group
            # self.init_angles.append(np.concatenate([init_per_group] * self.num_groups))

            theta_high = self.theta_configs[2][i]
            phi_high = self.phi_configs[2][i]
            per_group_high = [phi_high] + [theta_high] * self.num_elements_per_group
            angle_high = np.concatenate([per_group_high] * self.num_groups)
            theta_low = self.theta_configs[1][i]
            phi_low = self.phi_configs[1][i]
            per_group_low = [phi_low] + [theta_low] * self.num_elements_per_group
            angle_low = np.concatenate([per_group_low] * self.num_groups)
            self.angle_spaces.append(spaces.Box(low=angle_low, high=angle_high, dtype=np.float32))

            # storage
            self.init_thetas.append(init_theta)
            self.init_phis.append(init_phi)
            theta_highs.append(theta_high)
            phi_highs.append(phi_high)
            theta_lows.append(theta_low)
            phi_lows.append(phi_low)

        self.global_angle_space = spaces.Box(
            low=np.concatenate([space.low for space in self.angle_spaces]),
            high=np.concatenate([space.high for space in self.angle_spaces]),
            dtype=np.float32,
        )

        # position space
        self.position_spaces = []
        for i in range(len(self.rt_pos)):
            self.position_spaces.append(
                spaces.Box(
                    low=-100.0,
                    high=100.0,
                    shape=(len(np.concatenate([self.rx_pos, self.rt_pos[i : i + 1]]).flatten()),),
                    dtype=np.float32,
                )
            )
        self.global_position_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(len(np.concatenate([self.rx_pos, self.rt_pos]).flatten()),),
            dtype=np.float32,
        )

        # focal vecs space <-> action space
        # Action is a changes in focals [delta_r, delta_theta, _delta_phi] for each group
        # focals = [r, theta, phi] for each group
        self._initialize_focal_spaces(theta_highs, phi_highs, theta_lows, phi_lows)

        # Observation space
        self._initialize_observation_space()

        # action space
        self._initialize_action_space()

        # initialize local observation space
        self._initialize_local_observation_spaces()

        # initialize local action space
        self._initialize_local_action_spaces()

        # Reward set up
        self.taken_steps = 0.0
        self.prev_gains = [0.0 for _ in range(len(self.rx_pos))]
        self.cur_gains = [0.0 for _ in range(len(self.rx_pos))]

        # range for new rx positions
        # self.rx_pos_range = [[-4.0, 4.0], [-19.5, -10.0]]  # x  # y
        self.obstacle_pos = [[2.37, -17.656], [-2.6, -11.945], [1.86, -10.7654], [-1.71, -15.6846]]

        # range for new rx positions: [[x_min, x_max], [y_min, y_max]]
        self.rx_pos_ranges = [
            [[0.2, 3.5], [-1.0, 2.0]],
            [[0.2, 3.5], [-5.5, -1.5]],
            [[-3.5, -0.2], [-1.0, 2.0]],
            [[-3.5, -0.2], [-5.5, -1.5]],
        ]

        # range for rt positions: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        self.rt_pos_delta_ranges = [
            [[-0.1, 0.1], [-0.1, 0.1], [-0.2, 0.2]],
            [[-0.1, 0.1], [-0.1, 0.1], [-0.2, 0.2]],
        ]

        self.default_sionna_config = copy.deepcopy(self.sionna_config)

        self.eval_mode = False

    def _initialize_focal_spaces(self, theta_highs, phi_highs, theta_lows, phi_lows):
        r_high = 40.0
        r_low = 5.0
        self.focal_spaces = []
        for i in range(len(self.rt_pos)):
            theta_high = theta_highs[i]
            theta_low = theta_lows[i]
            phi_high = phi_highs[i]
            phi_low = phi_lows[i]
            self.focal_spaces.append(
                spaces.Box(
                    low=np.asarray([r_low, theta_low, phi_low] * self.num_groups),
                    high=np.asarray([r_high, theta_high, phi_high] * self.num_groups),
                    dtype=np.float32,
                )
            )

    def _initialize_action_space(self):
        action_space_shape = tuple((3 * self.num_groups,))
        action_spaces = []
        for i in range(len(self.rt_pos)):
            action_spaces.append(
                spaces.Box(low=-1.0, high=1.0, shape=action_space_shape, dtype=np.float32)
            )
        low = np.concatenate([space.low for space in action_spaces])
        high = np.concatenate([space.high for space in action_spaces])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _initialize_observation_space(self):
        low = np.concatenate([self.global_angle_space.low, self.global_position_space.low])
        high = np.concatenate([self.global_angle_space.high, self.global_position_space.high])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _initialize_local_observation_spaces(self):
        local_observation_spaces = []
        for i in range(len(self.rt_pos)):
            local_angle_space = np.concatenate(
                [self.angle_spaces[i].low, self.angle_spaces[i].high]
            )
            local_position_space = np.concatenate(
                [self.position_spaces[i].low, self.position_spaces[i].high]
            )
            low = np.concatenate(
                [
                    local_angle_space[: len(local_angle_space) // 2],
                    local_position_space[: len(local_position_space) // 2],
                ]
            )
            high = np.concatenate(
                [
                    local_angle_space[len(local_angle_space) // 2 :],
                    local_position_space[len(local_position_space) // 2 :],
                ]
            )
            local_observation_spaces.append(spaces.Box(low=low, high=high, dtype=np.float32))
        self.local_observation_spaces = local_observation_spaces

    def _initialize_local_action_spaces(self):
        action_space_shape = tuple((3 * self.num_groups,))
        local_action_spaces = []
        for i in range(len(self.rt_pos)):
            local_action_spaces.append(
                spaces.Box(low=-1.0, high=1.0, shape=action_space_shape, dtype=np.float32)
            )
        self.local_action_spaces = local_action_spaces

    def _cal_area(self, pt1, pt2, pt3):

        return abs(
            (pt1[0] * (pt2[1] - pt3[1]) + pt2[0] * (pt3[1] - pt1[1]) + pt3[0] * (pt1[1] - pt2[1]))
            / 2.0
        )

    def _cal_distance(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def _is_inside(self, border, target):
        # borrow and modified from https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon
        degree = 0
        for i in range(len(border)):
            a = border[i]
            b = border[(i + 1) % len(border)]

            # calculate distance of vector
            e1 = self._cal_distance(a[0], a[1], b[0], b[1])
            e2 = self._cal_distance(target[0], target[1], a[0], a[1])
            e3 = self._cal_distance(target[0], target[1], b[0], b[1])

            # calculate direction of vector
            ta_x = a[0] - target[0]
            ta_y = a[1] - target[1]
            tb_x = b[0] - target[0]
            tb_y = b[1] - target[1]

            cross = tb_y * ta_x - tb_x * ta_y
            clockwise = cross < 0

            # calculate sum of angles
            if clockwise:
                degree += math.degrees(math.acos((e2 * e2 + e3 * e3 - e1 * e1) / (2.0 * e2 * e3)))
            else:
                degree -= math.degrees(math.acos((e2 * e2 + e3 * e3 - e1 * e1) / (2.0 * e2 * e3)))

        if abs(abs(round(degree)) - 360.0) <= 2.0:
            return True
        return False

    def _is_eligible(self, pt, obstacle_pos, rx_pos):
        """
        Check if the position is eligible for new rx_pos.
        This function checks if the new rx_pos is not too close to obstacles and not too close to existing rx_pos.
        """
        # append new rx_pos that are not too close to obstacles
        is_obs_overlaped = False
        for pos in obstacle_pos:
            if np.linalg.norm(np.array(pos[:2]) - np.array(pt)) < 0.7:
                is_obs_overlaped = True
                break

        # make sure that distance between rx_pos is at least 1
        if not is_obs_overlaped:
            if not any(np.linalg.norm(np.array(pos[:2]) - np.array(pt)) < 1.5 for pos in rx_pos):
                return True
        return False

    def _prepare_rx_positions(self):
        # Pick ranges from rx_pos_ranges
        rx_pos = []
        rx_pos_ranges = self.np_rng.choice(self.rx_pos_ranges, len(self.rx_pos), replace=False)
        for rx_pos_range in rx_pos_ranges:
            pt = None
            while not pt:
                x = self.np_rng.uniform(low=rx_pos_range[0][0], high=rx_pos_range[0][1])
                y = self.np_rng.uniform(low=rx_pos_range[1][0], high=rx_pos_range[1][1])
                tmp = [x, y]
                if self._is_eligible(tmp, [], rx_pos):
                    pt = tmp
                    rx_pos.append([x, y, 1.5])

        return rx_pos

    def _prepare_rt_delta_positions(self):
        rt_pos_deltas = []
        for rt_pos_range in self.rt_pos_delta_ranges:
            x = self.np_rng.uniform(low=rt_pos_range[0][0], high=rt_pos_range[0][1])
            y = self.np_rng.uniform(low=rt_pos_range[1][0], high=rt_pos_range[1][1])
            z = self.np_rng.uniform(low=rt_pos_range[2][0], high=rt_pos_range[2][1])
            rt_pos_deltas.append([x, y, z])
        return np.array(rt_pos_deltas, dtype=np.float32)

    def reset(self, seed: int = None, options: dict = None) -> Tuple[dict, dict]:
        super().reset(seed=seed)

        self.sionna_config = copy.deepcopy(self.default_sionna_config)

        # reset rx_pos
        rx_pos = self._prepare_rx_positions()
        self.sionna_config["rx_positions"] = rx_pos
        self.rx_pos = np.array(rx_pos, dtype=np.float32)

        # reset rt_pos
        rt_pos_deltas = self._prepare_rt_delta_positions()
        self.rt_pos = np.array(self.rt_pos, dtype=np.float32) + rt_pos_deltas
        self.sionna_config["rt_positions"] = list(self.rt_pos)

        # orient the tx
        tx_orientations = []
        for i in range(len(self.rt_pos)):
            r, theta, phi = compute_rot_angle(self.tx_pos[i], self.rt_pos[i])
            tx_orientations.append([phi, theta - math.pi / 2, 0.0])
        self.sionna_config["tx_orientations"] = tx_orientations

        # We have 2 reflector -> 2 sets of Spherical_focal_vecs
        # theta_inits = [np.deg2rad(90.0), np.deg2rad(90.0)]
        # phi_inits = [np.deg2rad(30.0), np.deg2rad(-30.0)]

        # focal_spaces: contains the focal space for each reflector (implicitly represents the angles, observation)
        # focals: contain the focal action for each reflector (actual values to get the angles of each reflector)
        # angles: contain the angles of each reflector (from 'focals')
        # positions: UE positions + Reflector positions

        # noise to focals
        start_init = False
        if options is not None:
            start_init = options.get("start_init", False)
            print(f"\nRESET with start_init: {start_init}")

        self.focals = [None for _ in range(len(self.rt_pos))]
        for i in range(len(self.rt_pos)):
            low = self.focal_spaces[i].low
            high = self.focal_spaces[i].high
            center = (low + high) / 2.0
            # make low slightly higher than the old low
            low = center - (center - low) * 0.9
            high = center + (high - center) * 0.9
            if start_init:
                self.focals[i] = self.np_rng.uniform(low=low, high=high)
            else:
                self.focals[i] = self.np_rng.normal(
                    loc=(low + high) / 2.0, scale=abs(high - low) / 6.0
                )
            self.focals[i] = np.clip(self.focals[i], low, high)
        self.focals = np.asarray(self.focals, dtype=np.float32)

        # angles is a sets of angles from two reflectors
        # angles[0] is from reflector 1, angles[1] is from reflector 2
        self.angles = self._blender_step(self.focals)
        for i in range(len(self.angle_spaces)):
            self.angles[i] = np.asarray(self.angles[i], dtype=np.float32)
            self.angles[i] = np.clip(
                self.angles[i], self.angle_spaces[i].low, self.angle_spaces[i].high
            )

        if options and "eval_mode" in options:
            self.eval_mode = options["eval_mode"]
        self.prev_gains = self._run_sionna_dB(self.eval_mode)
        self.cur_gains = self.prev_gains

        # global observation
        global_angles = self.angles.flatten()
        global_positions = np.concatenate([self.rx_pos, self.rt_pos], axis=0).flatten()
        observation = np.concatenate([global_angles, global_positions], axis=-1).flatten()

        # Reflector local observation
        local_obs = [0 for _ in range(len(self.rt_pos))]
        for i in range(len(self.rt_pos)):
            local_positions = np.concatenate(
                [self.rx_pos, self.rt_pos[i : i + 1]], axis=0
            ).flatten()
            local_ob = np.concatenate(
                [self.angles[i].flatten(), local_positions], axis=-1
            ).flatten()
            local_obs[i] = local_ob
        info = {f"reset_local_obs": local_obs}

        self.taken_steps = 0.0

        return np.array(observation), info

    def step(self, action: np.ndarray, **kwargs) -> Tuple[dict, float, bool, bool, dict]:

        self.taken_steps += 1.0
        self.prev_gains = self.cur_gains

        # actions: [num_reflectors * num_groups * 3]: [r, theta, phi] for each group
        action = np.reshape(action, (len(self.rt_pos), -1))
        tmp = np.reshape(action, (len(self.rt_pos) * self.num_groups, 3))
        tmp[:, 0] = tmp[:, 0]  # r
        tmp[:, 1] = np.deg2rad(tmp[:, 1])  # theta
        tmp[:, 2] = np.deg2rad(tmp[:, 2])  # phi
        action = np.reshape(tmp, self.focals.shape)

        # focal shape: [num_reflectors, num_groups * 3]
        self.focals = self.focals + action

        low = np.asarray([space.low for space in self.focal_spaces])
        high = np.asarray([space.high for space in self.focal_spaces])
        local_out_of_bounds = [
            np.sum((focal < low[i]) + (focal > high[i]), dtype=np.float32)
            for i, focal in enumerate(self.focals)
        ]
        out_of_bounds = np.sum(local_out_of_bounds, dtype=np.float32)

        self.focals = np.clip(self.focals, low, high)

        self.angles = self._blender_step(self.focals)
        self.angles = np.asarray(self.angles, dtype=np.float32)
        # if angles values are out of bounds, print warning
        low = np.asarray([space.low for space in self.angle_spaces])
        high = np.asarray([space.high for space in self.angle_spaces])
        if np.any(self.angles < low) or np.any(self.angles > high):
            print("Warning: angles out of bounds")

        truncated = False
        if self.taken_steps > 100:
            truncated = True
        terminated = False
        self.cur_gains = self._run_sionna_dB(eval_mode=self.eval_mode)

        # np.mean(cur_gains, axis=1) -> [num_rx]
        global_reward = self._cal_reward(
            np.mean(self.prev_gains, axis=1), np.mean(self.cur_gains, axis=1), out_of_bounds
        )
        local_rewards = [
            self._cal_reward(self.prev_gains[:, i], self.cur_gains[:, i], local_out_of_bounds[i])
            for i in range(len(self.rt_pos))
        ]
        step_info = {
            "prev_path_gains": self.prev_gains,
            "path_gains": self.cur_gains,
        }

        global_angles = self.angles.flatten()
        global_positions = np.concatenate([self.rx_pos, self.rt_pos], axis=0).flatten()
        next_observation = np.concatenate([global_angles, global_positions], axis=-1).flatten()

        next_local_obs = [0 for _ in range(len(self.rt_pos))]
        for i in range(len(self.rt_pos)):
            local_positions = np.concatenate(
                [self.rx_pos, self.rt_pos[i : i + 1]], axis=0
            ).flatten()
            local_ob = np.concatenate(
                [self.angles[i].flatten(), local_positions], axis=-1
            ).flatten()
            next_local_obs[i] = local_ob
        step_info[f"next_local_obs"] = next_local_obs
        step_info[f"local_rewards"] = local_rewards

        return next_observation, global_reward, terminated, truncated, step_info

    def _cal_reward(
        self, prev_gains: np.ndarray, cur_gains: np.ndarray, out_of_bounds: float
    ) -> float:

        # path gain shape: [num_rx]
        adjusted_gain = np.mean(cur_gains)
        adjusted_gain = np.where(
            adjusted_gain < -100.0,
            (adjusted_gain + 100.0) / 10.0,
            (adjusted_gain + 100.0) / 10.0 + 1.0,
        )
        gain_diff = np.mean(cur_gains - prev_gains)

        reward = float(adjusted_gain + 0.1 * gain_diff - 0.3 * out_of_bounds) / 2.0

        return reward
        # return 0.0

    def _blender_step(self, focals: np.ndarray[float]) -> np.ndarray[float]:
        """
        Step the environment using Blender.

        If action is not given, the environment stays the same with the given angles.
        """
        # Blender export
        blender_app = utils.get_os_dir("BLENDER_APP")
        blender_dir = utils.get_os_dir("BLENDER_DIR")
        source_dir = utils.get_os_dir("SOURCE_DIR")
        assets_dir = utils.get_os_dir("ASSETS_DIR")
        tmp_dir = utils.get_os_dir("TMP_DIR")
        scene_name = f"{self.sionna_config['scene_name']}_{self.idx}"
        blender_output_dir = os.path.join(assets_dir, "blender", scene_name)

        # focal to angles
        focal_path = os.path.join(
            tmp_dir, f"data-{self.log_string}-{self.current_time}-{self.idx}.pkl"
        )
        with open(focal_path, "wb") as f:
            pickle.dump(focals, f)

        # rt_pos to rt_pos_deltas
        default_rt_pos = np.array(self.default_sionna_config["rt_positions"])
        current_rt_pos = np.array(self.rt_pos)
        rt_pos_deltas = current_rt_pos - default_rt_pos
        delta_rt_pos_path = os.path.join(
            tmp_dir, f"rt_pos_deltas-{self.log_string}-{self.current_time}-{self.idx}.pkl"
        )
        with open(delta_rt_pos_path, "wb") as f:
            pickle.dump(rt_pos_deltas, f)

        blender_script = os.path.join(source_dir, "marlis", "blender_script", "bl_data_center.py")
        blender_cmd = [
            blender_app,
            "-b",
            os.path.join(blender_dir, "models", f"{scene_name}.blend"),
            "--python",
            blender_script,
            "--",
            "-s",
            self.sionna_config["scene_name"],
            "--focal_path",
            focal_path,
            "--rt_delta_path",
            delta_rt_pos_path,
            "-o",
            blender_output_dir,
        ]
        bl_output_txt = os.path.join(tmp_dir, "bl_outputs.txt")
        subprocess.run(blender_cmd, check=True, stdout=open(bl_output_txt, "w"))
        # subprocess.run(blender_cmd, check=True)

        with open(focal_path, "rb") as f:
            angles = pickle.load(f)
        angles = np.asarray(angles, dtype=np.float32)
        return angles

    def _run_sionna_dB(self, eval_mode: bool = False) -> np.ndarray[np.complex64, float]:

        # self._prepare_geometry()
        # path gain shape: [num_rx, num_tx]
        path_gains = self._run_sionna(eval_mode=eval_mode)
        path_gain_dBs = utils.linear2dB(path_gains)
        return path_gain_dBs

    def _run_sionna(self, eval_mode: bool = False) -> Tuple[tf.Tensor, np.ndarray]:

        # Set up geometry paths for Sionna script
        assets_dir = utils.get_os_dir("ASSETS_DIR")
        scene_name = f"{self.sionna_config['scene_name']}_{self.idx}"
        blender_output_dir = os.path.join(assets_dir, "blender", scene_name)
        compute_scene_dir = os.path.join(blender_output_dir, "ceiling_idx")
        compute_scene_path = glob.glob(os.path.join(compute_scene_dir, "*.xml"))[0]
        viz_scene_dir = os.path.join(blender_output_dir, "idx")
        viz_scene_path = glob.glob(os.path.join(viz_scene_dir, "*.xml"))[0]

        sig_cmap = sigmap.engine.SignalCoverageMap(
            self.sionna_config, compute_scene_path, viz_scene_path
        )

        if eval_mode:
            coverage_map = sig_cmap.compute_cmap()

            # Path for outputing images if we want to visualize the coverage map
            img_dir = os.path.join(
                assets_dir, "images", self.log_string + self.current_time + f"_{self.idx}"
            )
            render_filename = utils.create_filename(img_dir, f"{scene_name}_00000.png")
            sig_cmap.render_to_file(coverage_map, filename=render_filename)

        sig_cmap.free_memory()

        paths = sig_cmap.compute_paths()
        # shape of a: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps]
        a, tau = paths.cir(ris=False, cluster_ris_paths=False)
        # [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size]
        h_freq = cir_to_ofdm_channel(self.sionna_config["frequency"], a, tau, normalize=False)

        path_gains = tf.reduce_mean(tf.square(tf.abs(h_freq)), axis=(0, 2, 4, 5, 6))
        # path_loss = tf.sqrt(path_loss)

        # print(f"path_loss from path computation: {utils.linear2dB(path_gains)}")

        # path_loss_avg = tf.reduce_mean(path_gains, axis=-1)
        # print(f"path_loss from path_loss_avg computation: {utils.linear2dB(path_loss_avg)}\n")

        # # h_avg_power = tf.reduce_mean(tf.abs(h_freq) ** 2)

        return path_gains


def compute_rot_angle(pt1: list, pt2: list) -> Tuple[float, float, float]:
    """Compute the rotation angles for vector pt1 to pt2."""
    x = pt2[0] - pt1[0]
    y = pt2[1] - pt1[1]
    z = pt2[2] - pt1[2]

    return cartesian2spherical(x, y, z)


def cartesian2spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    # theta: zenith angle (0, pi), phi: azimuthal angle (0, 2pi)
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.acos(z / r)
    phi = math.atan2(y, x)
    return r, theta, phi


def spherical2cartesian(r: float, theta: float, phi: float) -> Tuple[float, float, float]:
    # theta: zenith angle (0, pi), phi: azimuthal angle (0, 2pi)
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    return x, y, z
