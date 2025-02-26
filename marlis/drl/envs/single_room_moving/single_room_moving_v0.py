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


class SingleRoomMovingV0(Env):

    def __init__(
        self,
        idx: int,
        sionna_config_file: str,
        log_string: str = "SingleRoomMovingV0",
        eval_mode: bool = False,
        seed: int = 0,
        **kwargs,
    ):
        super(SingleRoomMovingV0, self).__init__()

        self.idx = idx
        self.log_string = log_string
        self.seed = seed + idx
        self.np_rng = np.random.default_rng(self.seed)

        # tf.config.experimental.set_memory_growth(
        #     tf.config.experimental.list_physical_devices("GPU")[0], True
        # )
        tf.random.set_seed(self.seed)
        print(f"using GPU: {tf.config.experimental.list_physical_devices('GPU')}")

        self.sionna_config = utils.load_config(sionna_config_file)

        ris_pos = self.sionna_config["ris_positions"][0]
        tx_pos = self.sionna_config["tx_positions"][0]
        r, theta, phi = compute_rot_angle(tx_pos, ris_pos)
        self.sionna_config["tx_orientations"] = [[phi, theta - math.pi / 2, 0.0]]

        # Set up logging
        self.current_time = "_" + time.strftime("%d-%m-%Y_%H-%M-%S")

        # Set up action and observation space
        reflector_config = shared_utils.get_reflector_config()
        self.theta_config = reflector_config[0]
        self.phi_config = reflector_config[1]
        self.num_groups = reflector_config[2]
        self.num_elements_per_group = reflector_config[3]

        # angles = [theta, phi] for each tile
        # theta: zenith angle, phi: azimuth angle
        init_theta = self.theta_config[0]
        init_phi = self.phi_config[0]
        init_per_group = [init_phi] + [init_theta] * self.num_elements_per_group
        self.init_angles = np.concatenate([init_per_group] * self.num_groups)

        # angles space
        theta_high = self.theta_config[2]
        phi_high = self.phi_config[2]
        per_group_high = [phi_high] + [theta_high] * self.num_elements_per_group
        angle_high = np.concatenate([per_group_high] * self.num_groups)
        theta_low = self.theta_config[1]
        phi_low = self.phi_config[1]
        per_group_low = [phi_low] + [theta_low] * self.num_elements_per_group
        angle_low = np.concatenate([per_group_low] * self.num_groups)
        self.angle_space = spaces.Box(low=angle_low, high=angle_high, dtype=np.float32)

        # position space
        rx_positions = np.array(self.sionna_config["rx_positions"], dtype=np.float32).flatten()
        ris_positions = np.array(self.sionna_config["ris_positions"], dtype=np.float32).flatten()
        # self.positions = np.concatenate([rx_positions, ris_positions], dtype=np.float32)
        self.positions = rx_positions
        self.position_space = spaces.Box(
            low=-100.0, high=100.0, shape=(len(self.positions),), dtype=np.float32
        )

        # channels space
        self.bandwidth = 100e6  # 100MHz
        self.maximum_delay_spread = 10e-9  # 10ns
        # self.maximum_delay_spread = 1e-6  # 1us
        # (self.l_min, self.l_max) = time_lag_discrete_time_channel(
        #     self.bandwidth, self.maximum_delay_spread
        # )
        self.l_min = 0
        self.l_max = 0
        num_rxs = len(self.sionna_config["rx_positions"])
        num_tx_ants = self.sionna_config["tx_num_rows"] * self.sionna_config["tx_num_cols"]
        self.channel_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_rxs, num_tx_ants, 2 * int(self.l_max - self.l_min + 1)),
            dtype=np.float32,
        )

        # observation space
        self.observation_space = spaces.Dict(
            OrderedDict(
                channels=self.channel_space,
                angles=self.angle_space,
                positions=self.position_space,
            )
        )

        # action space
        action_space_shape = tuple((3 * self.num_groups,))
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=action_space_shape, dtype=np.float32
        )

        # focal vecs space for action space
        self.init_focal_vecs = np.asarray([10.0, init_theta, np.deg2rad(125)] * self.num_groups)
        # self.init_focal_vecs = np.asarray([10.0, init_theta, init_phi] * self.num_groups)
        r_high = 40.0
        focal_vec_high = np.asarray([r_high, theta_high, phi_high] * self.num_groups)
        r_low = 5.0
        focal_vec_low = np.asarray([r_low, theta_low, phi_low] * self.num_groups)
        self.focal_vec_space = spaces.Box(low=focal_vec_low, high=focal_vec_high, dtype=np.float32)

        # noise for init
        self.focal_noise_high = np.asarray(
            [5.0, np.deg2rad(10.0), np.deg2rad(10.0)] * self.num_groups
        )
        self.focal_noise_low = -self.focal_noise_high

        # Action is a focal_vec [delta_r, delta_theta, _delta_phi] for each group
        # spherical_focal_vecs = [r, theta, phi] for each group
        self.spherical_focal_vecs = None
        self.angles = None
        self.channels = None

        self.taken_steps = 0.0
        self.prev_gains = [0.0 for _ in range(len(self.sionna_config["rx_positions"]))]
        self.cur_gains = [0.0 for _ in range(len(self.sionna_config["rx_positions"]))]
        self.ep_step = 0

        self.info = {}
        self.eval_mode = eval_mode
        self.default_positions = copy.deepcopy(self.positions)
        self.default_sionna_config = copy.deepcopy(self.sionna_config)

        # range for new rx positions
        self.rangee = [[-15.0, -7.0], [-7.0, 3.5]]  # x  # y
        self.restricted_areas = [
            [[-5.01257, 0.896045], [-9.01329, -3.62248], [-3.87024, -7.24338]],
            [[-12.3042, 0.034565], [-14.487, -1.39775], [-12.2953, -2.18156]],
        ]

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

    def reset(self, seed: int = None, options: dict = None) -> Tuple[dict, dict]:
        super().reset(seed=seed)

        self.sionna_config = copy.deepcopy(self.default_sionna_config)

        # append new rx_positions that are not in the restricted_areas
        rx_positions = []
        while len(rx_positions) < len(self.sionna_config["rx_positions"]):
            x = self.np_rng.uniform(low=self.rangee[0][0], high=self.rangee[0][1])
            y = self.np_rng.uniform(low=self.rangee[1][0], high=self.rangee[1][1])
            pt = [x, y]
            is_inside = False
            for area in self.restricted_areas:
                if self._is_inside(area, pt):
                    is_inside = True
                    break

            # make sure that distance between rx_positions is at least 1
            if not is_inside:
                too_close = False
                for pos in rx_positions:
                    if np.linalg.norm(np.array(pos[:2]) - np.array(pt)) < 1.0:
                        too_close = True
                        break
                if not too_close:
                    rx_positions.append([x, y, 1.5])

        self.sionna_config["rx_positions"] = rx_positions
        self.positions = np.asarray(rx_positions, dtype=np.float32).flatten()

        start_init = False
        if options is not None:
            start_init = options.get("start_init", False)
            print(f"\nRESET with start_init: {start_init}")

        # noise to spherical_focal_vecs
        if start_init:
            noise = self.np_rng.uniform(low=self.focal_noise_low, high=self.focal_noise_high)
            self.spherical_focal_vecs = np.asarray(
                [15.0, np.deg2rad(90), np.deg2rad(135)] * self.num_groups
            )
            self.spherical_focal_vecs += noise
        else:
            low = self.focal_vec_space.low
            high = self.focal_vec_space.high
            self.spherical_focal_vecs = self.np_rng.normal(
                loc=(low + high) / 2.0, scale=abs(high - low) / 9.0
            )

        # tmp = np.reshape(copy.deepcopy(self.spherical_focal_vecs), (self.num_groups, 3))
        # tmp[:, 1:] = np.rad2deg(tmp[:, 1:])
        # print(f"init_focal_vecs: {tmp}")

        # self.spherical_focal_vecs += noise
        self.spherical_focal_vecs = np.clip(
            self.spherical_focal_vecs, self.focal_vec_space.low, self.focal_vec_space.high
        )
        # tmp = np.reshape(copy.deepcopy(self.spherical_focal_vecs), (self.num_groups, 3))
        # tmp[:, 1:] = np.rad2deg(tmp[:, 1:])
        # print(f"init_focal_vecs: {tmp}")
        self.angles = self._blender_step(self.spherical_focal_vecs)
        # print(f"angles: {np.rad2deg(self.angles).reshape(-1, 8)}")
        self.angles = np.clip(
            self.angles, self.angle_space.low, self.angle_space.high, dtype=np.float32
        )

        self.channels, self.prev_gains = self._run_sionna_dB(eval_mode=self.eval_mode)
        self.cur_gains = self.prev_gains

        real_channels = np.asarray(self.channels.real, dtype=np.float32)
        imag_channels = np.asarray(self.channels.imag, dtype=np.float32)
        channels = np.concatenate([real_channels, imag_channels], axis=-1)
        observation = OrderedDict(channels=channels, angles=self.angles, positions=self.positions)

        self.taken_steps = 0.0

        return observation, {}

    def step(self, action: np.ndarray, **kwargs) -> Tuple[dict, float, bool, bool, dict]:

        self.taken_steps += 1.0
        self.prev_gains = self.cur_gains

        # action: [num_groups * 3]: num_groups * [phi, theta, r]
        tmp = np.reshape(action, (self.num_groups, 3))
        tmp[:, 0] = tmp[:, 0]
        tmp[:, 1] = np.deg2rad(tmp[:, 1])
        tmp[:, 2] = np.deg2rad(tmp[:, 2])
        action = np.reshape(tmp, action.shape)

        self.spherical_focal_vecs = self.spherical_focal_vecs + action
        out_of_bounds = np.sum(
            (self.spherical_focal_vecs < self.focal_vec_space.low)
            + (self.spherical_focal_vecs > self.focal_vec_space.high),
            dtype=np.float32,
        )
        self.spherical_focal_vecs = np.clip(
            self.spherical_focal_vecs, self.focal_vec_space.low, self.focal_vec_space.high
        )

        # tmp = np.reshape(copy.deepcopy(self.spherical_focal_vecs), (self.num_groups, 3))
        # tmp[:, 1:] = np.rad2deg(tmp[:, 1:])
        # print(f"spherical_focal_vecs: {tmp}")

        self.angles = self._blender_step(self.spherical_focal_vecs)
        self.angles = np.asarray(self.angles, dtype=np.float32)
        # print(f"angles: {np.rad2deg(self.angles).reshape(-1, 8)}")
        # if angles values are out of bounds, print warning
        if np.any(self.angles < self.angle_space.low) or np.any(
            self.angles > self.angle_space.high
        ):
            print("Warning: angles out of bounds")

        truncated = False
        if self.taken_steps > 100:
            truncated = True
        terminated = False
        self.channels, self.cur_gains = self._run_sionna_dB(eval_mode=self.eval_mode)

        real_channels = np.asarray(self.channels.real, dtype=np.float32)
        imag_channels = np.asarray(self.channels.imag, dtype=np.float32)
        channels = np.concatenate([real_channels, imag_channels], axis=-1)
        next_observation = OrderedDict(
            channels=channels, angles=self.angles, positions=self.positions
        )
        # next_observation = (real_channels, imag_channels, self.angles, self.positions)

        reward = self._cal_reward(self.prev_gains, self.cur_gains, out_of_bounds)

        step_info = {
            "prev_path_gains": self.prev_gains,
            "path_gains": self.cur_gains,
        }

        return next_observation, reward, terminated, truncated, step_info

    def _cal_reward(
        self, prev_gains: np.ndarray, cur_gains: np.ndarray, out_of_bounds: float
    ) -> float:

        adjusted_gain = np.mean(cur_gains)
        adjusted_gain = np.where(
            adjusted_gain < -95.0,
            (adjusted_gain + 95.0) / 10.0,
            (adjusted_gain + 95.0) / 5.0 + 1.5,
        )
        gain_diff = np.mean(cur_gains - prev_gains)

        reward = float(adjusted_gain + 0.03 * gain_diff - 0.3 * out_of_bounds) / 2.0

        return reward

    def _blender_step(self, spherical_focal_vecs: np.ndarray[float]) -> np.ndarray[float]:
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

        data_path = os.path.join(
            tmp_dir, f"data-{self.log_string}-{self.current_time}-{self.idx}.pkl"
        )
        with open(data_path, "wb") as f:
            pickle.dump(spherical_focal_vecs, f)

        blender_script = os.path.join(source_dir, "marlis", "blender_script", "bl_drl.py")
        blender_cmd = [
            blender_app,
            "-b",
            os.path.join(blender_dir, "models", f"{scene_name}.blend"),
            "--python",
            blender_script,
            "--",
            "-i",
            data_path,
            "-o",
            blender_output_dir,
        ]
        bl_output_txt = os.path.join(tmp_dir, "bl_outputs.txt")
        # subprocess.run(blender_cmd, check=True)
        subprocess.run(blender_cmd, check=True, stdout=open(bl_output_txt, "w"))

        with open(data_path, "rb") as f:
            angles = pickle.load(f)
        angles = np.asarray(angles, dtype=np.float32)
        return angles

    def _run_sionna_dB(self, eval_mode: bool = False) -> np.ndarray[np.complex64, float]:

        # self._prepare_geometry()
        # path gain shape: [num_rx]
        channels, path_gains = self._run_sionna(eval_mode=eval_mode)
        path_gain_dBs = utils.linear2dB(path_gains)
        return channels, path_gain_dBs

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

        paths = sig_cmap.compute_paths()
        paths.normalize_delays = False
        cir = paths.cir()
        # a: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps], tf.complex
        a, tau = cir
        # [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, self.l_max - self.l_min + 1], tf.complex
        channels: tf.Tensor = cir_to_time_channel(self.bandwidth, a, tau, self.l_min, self.l_max)

        coverage_map = sig_cmap.compute_cmap()
        # path_gains = sig_cmap.get_path_gain(coverage_map)
        path_gains = []
        for pos in coverage_map.rx_pos:
            path_gain = coverage_map.path_gain[:, pos[1], pos[0]]
            path_gains.append(path_gain[0])
        path_gains = np.asarray(path_gains)

        if eval_mode:
            # Path for outputing iamges if we want to visualize the coverage map
            img_dir = os.path.join(
                assets_dir, "images", self.log_string + self.current_time + f"_{self.idx}"
            )
            render_filename = utils.create_filename(img_dir, f"{scene_name}_00000.png")
            sig_cmap.render_to_file(coverage_map, filename=render_filename)

        channels = tf.squeeze(channels, axis=(0, 2, 3, 5))
        channels = np.asarray(channels, dtype=np.complex64)
        sig_cmap.free_memory()

        return channels, path_gains


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
