import math
from typing import Dict, List, Tuple, Sequence


def get_lead_follow_dict(
    num_rows: int,
    num_cols: int,
    follow_range: int,
):
    center_distance = follow_range * 2 + 1
    lead_follow_dict = {}
    # Find lead_idx
    for i in range(follow_range, num_cols, center_distance):
        for j in range(follow_range, num_rows, center_distance):
            lead_idx = i * num_rows + j

            # Find follow_idxs
            follow_idxs = []
            for n in range(i - follow_range, i + follow_range + 1):
                for m in range(j - follow_range, j + follow_range + 1):
                    follow_idx = n * num_rows + m
                    if follow_idx != lead_idx:
                        follow_idxs.append(follow_idx)

            lead_follow_dict.update({lead_idx: follow_idxs})
    return lead_follow_dict


def get_reflector_config():
    """
    Set up reflector configuration.
    """
    max_delta = math.radians(44.99)
    min_delta = math.radians(-44.99)

    # zenith angle
    init_theta = math.radians(90.0)
    theta_min = init_theta + min_delta
    theta_max = init_theta + max_delta
    theta_config = (init_theta, theta_min, theta_max)

    # azimuthal angle
    init_phi = math.radians(135.0)
    phi_min = init_phi + min_delta
    phi_max = init_phi + max_delta
    phi_config = (init_phi, phi_min, phi_max)

    num_groups = 9
    num_elements_per_group = 7

    return theta_config, phi_config, num_groups, num_elements_per_group


def get_config_shared_ap():
    """
    Set up reflector configuration for Shared AP senarios.
    """
    max_delta = math.radians(29.9)
    min_delta = math.radians(-29.9)

    # Rotation in x-y plane, horizontal rotation
    init_thetas = [math.radians(90.0), math.radians(90.0)]
    min_thetas = [init_thetas[i] + min_delta for i in range(len(init_thetas))]
    max_thetas = [init_thetas[i] + max_delta for i in range(len(init_thetas))]
    theta_config = (init_thetas, min_thetas, max_thetas)

    # Rotation in x-z plane, vertical rotation
    init_phis = [math.radians(30.0), math.radians(-30.0)]
    min_phis = [init_phis[i] + min_delta for i in range(len(init_phis))]
    max_phis = [init_phis[i] + max_delta for i in range(len(init_phis))]
    phi_config = (init_phis, min_phis, max_phis)

    num_groups = 9
    num_elements_per_group = 7

    return theta_config, phi_config, num_groups, num_elements_per_group


def get_config_data_center():
    """
    Set up reflector configuration for Data Center senarios.
    """
    max_delta = math.radians(29.9)
    min_delta = math.radians(-29.9)

    # Rotation in x-y plane, horizontal rotation
    init_thetas = [math.radians(90.0), math.radians(90.0)]
    min_thetas = [init_thetas[i] + min_delta for i in range(len(init_thetas))]
    max_thetas = [init_thetas[i] + max_delta for i in range(len(init_thetas))]
    theta_config = (init_thetas, min_thetas, max_thetas)

    # Rotation in x-z plane, vertical rotation
    init_phis = [math.radians(0.0), math.radians(0.0)]
    min_phis = [init_phis[i] + min_delta for i in range(len(init_phis))]
    max_phis = [init_phis[i] + max_delta for i in range(len(init_phis))]
    phi_config = (init_phis, min_phis, max_phis)

    num_groups = 9
    num_elements_per_group = 7

    return theta_config, phi_config, num_groups, num_elements_per_group


def constraint_angle(angle: float, angle_delta: Sequence[float]) -> float:
    """
    Constrain the angle within the angle_delta.

    Args:
    angle: float
        The angle to be constrained. Unit: degree.
    angle_delta: Sequence[float]
        The maximum and minimum angle deltas. Unit: degree.
    """
    min_angle = angle_delta[0]
    angle = min_angle if angle <= min_angle else angle
    max_angle = angle_delta[1]
    angle = max_angle if angle >= max_angle else angle
    return angle
