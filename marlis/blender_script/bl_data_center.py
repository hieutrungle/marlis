import bpy
from mathutils import Vector
import math
import os
import os, sys, inspect
import pickle

# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(
    os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])
)
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

import bl_utils, bl_parser, shared_utils


def export_drl_hallway_hex(args):

    # unit: radians
    if args.scene_name.lower() == "shared_ap":
        theta_config, phi_config, num_groups, num_elements_per_group = (
            shared_utils.get_config_shared_ap()
        )
    elif args.scene_name.lower() == "data_center":
        theta_config, phi_config, num_groups, num_elements_per_group = (
            shared_utils.get_config_data_center()
        )
    else:
        raise ValueError("Invalid scene name")

    theta_ranges = [(theta_config[1][i], theta_config[2][i]) for i in range(len(theta_config[1]))]
    phi_ranges = [(phi_config[1][i], phi_config[2][i]) for i in range(len(phi_config[1]))]

    with open(args.rt_delta_path, "rb") as f:
        rt_delta_pos = pickle.load(f)

    with open(args.focal_path, "rb") as f:
        # data: Tuple[np.ndarray[float], np.ndarray[float]]
        focals = pickle.load(f)
    focals = [focal.reshape(num_groups, 3) for focal in focals]

    # Angle container
    angles = [
        [0.0 for _ in range(num_groups * (num_elements_per_group + 1))] for _ in range(len(focals))
    ]

    # Set up devices
    # 'devices' contains multiple reflectors
    # each reflector has groups of tiles
    obstacles_names = []
    racks_names = []
    devices_names = []
    devices = {}
    top_panel_names = []

    ref_idx = 0
    for k, v in bpy.data.collections.items():
        ref_name = f"Reflector.{ref_idx:03d}"
        if ref_name in k:
            object_dict = {}
            devices_names.append(k)
            sorted_objects = sorted(v.objects, key=lambda x: x.name)
            for obj in sorted_objects:
                concat_name = obj.name.strip().split(".")
                group_name = concat_name[1]
                if group_name not in object_dict:
                    object_dict[group_name] = [obj]
                else:
                    object_dict[group_name].append(obj)
            devices[k] = object_dict
            ref_idx += 1

        if "RackFrame" in k:
            racks_names.append(k)

        if "Obstacles" in k:
            obstacles_names.append(k)

        if "TopPanel" in k:
            top_panel_names.append(k)

    for i, (reflector, focal) in enumerate(zip(devices_names, focals)):
        object_dict = devices[reflector]
        rt_delta = rt_delta_pos[i]
        for j, (group_name, objects) in enumerate(object_dict.items()):
            r_mid, theta_mid, phi_mid = focal[j]
            focal_vec = bl_utils.spherical2cartesian(r_mid, theta_mid, phi_mid)
            focal_pt = bl_utils.get_center_bbox(objects[num_elements_per_group // 2]) + Vector(
                focal_vec
            )
            theta_mid = shared_utils.constraint_angle(theta_mid, theta_ranges[i])
            angles[i][j * (num_elements_per_group + 1)] = phi_mid

            # loop through all elements in a group/column
            for k, obj in enumerate(objects):
                center = bl_utils.get_center_bbox(obj)
                r, theta, phi = bl_utils.compute_rot_angle(center, focal_pt)
                theta = shared_utils.constraint_angle(theta, theta_ranges[i])
                phi = shared_utils.constraint_angle(phi, phi_ranges[i])
                obj.rotation_euler = [0, theta, phi]
                obj.location = [p + d for p, d in zip(obj.location, rt_delta)]
                angles[i][j * (num_elements_per_group + 1) + k + 1] = theta

    result_path = args.focal_path
    with open(result_path, "wb") as f:
        pickle.dump(angles, f)

    result_path = args.focal_path
    with open(result_path, "wb") as f:
        pickle.dump(angles, f)

    # Save files without ceiling
    folder_dir = os.path.join(args.output_dir, f"idx")
    bl_utils.mkdir_with_replacement(folder_dir)
    bl_utils.save_mitsuba_xml(
        folder_dir, "scenee", [*devices_names, *racks_names, *obstacles_names, "Wall", "Floor"]
    )

    # Save files with ceiling
    folder_dir = os.path.join(args.output_dir, f"ceiling_idx")
    bl_utils.mkdir_with_replacement(folder_dir)
    bl_utils.save_mitsuba_xml(
        folder_dir,
        "scenee",
        [
            *devices_names,
            *racks_names,
            *obstacles_names,
            "Wall",
            "Floor",
            "Ceiling",
            *top_panel_names,
        ],
    )


def main():
    args = create_argparser().parse_args()
    export_drl_hallway_hex(args)


def create_argparser() -> bl_parser.ArgumentParserForBlender:
    """Parses command line arguments."""
    parser = bl_parser.ArgumentParserForBlender()
    parser.add_argument("--scene_name", "-s", type=str, required=True)
    parser.add_argument("--focal_path", type=str, required=True)
    parser.add_argument("--rt_delta_path", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--index", type=str)
    return parser


if __name__ == "__main__":
    main()
