from gymnasium.envs.registration import register


def register_envs():
    register(
        id="shared-ap-v0",
        entry_point="marlis.drl.envs.shared_ap:SharedAPV0",
        max_episode_steps=100,
    )
    register(
        id="data-center-v0",
        entry_point="marlis.drl.envs.data_center:DataCenterV0",
        max_episode_steps=100,
    )
    register(
        id="data-center-v1",
        entry_point="marlis.drl.envs.data_center:DataCenterV1",
        max_episode_steps=100,
    )
