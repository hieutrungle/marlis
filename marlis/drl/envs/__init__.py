from gymnasium.envs.registration import register


def register_envs():
    register(
        id="shared-ap-v0",
        entry_point="marlis.drl.envs.shared_ap:SharedAPV0",
        max_episode_steps=100,
    )
