from gymnasium.envs.registration import register, registry

if "BlueRov-v0" not in registry:
    register(
        id="BlueRov-v0",
        entry_point="bluerov2_gym.envs:BlueRov",
        max_episode_steps=5000,
    )
