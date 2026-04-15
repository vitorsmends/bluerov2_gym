# setup.py
from setuptools import find_packages, setup

setup(
    name="bluerov2_gym",
    version="0.1.0",
    description="OpenAI Gym environment for the BlueROV2 underwater vehicle",
    author="Vitor Mendes",
    # find_packages() searches for all directories with an __init__.py file
    packages=find_packages(include=["bluerov2_gym", "bluerov2_gym.*"]),
    include_package_data=True,
    package_data={
        "bluerov2_gym": ["assets/*.dae", "assets/*"],
    },
    install_requires=[
        "gymnasium>=0.29.1",
        "meshcat>=0.3.2",
        # Critical: Capping numpy below 2.1.0 to avoid breaking Google Colab's 
        # internal tools (TensorFlow/Numba) while staying compatible with SB3.
        "numpy>=1.26.0,<2.1.0", 
        "scipy>=1.14.1",
        "stable-baselines3[extra]>=2.3.2",
        "transforms3d>=0.4.1",
        "pygame>=2.1.0",
    ],
    # This entry_point allows gym.make("BlueRov-v0") to work after pip install
    entry_points={
        "gymnasium.envs": [
            "BlueRov-v0 = bluerov2_gym.envs.bluerov_env:BlueRov",
        ],
    },
    python_requires=">=3.8",
    zip_safe=False, # Essential for accessing non-python assets like the .dae model
)