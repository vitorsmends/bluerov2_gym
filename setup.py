# setup.py
from setuptools import find_packages, setup

setup(
    name="bluerov2_gym",
    version="0.1.0",
    description="OpenAI Gym environment for the BlueROV2 underwater vehicle",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "bluerov2_gym": ["assets/*"],
    },
    install_requires=[
        "gymnasium>=0.29.1",
        "meshcat>=0.3.2",
        # We cap numpy below 2.1 to ensure compatibility with Google Colab,
        # Numba, and older TensorFlow versions while keeping it modern enough for SB3.
        "numpy>=1.26.0,<2.1.0", 
        "scipy>=1.14.1",
        "stable-baselines3[extra]>=2.3.2",
        "transforms3d>=0.4.1",
    ],
    python_requires=">=3.8", # Standard for modern RL libraries
)