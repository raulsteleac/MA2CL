from setuptools import setup, find_packages

setup(
    name="MA2CL",
    url="https://github.com/song-hl/MA2CL",
    packages=find_packages(),
    install_requires=[
        "hydra-core",
        "kornia",
        "pysc2",
        "mujoco",
        "mujoco-py",
        "scikit-image",
        "scikit-learn",
        "scikit-video",
        "tensorboard",
        "tensorboardX",
        "pandas",
        "seaborn",
        "matplotlib",
        "opencv-python==4.5.5.64"
    ],
)
