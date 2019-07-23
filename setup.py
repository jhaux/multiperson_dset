from setuptools import setup, find_packages

setup(
    name="multiperson_video_dataset",
    version="0.1",
    description="Given a set of sequences of frames annotated with person ids "
        "and keypoints, supplies functionality to use these on a frame basis. "
        "Needs edflow: https://github.com/pesser/edflow.",
    url="https://github.com/jhaux/multiperson",
    author="Johannes Haux",
    author_email="johannes.haux@iwr.uni-heidelberg.de",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "tqdm",
        "Pillow",
        "numpy",
    ],
    zip_safe=False,
)
