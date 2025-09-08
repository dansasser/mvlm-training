from setuptools import setup, find_packages

setup(
    name="prioritary-mvlm",
    version="0.1.0",
    description="Prioritary MVLM training utilities",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
    ],
)
