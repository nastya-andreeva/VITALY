# setup.py
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="air-quality-analyzer",
    version="1.0.0",
    description="Система анализа качества воздуха с GUI интерфейсом",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    author="Air Quality Team",
    author_email="example@email.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)