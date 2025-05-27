"""项目安装配置."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="credit-card-default-prediction",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="信用卡违约预测项目",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/credit-card-default-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "black>=21.7b0",
            "flake8>=3.9.2",
            "isort>=5.9.3",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-model=src.models.train:main",
            "start-api=src.api.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml"],
    },
) 