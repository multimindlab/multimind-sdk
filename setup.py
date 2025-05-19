"""
Setup configuration for the Multimind SDK.
"""

from setuptools import setup, find_packages

# Read requirements from files
def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Base requirements
base_requirements = read_requirements('requirements-base.txt')

# Gateway requirements (excluding base)
gateway_requirements = [
    req for req in read_requirements('multimind/gateway/requirements.txt')
    if not req.startswith('-r')
]

# SDK requirements (excluding base)
sdk_requirements = [
    req for req in read_requirements('requirements.txt')
    if not req.startswith('-r')
]

setup(
    name="multimind-sdk",
    version="0.1.0",
    author="AI2Innovate Team",
    author_email="contact@multimind.dev",
    description="A unified interface for multiple LLM providers and local models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/multimindlabs/multimind-sdk",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
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
    install_requires=base_requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
        ],
        "gateway": gateway_requirements,
        "full": sdk_requirements + gateway_requirements,
    },
    entry_points={
        'console_scripts': [
            'multimind=multimind.gateway.cli:main',
        ],
    },
) 