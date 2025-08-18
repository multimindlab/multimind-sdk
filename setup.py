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

# Define long_description by reading the README.md file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="multimind-sdk",
    version="0.2.2",
    author="AI2Innovate Team",
    author_email="contact@multimind.dev",
    description="The Future of AI Development - 60+ Vector Databases • 100+ AI Models • Quantum Memory • Hybrid RAG • Enterprise Compliance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/multimind-dev/multimind-sdk",
    project_urls={
        "Bug Tracker": "https://github.com/multimind-dev/multimind-sdk/issues",
        "Website": "https://multimind.dev",
        "Source Code": "https://github.com/multimind-dev/multimind-sdk",
        "Discord": "https://discord.gg/K64U65je7h",
        "OpenCollective": "https://opencollective.com/multimind-sdk",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Database :: Database Engines/Servers",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Topic :: System :: Distributed Computing",
        "Topic :: System :: Systems Administration",
        "Topic :: Utilities",
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
            "pre-commit>=3.0.0",
        ],
        "compliance": [
            "cryptography>=41.0.0",
            "pyjwt>=2.8.0",
            "bcrypt>=4.0.0",
        ],
        "gateway": gateway_requirements,
        "full": sdk_requirements + gateway_requirements,
        "all": sdk_requirements + gateway_requirements + [
            "cryptography>=41.0.0",
            "pyjwt>=2.8.0",
            "bcrypt>=4.0.0",
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
            "pre-commit>=3.0.0",
        ],
    },
    entry_points={
        'console_scripts': [
            'multimind=multimind.gateway.cli:main',
        ],
    },
    keywords=[
        "ai", "artificial-intelligence", "llm", "machine-learning", 
        "rag", "vector-database", "agents", "fine-tuning", "quantum-memory",
        "hybrid-rag", "enterprise-ai", "compliance", "multi-modal",
        "federated-learning", "self-evolving-agents", "mcp", "workflow-automation"
    ],
    include_package_data=True,
    zip_safe=False,
)