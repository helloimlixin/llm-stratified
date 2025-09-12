"""Setup script for the fiber bundle hypothesis test package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Fiber Bundle Hypothesis Test on LLM Embeddings"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as fh:
        requirements = [
            line.strip() for line in fh 
            if line.strip() and not line.startswith("#") and not line.startswith("pytest")
        ]
else:
    requirements = [
        "numpy>=1.21.0",
        "scipy>=1.7.0", 
        "torch>=1.9.0",
        "transformers>=4.21.0",
        "statsmodels>=0.13.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pyyaml>=6.0"
    ]

setup(
    name="fiber-bundle-test",
    version="1.0.0",
    author="Fiber Bundle Research Team",
    author_email="research@example.com",
    description="Fiber Bundle Hypothesis Test on LLM Embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/fiber-bundle-test",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fiber-bundle-test=run_analysis:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
