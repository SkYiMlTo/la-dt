"""
Setup configuration for LA-DT (Look-Ahead Digital Twin).

Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="ladt-digital-twin",
    version="1.0.0",
    author="Hugo Bourreau",
    author_email="hugo.bourreau@cybercni.fr",
    description="Look-Ahead Digital Twin: Proactive Byzantine Attack Attribution in IoT-CPS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SkYiMlTo/la-dt",
    project_urls={
        "Documentation": "https://github.com/SkYiMlTo/la-dt#readme",
        "Bug Tracker": "https://github.com/SkYiMlTo/la-dt/issues",
        "Appendix": "https://github.com/SkYiMlTo/la-dt/tree/main/appendix",
    },
    license="MIT",
    packages=find_packages(where=".", include=["src*"]),
    package_dir={"": "."},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "Flask>=2.0.0",
        "py7zr>=0.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.9b0",
            "flake8>=3.9.0",
            "jupyter>=1.0.0",
            "ipython>=7.0.0",
        ],
        "gpu": [
            "torch>=2.0.0[cuda]",
        ],
        "viz": [
            "plotly>=5.0.0",
            "bokeh>=2.4.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "digital-twin",
        "byzantine-attacks",
        "anomaly-detection",
        "IoT",
        "cyber-physical-systems",
        "graph-neural-networks",
        "security",
    ],
    entry_points={
        "console_scripts": [
            "ladt-validate=src.training.phase_5_real_data_validation:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
