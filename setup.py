from setuptools import setup, find_packages

setup(
    name="aede",
    version="0.1.0",
    description="Adaptive Experimental Decision Engine — decision-theoretic framework for self-driving labs",
    author="AEDE Contributors",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.2",
        "pandas>=1.5",
        "pyyaml>=6.0",
    ],
    extras_require={
        "ui": ["streamlit>=1.28", "plotly>=5.15"],
        "production": ["torch>=2.1", "gpytorch>=1.11", "botorch>=0.9.4"],
        "molecular": ["rdkit>=2023.3.2"],
        "all": ["streamlit>=1.28", "plotly>=5.15",
                "torch>=2.1", "gpytorch>=1.11", "botorch>=0.9.4"],
    },
    entry_points={
        "console_scripts": [
            "aede=aede.cli:main",
        ],
    },
)
