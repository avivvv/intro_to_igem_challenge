from setuptools import setup, find_packages

setup(
    name="promoter_expression_predictor",
    version="0.1.0",
    description="ML project to predict gene expression levels from promoter sequences",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "biopython>=1.81",
        "tensorflow>=2.13.0",
    ],
)
