from setuptools import setup, find_packages

setup(
    name="joint-force-calculator",
    version="0.1.0",
    description="Open-source Python tools for biomechanics education",
    author="Abigail Wu",
    author_email="abigail.n.wu@gmail.com",
    url="https://github.com/abigail172/joint-force-calculator",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
