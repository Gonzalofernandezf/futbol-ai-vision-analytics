"""
Setup configuration for Football AI Vision Analytics
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="futbol-ai-vision-analytics",
    version="1.0.0",
    author="Football Analytics Team",
    description="Computer Vision pipeline for automated telemetry extraction from football broadcasts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/futbol-ai-vision-analytics",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Intended Audience :: Developers",
    ],
    python_requires=">=3.9",
    install_requires=[
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "supervision>=0.19.0",
        "Flask>=3.0.0",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "futbol-ai=Main:main",
        ],
    },
)
