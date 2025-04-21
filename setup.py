from setuptools import setup, find_packages

setup(
    name="encampment-detection-system",
    version="0.1.0",
    description="A Streamlit-based web application for detecting encampment-related objects in Google Street View images",
    author="BDA600 Project Team",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.31.0",
        "google-streetview>=1.2.9",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "pillow>=10.1.0",
        "pandas>=2.1.1",
        "matplotlib>=3.8.0",
        "folium>=0.14.0",
        "streamlit-folium>=0.14.0",
        "requests>=2.31.0",
        "numpy>=1.24.0",
        "shapely>=2.0.1",
        "geopandas>=0.13.2",
    ],
    python_requires=">=3.8",
) 