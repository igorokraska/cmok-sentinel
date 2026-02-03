from setuptools import setup, find_packages

setup(
    name="sentinel_tool",
    version="0.1.0",
    description="A tool for downloading Sentinel-2 imagery and computing spectral indices (NDVI, NDMI, NDRE, NDWI)",
    url="https://github.com/igorokraska/cmok-sentinel",  # Optional: GitHub repo
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.1",
        "rasterio>=1.3",
        "geopandas>=1.1",
        "pyproj>=3.6",
        "shapely>=2.0",
        "matplotlib>=3.7",
        "requests>=2.31",
        "urllib3>=2.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "sentinel-tool=sentinel_tool.cli:main",
        ],
    },
)
