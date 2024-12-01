from setuptools import find_packages, setup

setup(
    name="SubGridify",  # Package name
    version="0.1.0",  # Initial version
    author="K. Roberts, J.L. Woodruff, [Other Author Name]",  # List all authors
    author_email="krober@example.com, jlwoodr3@example.com, other.author@example.com",  # Replace with actual emails
    description="Preprocessor calculator for subgrid scale terms.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/SubGridify",  # Replace with your repo URL
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Adjust license if needed
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "netCDF4",
        "cmocean",
        "gdal", 
    ],
    entry_points={
        "console_scripts": [
            "subgridify=subgridify.main:main",  # Adjust based on your entry point
        ],
    },
)
