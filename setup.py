from setuptools import setup, find_packages

setup(
    name="phd_research",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    author="Edan Meyer",
    author_email="ejmejm98@gmail.com",
    description="Research for my PhD, mainly on the discovery problem",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ejmejm/phd_research",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
