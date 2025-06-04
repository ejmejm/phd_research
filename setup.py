from setuptools import setup, find_packages

setup(
    name="phd",
    version="0.3.0",
    packages=find_packages(),
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
    entry_points={
        'console_scripts': [
            'comet_sweep=phd.research_utils.scripts.comet_sweep:main',
            'comet_upload=phd.research_utils.scripts.comet_upload:main',
        ],
    },
)
