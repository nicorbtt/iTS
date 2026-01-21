from setuptools import setup, find_packages

setup(
    name="tweediegp",
    version="0.0.0",
    description="Intermittent demand GP utilities",
    packages=find_packages(),
    install_requires=["numpy"],
    python_requires=">=3.8",
)
