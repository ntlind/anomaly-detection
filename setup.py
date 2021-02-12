from setuptools import setup, find_packages

setup(
    name="anomaly-detection",
    maintainer="Nick Lind",
    version="1.0",
    packages=find_packages(include=["anomaly_dection"]),
    maintainer_email="nick@quantilegroup.com",
    description="Detect anomalies in hierarchical time-series data",
    platforms="any",
    python_requires=">=3.7.1",
)
