from setuptools import setup
import os
from glob import glob

package_name = "regrasp_experiment"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Lukman Awaludin",
    maintainer_email="lukman.awaludin@ugm.ac.id",
    description="Uncertainty-guided closed-loop re-grasp experiment (RGB-D D455, ROS2).",
    license="MIT",
    entry_points={
        "console_scripts": [
            "experiment = regrasp_experiment.experiment_node:main",
            "spearhead_servo = regrasp_experiment.spearhead_servo:main",
            "spearhead_dashboard = regrasp_experiment.spearhead_dashboard:main",
        ],
    },
)
