"""Nyalakan RealSense D455 (depth teraligned ke color) + node eksperimen."""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    rs_launch = os.path.join(
        get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py")

    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rs_launch),
        launch_arguments={
            # ---- Konfigurasi D455 ----
            "align_depth.enable": "true",             # depth teralign ke color (WAJIB)
            "enable_color": "true",
            "enable_depth": "true",
            "rgb_camera.color_profile": "848x480x30", # seimbang FOV/akurasi utk D455
            "depth_module.depth_profile": "848x480x30",
            "pointcloud.enable": "false",
            "enable_sync": "true",
        }.items(),
    )

    experiment = Node(
        package="regrasp_experiment",
        executable="experiment",
        name="regrasp_experiment",
        output="screen",
        emulate_tty=True,
        parameters=[os.path.join(
            get_package_share_directory("regrasp_experiment"), "config", "params.yaml")],
    )

    return LaunchDescription([realsense, experiment])
