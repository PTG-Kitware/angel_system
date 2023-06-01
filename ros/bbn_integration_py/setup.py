from setuptools import setup, find_packages

package_name = "bbn_integration_py"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=(package_name + "*",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Paul Tunison",
    maintainer_email="paul.tunison@kitware.com",
    # ---
    # Shared with properties in package.xml.
    description="ROS2 nodes in python for BBN integration support",
    license="BSD 3-Clause",
    # ---
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "task_to_bbn_update = bbn_integration_py.nodes.task_to_bbn_update:main",
        ],
    },
)
