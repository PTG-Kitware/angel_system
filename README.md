# PTG ANGEL System

Initial repo for PTG project.

This repo contains:
* Experimental Unity app (and companion [HL2SS] plugin) for
  transmitting appropriate sensor data off of the HoloLens2 platform,
  and for running the ANGEL ARUI.
* ROS 2 system for receiving sensor data, processing analytics, and pushing 
  results out. 

Note: This repository contains submodules that will need to be initialized upon first checkout:
```bash
$ git submodule update --init --recursive
```

# Windows Development for HL2

## Tool versions used

Unity Hub - 3.0.0 - https://unity.com/unity-hub  
Unity - 2020.3.25f1 - https://unity3d.com/get-unity/download/archive  
Visual Studio 2019 - 16.11.8 - https://visualstudio.microsoft.com/vs/older-downloads/  
Mixed Reality Feature Tool - 1.02111-Preview - https://www.microsoft.com/en-us/download/details.aspx?id=102778  
HoloLens 2 headset - OS version 20438.1432  
Anaconda/Miniconda - py3.9  
HL2SS plugin - 1.0.15 - https://github.com/jdibenes/hl2ss

## First time build/deploy instructions

1) Open the unity/ARUI project with Unity Hub.
2) Open the arui_engineering scene in the Unity editor (unity\ARUI\Assets\my_scene).
3) Modify the default project build settings (``File -> Build Settings...``).
   - Ensure the "Scenes/arui_engineering" scene is checked for inclusion in the build
     (others may be unchecked).
   - Under the Universal Windows Platform tab
     - Target Device = HoloLens  
     - Architecture = ARM64  
     - Minimum Platform Version = 10.0.19041.0  
     - Build and Run on = USB Device  
   - Click ``Switch Platform`` after applying the new settings.  
4) Follow the instructions in the [ROS Unity setup section](#ros-unity-setup) to generate the message files and set the endpoint IP.
5) Create a build folder to place completed Unity builds (e.g. unity/ARUI/Build).
6) Click ``Build`` and specify your desired build folder. After the build completes, a new file explorer windows will pop up with the location of the build.
7) Open the .sln file with Visual Studio.  
8) In Visual Studio, switch ``Solution Configurations`` to Release and ``Solution Platforms`` to ARM64.  
9) Open the project properties window (In the ``Solution Explorer`` pane, right-click ``Angel_ARUI`` and select ``Properties``) and switch to the ``Configurations -> Debugging`` tab. Enter the IP address of your HoloLens in the Machine Name field.
10) Deploy the app to the Hololens by clicking ``Build -> Deploy Solution``
11) After deployment completes, open the Windows menu in the Hololens and select All Apps, and then click on the Angel_ARUI application.

## ROS Unity Setup
### ROS IP configuration
1) In Unity, click ``Robotics -> ROS Settings`` to open the ROS Settings menu.
2) Set the protocol to ``ROS2``.
3) Enter the IP address of the machine the TCP endpoint node will be running on (i.e. the machine the HoloLens 2 will be connecting to) and close the ROS Settings menu.
### ROS message C# script generation
1) In Unity, click ``Robotics -> Generate ROS Messages...`` to open the ROS Message Browser.
2) Set the ROS message path to the directory containing the ROS2 messages for the project (../../ros/angel_msgs for this project's ANGEL message folder). The Message Browser should display the .msg files it found in the ROS message path.
3) Click ``Build msgs`` and wait for message generation to finish. You should now see new files in the Built message path location (default location is unity/ARUI/Assets/RosMessages).
4) Close the ROS Message Browser.

## Running application without a development environment
See [Unity README.md](unity/README.md) for instructions on creating an application package and installing it via the HoloLens 2 device portal.

## Misc. notes

- Research mode must be enabled in the HoloLens headset (see "Enabling Research Mode" section here https://docs.microsoft.com/en-us/windows/mixed-reality/develop/advanced-concepts/research-mode).  
- The first time you start your application on the HoloLens, you may have to allow camera/microphone access to the app. Click yes on the popups and then restart the application.
- Due to issues accessing the depth and PV camera simultaneously, the HoloLens OS version should be less than 20348.1501. Version 20348.1432 has been tested with the app and confirmed to work. See this [issue](https://github.com/microsoft/HoloLens2ForCV/issues/133) for more information. To install an older OS version on the HoloLens, follow the instructions [here](https://docs.microsoft.com/en-us/hololens/hololens-recovery#clean-reflash-the-device).
- Although the short throw depth camera data provided by research mode is provided as UINT16 data, the max valid value is 4090 (values above 4090 signify an invalid reading from the camera). So, the true range of the short throw depth camera is [0, 4090].

# ROS 2 System
ROS 2 Foxy is utilized to support our system of streaming analytics and
report-back to the HoloLens2 platform.

Workspace root: `./ros/`

System requirements
* ansible
* docker
* (pip-installed) docker-compose

Some files required from the `https://data.kitware.com` Girder service require
authentication due to their protected nature.
The environment variable `GIRDER_API_KEY` must be defined with a valid API key,
otherwise an authentication token cannot be retrieved.

We currently require a pip-installed `docker-compose` tool, as opposed to a
system package-manager installed `docker-compose-plugin` package.
The package manager docker plugin behaves a little differently that our current
docker-compose configuration and scripting does not yet handle.

## Provision Files
External large files should be provisioned by running the ansible tool:

    ansible-playbook -i ansible/hosts.yml ansible/provision_files.yml

This may include large files for running the system, like ML model files, or
other files required for building docker images.

This provisioning may require additional configuration and variables set in
your environment in order to satisfy some permissions:
* `GIRDER_API_KEY` will need to be set in order to acquire protected files from
  `data.kitware.com`.

The configuration that controls what is staged and where is located
in the `ansible/roles/provision-files/vars/main.yml` file.

## Docker-based Workflow
**Intention**: Use containerization to standardize development and runtime
environment and practice.
* Docker is utilized for containerization.
* Docker-compose is used to manage container build and run  
  configurations.

Docker functionality is located under the `./docker/` directory.
* `workspace-base-dev` provides a base environment that supports building and
  running our workspace.
* `workspace-build` provides a build of our workspace.

### Building Docker Images
Quick-start
```bash
./angel-docker-build.sh
```

### Developing within Docker
Quick-start
```bash
# Start the containerized environment shell.
./angel-workspace-shell.sh
# Build your local workspace for use.
./workspace_build.sh
```

To develop adequately in docker, the container environment needs to have access
to the source code that is being edited, likely by an IDE on the host.
Running `./angel-workspace-shell.sh` will create a temporary docker container
and drop into a bash shell.
This script has some usage output that can be seen by passing a `-h`/`--help`
option.
By default, the `workspace-shell-dev-gpu` service will be run.
The definition of this service is found in the `docker/docker-compose.yml`
configuration.

This will mount the `./ros/` subtree on the host system into the
`/angel_workspace/src/` directory in the run container.

This "workspace" context additionally mounts build, install and log output
directories to a spot on the host in order to:
* persist between shell start-ups and shutdowns
* facilitate build sharing between multiple workspace shells

Due to this, there will not be a build available upon first shell start-up.
The script `/angel_workspace/workspace_build.sh` is available to run.
This script is used during the image build process, so using this script
ensures that the same build method is performed.

Other directories and files are mounted into the container environment for
development purposes and external file sharing.

This shell will **_NOT_** have the local installation sourced in order to
facilitate further safe build actions.
`ros2 run` actions should be performed in a separate shell from where builds
are performed.

**_NOTE_** that any new system requirements may of course be installed locally,
but these will be lost upon container shutdown.
* Additional core requirements should be reflected in the
  `./docker/workspace-build/` image definition.
* Changes to the workspace build process should be reflected in the
  `./docker/workspace-build/` image definition.
* Additional development only dependencies should be added to the
  `./docker/workspace-base-dev` image definition.

### Container Cyclone DDS configuration
A basic template config may be auto-generated to specify which host network
interface for it to use by uncommenting and setting the `CYCLONE_DDS_INTERFACE`
in the `docker/.env` or exporting it on your commandline to the desired value.

## Run Configurations -- Tmuxinator
The "standard" way to set up and run collections of nodes in ROS2 is via launch
files.
This is all well and good and provides great features, but what it doesn't do
is provide access to individual components at runtime.
During development, and often in the field as well, it is important to be able
to get at individual components to see what might be going wrong or to
restart/tweak one thing without impacting everything else that is running.
We utilize tmux, configured and run by tmuxinator, to manage multiple
windows/panes to host individual components.

Tmuxinator configurations are stored in the `./tmux/` directory.

### Example: Object detection system fragment
```bash
./angel-workspace-shell.sh -r -- tmuxinator start fragment_object_detection_debug
```
Anatomy of the call:
1) (Re)uses `angel-workspace-shell.sh` to run a command in a docker container.
2) `-r` option to script sets "use our built workspace" flag.
3) Things to the right of `--` are run inside the container.
4) `tmux` directory is (by default) in the same directory as the working
   directory the container starts in.
5) `tmuxinator` command creates a new server session defined by the given
   config.

To stop the system after using the above command, we simply need to exit the
tmux instance or kill the tmux session.
This can be done by providing the tmux keyboard command `<Ctrl-B, D>`.
Alternatively, in a new or existing pane, running `tmux kill-session` or
`tmux kill-server`.
When using the above example "start" command, exiting the tmux session will
also shut down the container.

If starting an `angel-workspace-shell.sh -r` first, dropping into a bash
terminal, and _**then**_ calling `tmuxinator start fragment_object_detection_debug`,
we have the option to exit the tmux session and stop it using another
tmuxinator command:
```bash
tmuxinator stop fragment_object_detection_debug
```

## Configuring ROS nodes that utilize SMQTK-Core
Some ROS nodes utilize the smqtk-core plugin system so specification of what
algorithm is utilized is determined based on a configuration file.

Configuration files are currently located in: `ros/angel_system_nodes/configs/`
* `default_object_det_config.json`
  * Determines the image object detection plugin used in the
    `angel_system_nodes object_detector.py` node.
* `default_activity_det_config.json`
  * Determines the activity detection plugin used in the
    `angel_system_nodes activity_detector.py` node.

## Setting up the foot pedal for annotations
The `annotation_event_monitor` ROS node uses the up and down arrow keys to
generate `AnnotationEvent` messages. These messages can be used to determine
when the beginning and end of an activity or error occur during a recording.

To help with this process, a foot pedal can be used to map foot pedal presses
to keyboard presses. For this system, we are using the [Infinity 3 USB foot pedal].

To configure your Linux system to recognize foot pedal presses as keyboard
presses, see this [guide].

In the .hwdb file, make sure to map the keyboard presses to the up and down arrow keys.

## ANGEL System Python Package

`angel_system/` contains the interfaces and implementations for the
various components in the ANGEL system python package.

### Running PTG evaluation
See `angel_system/eval/README.md` for details.

## Lessons Learned
### `rosdep`
References to the lists that rosdep uses to resolve names:
/etc/ros/rosdep/sources.list.d/20-default.list


[Infinity 3 USB foot pedal]: https://www.amazon.com/Infinity-Digital-Control-Computer-USB2/dp/B002MY6I7G?th=1
[guide]: https://catswhisker.xyz/log/2018/8/27/use_vecinfinity_usb_foot_pedal_as_a_keyboard_under_linux/
[hl2ss]: https://github.com/jdibenes/hl2ss
