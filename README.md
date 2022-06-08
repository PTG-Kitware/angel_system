# PTG ANGEL System

Initial repo for PTG project.

This repo contains:
* Experimental Unity app (and companion research-mode plugin library) for
  transmitting appropriate sensor data off of the HoloLens2 platform
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
HoloLens 2 headset
Anaconda/Miniconda - py3.9

## First time build/deploy instructions

1) Open the unity/ARUI project with Unity Hub.
2) Open the demo scene in the Unity editor (unity\ARUI\Assets\my_scene).
3) Modify the default project build settings (``File -> Build Settings...``).  
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
10) Modify the ``Package.appxmanifest`` per the README in HoloLens2-ResearchMode-Unity/  
11) Deploy the app to the Hololens by clicking ``Build -> Deploy Solution``
12) After deployment completes, open the Windows menu in the Hololens and select All Apps, and then click on the Angel_ARUI application.

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
- Assuming that only one ethernet connection is active at the time of running the experimental app.


# ROS 2 System
ROS 2 Foxy is utilized to support our system of streaming analytics and
report-back to the HoloLens2 platform.

Workspace root: `./ros/`

System requirements
* docker
* docker-compose

## Docker-based Workflow
**Intention**: Use containerization to standardize development and runtime
environment and practice.
* Docker is utilized for containerization.
* Docker-compose is used to manage container build and run  
  configurations.

Docker functionality is located under the `./docker/` directory.
* `ros2-base` provides a base environment that supports building and running
  our workspace.
* `ros2-workspace-build` provides a build of our workspace.

### Building Docker Images
Run `./angel-docker-build.sh`.

### Developing within Docker
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
`/angel_workspace/src` directory in the run container.

This "workspace" context additionally mounts build, install and log output
directories to a spot on the host in order to:
* persist between shell start-ups and shutdowns
* facilitate build sharing between multiple workspace shells

Due to this, there will not be a build available upon first shell start-up.
The script `/angel_workspace/workspace_build.sh` is available to run.
This script is used during the image build process, so using this script
ensures that the same build method is performed.

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

As a rosetta stone, this example configuration is symmetric to the launch file
version located at `ros/angel_debug/launch/online_debug_demo.py`.

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

## ANGEL System Python Package

`angel_system/` contains the interfaces and implementations for the
various components in the ANGEL system python package.

## Lessons Learned
### `rosdep`
References to the lists that rosdep uses to resolve names:
/etc/ros/rosdep/sources.list.d/20-default.list
