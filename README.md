# PTG ANGEL System

Initial repo for PTG project.

This repo contains:
* Experimental Unity app (and companion research-mode plugin library) for
  transmitting appropriate sensor data off of the HoloLens2 platform
* ROS 2 system for receiving sensor data, processing analytics, and pushing 
  results out. 


# Windows Development for HL2

## Tool versions used

Unity Hub - 3.0.0 - https://unity.com/unity-hub  
Unity - 2020.3.25f1 - https://unity3d.com/get-unity/download/archive  
Visual Studio 2019 - 16.11.8 - https://visualstudio.microsoft.com/vs/older-downloads/  
Mixed Reality Feature Tool - 1.02111-Preview - https://www.microsoft.com/en-us/download/details.aspx?id=102778  
HoloLens 2 headset
Anaconda/Miniconda - py3.9

## First time build/deploy instructions

1) Open the unity/Hello_World project with Unity Hub.
2) Open the demo scene in the Unity editor (unity\Hello_World\Assets\my_scene).
3) Modify the default project build settings (``File -> Build Settings...``).  
   - Under the Universal Windows Platform tab
     - Target Device = HoloLens  
     - Architecture = ARM64  
     - Minimum Platform Version = 10.0.19041.0  
     - Build and Run on = USB Device  
   - Click ``Switch Platform`` after applying the new settings.  
4) Create a build folder to place completed Unity builds (e.g. unity/Hello_World/Builds/my_first_build).  
5) Click ``Build`` and specify your desired build folder. After the build completes, a new file explorer windows will pop up with the location of the build.  
6) Open the .sln file with Visual Studio.  
7) In Visual Studio, switch ``Solution Configurations`` to Release and ``Solution Platforms`` to ARM64.  
8) Open the project properties window (``Project -> Hello_World Properties``) and switch to the ``Configurations -> Debugging`` tab. Enter the IP address of your HoloLens in the Machine Name field.  
9) Modify the ``Package.appxmanifest`` per the README in HoloLens2-ResearchMode-Unity/  
10) Deploy the app to the Hololens by clicking ``Build -> Deploy Solution``
11) After deployment completes, open the Windows menu in the Hololens and select All Apps, and then click on the Hello_World application.

## Misc. notes

- Research mode must be enabled in the HoloLens headset (see "Enabling Research Mode" section here https://docs.microsoft.com/en-us/windows/mixed-reality/develop/advanced-concepts/research-mode).  
- The first time you start your application on the HoloLens, you may have to allow camera/microphone access to the app. Click yes on the popups and then restart the application.


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

## Lessons Learned
### `rosdep`
References to the lists that rosdep uses to resolve names:
/etc/ros/rosdep/sources.list.d/20-default.list
