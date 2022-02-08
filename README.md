# PTG

Inital repo for PTG project. Contains the Unity project, the HoloLens 2 research mode visual studio project, and various python scripts to communicate with the HoloLens.

# Tool versions used

Unity Hub - 3.0.0 - https://unity.com/unity-hub  
Unity - 2020.3.25f1 - https://unity3d.com/get-unity/download/archive  
Visual Studio 2019 - 16.11.8 - https://visualstudio.microsoft.com/vs/older-downloads/  
Mixed Reality Feature Tool - 1.02111-Preview - https://www.microsoft.com/en-us/download/details.aspx?id=102778  
HoloLens 2 headset
Anaconda/Miniconda - py3.9


# First time build/deploy instructions

1) Open the unity/Hello_World project with Unity Hub.
2) Open the demo scene in the Unity editor (unity\Hello_World\Assets\my_scene).
3) Modify the default project build settings (``File -> Build Settings...``).  
&nbsp;&nbsp;- Under the Universal Windows Platform tab  
&nbsp;&nbsp;&nbsp;&nbsp;- Target Device = HoloLens  
&nbsp;&nbsp;&nbsp;&nbsp;- Architecture = ARM64  
&nbsp;&nbsp;&nbsp;&nbsp;- Minimum Platform Version = 10.0.19041.0  
&nbsp;&nbsp;&nbsp;&nbsp;- Build and Run on = USB Device  
&nbsp;&nbsp;- Click ``Switch Platform`` after applying the new settings.  
4) Create a build folder to place completed Unity builds (e.g. unity/Hello_World/Builds/my_first_build).  
5) Click ``Build`` and specify your desired build folder. After the build completes, a new file explorer windows will pop up with the location of the build.  
6) Open the .sln file with Visual Studio.  
7) In Visual Studio, switch ``Solution Configurations`` to Release and ``Solution Platforms`` to ARM64.  
8) Open the project properties window (``Project -> Hello_World Properties``) and switch to the ``Configurations -> Debugging`` tab. Enter the IP address of your HoloLens in the Machine Name field.  
9) Modify the ``Package.appxmanifest`` per the README in HoloLens2-ResearchMode-Unity/  
10) Deploy the app to the Hololens by clicking ``Build -> Deploy Solution``
11) After deployment completes, open the Windows menu in the Hololens and select All Apps, and then click on the Hello_World application.

# Misc. notes

- Research mode must be enabled in the HoloLens headset (see "Enabling Research Mode" section here https://docs.microsoft.com/en-us/windows/mixed-reality/develop/advanced-concepts/research-mode).  
- The first time you start your application on the HoloLens, you may have to allow camera/microphone access to the app. Click yes on the popups and then restart the application.
