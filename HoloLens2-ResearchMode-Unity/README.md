# C++/WinRT HoloLens2ResearchMode Project Overview

This folder was copied from the microsoft/psi project, found [here](https://github.com/microsoft/psi/tree/master/Sources/MixedReality/HoloLens2ResearchMode).

This project wraps [HoloLens 2 Research Mode APIs](https://github.com/microsoft/HoloLens2ForCV/blob/main/Docs/ResearchMode-ApiDoc.pdf) in Windows Runtime 
classes. The generated Windows Runtime component may then be consumed by a C# Universal Windows Platform (UWP) app.

Note that this project includes the Research Mode header file [ResearchModeApi.h](./ResearchModeApi.h), which was copied directly from [the HoloLens 2 Research Mode samples repository](https://github.com/microsoft/HoloLens2ForCV). The original file is available [here](https://github.com/microsoft/HoloLens2ForCV/blob/5b0fa70a6e67997b6efe8a2ea1d41e06264aec3c/Samples/ResearchModeApi/ResearchModeApi.h).

# Using in Unity
To use it in Unity,
- Build this project (ARM64,Release) and copy the .dll and .winmd files in `HL2UnityPlugin\ARM64\Release\HL2UnityPlugin` into `Assets/Plugins/WSA/ARM64` folder of your Unity project.
- Change the architecture in your Unity build settings to be ARM64.
- After building the visual studio solution from Unity, go to `App/[Project name]/Package.appxmanifest` and add the restricted capability to the manifest file. (Same as what you would do to enable research mode on HoloLens 1, reference: http://akihiro-document.azurewebsites.net/post/hololens_researchmode2/)
```xml
<Package
  xmlns:mp="http://schemas.microsoft.com/appx/2014/phone/manifest"
  xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"
  xmlns:uap2="http://schemas.microsoft.com/appx/manifest/uap/windows10/2"
  xmlns:uap3="http://schemas.microsoft.com/appx/manifest/uap/windows10/3"
  xmlns:uap4="http://schemas.microsoft.com/appx/manifest/uap/windows10/4"
  xmlns:iot="http://schemas.microsoft.com/appx/manifest/iot/windows10"
  xmlns:mobile="http://schemas.microsoft.com/appx/manifest/mobile/windows10"
  xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities"
  IgnorableNamespaces="uap uap2 uap3 uap4 mp mobile iot rescap"
  xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10">
```

```xml
  <Capabilities>
    <rescap:Capability Name="perceptionSensorsExperimental" />
    <Capability Name="internetClient" />
    <Capability Name="internetClientServer" />
    <Capability Name="privateNetworkClientServer" />
    <uap2:Capability Name="spatialPerception" />
    <DeviceCapability Name="backgroundSpatialPerception"/>
    <DeviceCapability Name="webcam" />
  </Capabilities>
```
`<DeviceCapability Name="backgroundSpatialPerception"/>` is only necessary if you use IMU sensor.
- Save the changes and deploy the solution to your HoloLens 2.

## References
[Dorin Ungureanu, Federica Bogo, Silvano Galliani, Pooja Sama, Xin Duan, Casey Meekhof, Jan Stühmer, Thomas J. Cashman, Bugra Tekin, Johannes L. Schönberger, Pawel Olszta, and Marc Pollefeys. HoloLens 2 Research Mode as a Tool for Computer Vision Research. arXiv:2008.11239, 2020.](https://arxiv.org/abs/2008.11239)

[Research mode repository](https://github.com/microsoft/HoloLens2ForCV)

[Research mode API documentation](https://github.com/microsoft/HoloLens2ForCV/blob/main/Docs/ResearchMode-ApiDoc.pdf)

