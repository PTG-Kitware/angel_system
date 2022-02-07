using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.SpatialAwareness;
using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using System.Runtime.InteropServices;

#if ENABLE_WINMD_SUPPORT
using Windows.Graphics.Imaging;
using Windows.Media;
using Windows.Media.Capture;
using Windows.Media.Capture.Frames;
using Windows.Media.MediaProperties;
using HL2UnityPlugin;
using System.Runtime.InteropServices.WindowsRuntime;
#endif

public class ResearchModeCapture : MonoBehaviour
{
#if ENABLE_WINMD_SUPPORT
    HL2ResearchMode researchMode;
    enum DepthSensorMode
    {
        ShortThrow,
        LongThrow,
        None
    };
    DepthSensorMode depthSensorMode = DepthSensorMode.LongThrow;
    bool enablePointCloud = false;

    Windows.Perception.Spatial.SpatialCoordinateSystem unityWorldOrigin;
#endif

    GameObject loggerObject = null;

    // Spatial awareness stuff
    IEnumerable<SpatialAwarenessMeshObject> meshes;
    IMixedRealitySpatialAwarenessMeshObserver observer = null;

    string debugString = "";

    private void Awake()
    {
#if ENABLE_WINMD_SUPPORT
        unityWorldOrigin = Windows.Perception.Spatial.SpatialLocator.GetDefault().CreateStationaryFrameOfReferenceAtCurrentLocation().CoordinateSystem;
#endif
    }

    // Start is called before the first frame update
    void Start()
    {
        this.loggerObject = GameObject.Find("Logger");

#if ENABLE_WINMD_SUPPORT
        // Configure research mode
        this.loggerObject.GetComponent<Logger>().LogInfo("Research mode enabled");
        researchMode = new HL2ResearchMode();

        // Depth sensor should be initialized in only one mode
        if (depthSensorMode == DepthSensorMode.LongThrow) researchMode.InitializeLongDepthSensor();
        else if (depthSensorMode == DepthSensorMode.ShortThrow) researchMode.InitializeDepthSensor();
        
        researchMode.InitializeSpatialCamerasFront();
        researchMode.SetReferenceCoordinateSystem(unityWorldOrigin);
        researchMode.SetPointCloudDepthOffset(0);

        // Depth sensor should be initialized in only one mode
        if (depthSensorMode == DepthSensorMode.LongThrow) researchMode.StartLongDepthSensorLoop(enablePointCloud);
        else if (depthSensorMode == DepthSensorMode.ShortThrow) researchMode.StartDepthSensorLoop(enablePointCloud);

        researchMode.StartSpatialCamerasFrontLoop();
        this.loggerObject.GetComponent<Logger>().LogInfo("Research mode initialized");
#endif
    }

    void Update()
    {
        // Setup the spatial awareness observer
        if (observer == null)
        {
            var meshObservers = (CoreServices.SpatialAwarenessSystem as IMixedRealityDataProviderAccess).GetDataProviders<IMixedRealitySpatialAwarenessMeshObserver>();
            foreach (var observers in meshObservers)
            {
                if (observers.Meshes.Count != 0)
                {
                    observer = observers;
                    observer.DisplayOption = SpatialAwarenessMeshDisplayOptions.None;
                    this.loggerObject.GetComponent<Logger>().LogInfo("Detail level: " + observer.LevelOfDetail.ToString());
                }
            }
        }

#if ENABLE_WINMD_SUPPORT
        //if (researchMode.PrintDebugString() != "")
        // {
        //    this.loggerObject.GetComponent<Logger>().LogInfo(researchMode.PrintDebugString());
        //}
#endif

        if (debugString != "")
        {
            //this.loggerObject.GetComponent<Logger>().LogInfo(debugString);
        }
    }

}