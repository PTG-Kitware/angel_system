using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.SpatialAwareness;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Windows.WebCam;
using DilmerGames.Core.Singletons;
using TMPro;
using System.Runtime.InteropServices;

#if ENABLE_WINMD_SUPPORT
using HL2UnityPlugin;
#endif


public class CaptureSensorData : MonoBehaviour
{

#if ENABLE_WINMD_SUPPORT
    HL2ResearchMode researchMode;
    enum DepthSensorMode
    {
        ShortThrow,
        LongThrow,
        None
    };
    DepthSensorMode depthSensorMode = DepthSensorMode.ShortThrow;
    bool enablePointCloud = true;

    Windows.Perception.Spatial.SpatialCoordinateSystem unityWorldOrigin;
#endif

    // Network stuff
    System.Net.Sockets.TcpClient tcpClient;
    NetworkStream tcpStream;

    private Logger _logger = null;

    // Spatial awareness stuff
    IEnumerable<SpatialAwarenessMeshObject> meshes;
    IMixedRealitySpatialAwarenessMeshObserver observer = null;

    long prev_ts;
    // IP Address hosting the server to connect to.
    // USB-C :: Worked after opening up incoming port through the firewall.
    string ip_address = "169.254.70.247";
    int ip_port = 11000;

    /// <summary>
    /// Lazy acquire the logger object and return the reference to it.
    /// </summary>
    /// <returns>Logger instance reference.</returns>
    private ref Logger logger()
    {
        if( this._logger == null )
        {
            // TODO: Error handling for null loggerObject?
            this._logger = GameObject.Find("Logger").GetComponent<Logger>();
        }
        return ref this._logger;
    }

    private void Awake()
    {
#if ENABLE_WINMD_SUPPORT
        unityWorldOrigin = Windows.Perception.Spatial.SpatialLocator.GetDefault().CreateStationaryFrameOfReferenceAtCurrentLocation().CoordinateSystem;
#endif
    }

    // Start is called before the first frame update
    void Start()
    {
        Logger log = logger();

#if ENABLE_WINMD_SUPPORT
        // Configure research mode
        log.LogInfo("Trying to enable research mode...");
        researchMode = new HL2ResearchMode();
        log.LogInfo("Research mode enabled");

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
        log.LogInfo("Research mode initialized");
#endif

        // Connect to the python TCP server
        this.tcpClient = new System.Net.Sockets.TcpClient();
        try
        {
            log.LogInfo("Attempting to connect to TCP socket @ IP address " + 
                        ip_address + ":" + ip_port);
            this.tcpClient.Connect(ip_address, ip_port);
            log.LogInfo("TCP client connected!");
            this.tcpStream = this.tcpClient.GetStream();
        }
        catch (Exception e)
        {
            log.LogInfo(e.ToString());
        }
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
                    //observer.LevelOfDetail = SpatialAwarenessMeshLevelOfDetail.Unlimited;
                    //observer.UpdateInterval = 0.5f;
                    this.logger().LogInfo("Detail level: " + observer.LevelOfDetail.ToString());
                    this.logger().LogInfo("Update interval: " + observer.UpdateInterval.ToString());
                }
            }
        }
    }

    void LateUpdate()
    {
#if ENABLE_WINMD_SUPPORT
        while (true) 
        {
            // Try to get the frame from research mode
            if (researchMode.LFImageUpdated())
            {
                //this.loggerObject.GetComponent<Logger>().LogInfo("LF image available");

                long ts;
                byte[] frameTexture = researchMode.GetLFCameraBuffer(out ts);

                //this.loggerObject.GetComponent<Logger>().LogInfo("time diff: " + (ts - prev_ts).ToString());
                prev_ts = ts;
               
                //this.loggerObject.GetComponent<Logger>().LogInfo(researchMode.PrintDebugString());
                //this.loggerObject.GetComponent<Logger>().LogInfo(frameTexture[0].ToString());
                //this.loggerObject.GetComponent<Logger>().LogInfo(researchMode.PrintLFResolution());
                //this.loggerObject.GetComponent<Logger>().LogInfo(researchMode.PrintLFFrameBuffer());
                //this.loggerObject.GetComponent<Logger>().LogInfo(researchMode.PrintLFExtrinsics());
                //this.loggerObject.GetComponent<Logger>().LogInfo("TS: " + ts.ToString());
                //this.loggerObject.GetComponent<Logger>().LogInfo("LF image: " + researchMode.m_lastSpatialFrame.LFFrame.image.Length.ToString());

                if (frameTexture.Length > 0)
                {
                    //this.loggerObject.GetComponent<Logger>().LogInfo("got something: " + frameTexture.Length.ToString());
                
                    // Prepend width and length
                    uint width = 640;
                    uint height = 480;
                    List<byte> screenshotBytes = new List<byte>();

                    screenshotBytes.Add((byte)((width & 0xFF000000) >> 24));
                    screenshotBytes.Add((byte)((width & 0x00FF0000) >> 16));
                    screenshotBytes.Add((byte)((width & 0x0000FF00) >> 8));
                    screenshotBytes.Add((byte)((width & 0x000000FF) >> 0));
                    screenshotBytes.Add((byte)((height & 0xFF000000) >> 24));
                    screenshotBytes.Add((byte)((height & 0x00FF0000) >> 16));
                    screenshotBytes.Add((byte)((height & 0x0000FF00) >> 8));
                    screenshotBytes.Add((byte)((height & 0x000000FF) >> 0));

                    for (int i = 0; i < frameTexture.Length; i++)
                    {
                        screenshotBytes.Add(frameTexture[i]);
                    }

                    byte[] screenshotBytesArray = AddMessageHeader(screenshotBytes.ToArray());

                    // Send the data through the socket.  
                    this.tcpStream.Write(screenshotBytesArray, 0, screenshotBytesArray.Length);
                    this.tcpStream.Flush();
                }
            
            }
            else
            {
                //this.loggerObject.GetComponent<Logger>().LogInfo("No frame available");
                //this.loggerObject.GetComponent<Logger>().LogInfo(researchMode.PrintDebugString());
                break;
            }
        }
#endif

    }

    /// <summary>
    /// Add a sync marker of 0x1ACFFC1D and a 4 byte length
    /// to the given message
    /// </summary>
    /// <param name="message"></param>
    /// <returns></returns>
    private static byte[] AddMessageHeader(byte[] message)
    {
        //Debug.Log(String.Format("Adding sync and length marker. Message length = {0}", message.Length));
        byte[] sync = { 0x1A, 0xCF, 0xFC, 0x1D };
        byte[] length = {(byte)((message.Length & 0xFF000000) >> 24),
                         (byte)((message.Length & 0x00FF0000) >> 16),
                         (byte)((message.Length & 0x0000FF00) >> 8),
                         (byte)((message.Length & 0x000000FF) >> 0)};
        byte[] newMessage = new byte[message.Length + 8]; // 4 byte sync + 4 byte length

        System.Buffer.BlockCopy(sync, 0, newMessage, 0, sync.Length);
        System.Buffer.BlockCopy(length, 0, newMessage, sync.Length, length.Length);
        System.Buffer.BlockCopy(message, 0, newMessage, sync.Length + length.Length, message.Length);

        return newMessage;
    }

}