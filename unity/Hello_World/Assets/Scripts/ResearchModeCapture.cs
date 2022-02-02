using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.SpatialAwareness;
using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Windows.WebCam;
using DilmerGames.Core.Singletons;
using TMPro;
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
    DepthSensorMode depthSensorMode = DepthSensorMode.ShortThrow;
    bool enablePointCloud = true;

    Windows.Perception.Spatial.SpatialCoordinateSystem unityWorldOrigin;
#endif

    // Network stuff
    System.Net.Sockets.TcpClient tcpClient1;
    NetworkStream tcpStream1;
    System.Net.Sockets.TcpClient tcpClient2;
    NetworkStream tcpStream2;
    System.Net.Sockets.TcpClient tcpClient3;
    NetworkStream tcpStream3;
    System.Net.Sockets.TcpClient tcpClient4;
    NetworkStream tcpStream4;

    GameObject loggerObject = null;

    // Spatial awareness stuff
    IEnumerable<SpatialAwarenessMeshObject> meshes;
    IMixedRealitySpatialAwarenessMeshObserver observer = null;

    long prev_ts;
    uint framesRcvd;
    string debugString = "";

    private void Awake()
    {
#if ENABLE_WINMD_SUPPORT
        unityWorldOrigin = Windows.Perception.Spatial.SpatialLocator.GetDefault().CreateStationaryFrameOfReferenceAtCurrentLocation().CoordinateSystem;
#endif
    }

    // Start is called before the first frame update
    async void Start()
    {
        this.loggerObject = GameObject.Find("Logger");

        // Connect to the python TCP servers
        this.tcpClient1 = new System.Net.Sockets.TcpClient();
        this.tcpClient2 = new System.Net.Sockets.TcpClient();
        this.tcpClient3 = new System.Net.Sockets.TcpClient();
        this.tcpClient4 = new System.Net.Sockets.TcpClient();
        try
        {
            this.tcpClient1.Connect("169.254.103.120", 11000);
            this.loggerObject.GetComponent<Logger>().LogInfo("TCP client 1 connected!");
            this.tcpStream1 = this.tcpClient1.GetStream();

            this.tcpClient2.Connect("169.254.103.120", 11001);
            this.loggerObject.GetComponent<Logger>().LogInfo("TCP client 2 connected!");
            this.tcpStream2 = this.tcpClient2.GetStream();

            this.tcpClient3.Connect("169.254.103.120", 11002);
            this.loggerObject.GetComponent<Logger>().LogInfo("TCP client 3 connected!");
            this.tcpStream3 = this.tcpClient3.GetStream();

            this.tcpClient4.Connect("169.254.103.120", 11003);
            this.loggerObject.GetComponent<Logger>().LogInfo("TCP client 4 connected!");
            this.tcpStream4 = this.tcpClient4.GetStream();
        }
        catch (Exception e)
        {
            this.loggerObject.GetComponent<Logger>().LogInfo(e.ToString());
        }

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

        // Start the publishing thread
        Thread tLFCameraThread = new Thread(LFCameraThread);
        tLFCameraThread.Start();
        Thread tRFCameraThread = new Thread(RFCameraThread);
        tRFCameraThread.Start();
        Thread tLLCameraThread = new Thread(LLCameraThread);
        tLLCameraThread.Start();
        Thread tRRCameraThread = new Thread(RRCameraThread);
        tRRCameraThread.Start();
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
                    //observer.LevelOfDetail = SpatialAwarenessMeshLevelOfDetail.Unlimited;
                    //observer.UpdateInterval = 0.5f;
                    this.loggerObject.GetComponent<Logger>().LogInfo("Detail level: " + observer.LevelOfDetail.ToString());
                    this.loggerObject.GetComponent<Logger>().LogInfo("Update interval: " + observer.UpdateInterval.ToString());
                }
            }
        }

        if (debugString != "")
        {
            //this.loggerObject.GetComponent<Logger>().LogInfo(debugString);
        }
    }

    public void LFCameraThread()
    {
#if ENABLE_WINMD_SUPPORT
        bool sendImage = true;
        while (true)
        {
            // Try to get the frame from research mode
            if (researchMode.LFImageUpdated())
            {
                long ts;
                byte [] framePayload = researchMode.GetLFCameraBuffer(out ts);

                // only send every other image
                if (!(sendImage))
                {
                    sendImage = true;
                    continue;
                }
                //debugString = (ts - prev_ts).ToString();

                //prev_ts = ts;
                if (framePayload.Length > 0)
                {
                    // Prepend width and length
                    uint width = 640;
                    uint height = 480;
                    byte[] frameHeader = { 0x1A, 0xCF, 0xFC, 0x1D,
                                        (byte)(((framePayload.Length + 8) & 0xFF000000) >> 24),
                                        (byte)(((framePayload.Length + 8) & 0x00FF0000) >> 16),
                                        (byte)(((framePayload.Length + 8) & 0x0000FF00) >> 8),
                                        (byte)(((framePayload.Length + 8) & 0x000000FF) >> 0),
                                        (byte)((width & 0xFF000000) >> 24),
                                        (byte)((width & 0x00FF0000) >> 16),
                                        (byte)((width & 0x0000FF00) >> 8),
                                        (byte)((width & 0x000000FF) >> 0),
                                        (byte)((height & 0xFF000000) >> 24),
                                        (byte)((height & 0x00FF0000) >> 16),
                                        (byte)((height & 0x0000FF00) >> 8),
                                        (byte)((height & 0x000000FF) >> 0) };

                    byte[] frame = new byte[framePayload.Length + 16];

                    System.Buffer.BlockCopy(frameHeader, 0, frame, 0, frameHeader.Length);
                    System.Buffer.BlockCopy(framePayload, 0, frame, frameHeader.Length, framePayload.Length);

                    // Send the data through the socket.  
                    tcpStream1.Write(frame, 0, frame.Length);
                    tcpStream1.Flush();
                    //sendImage = false;
                } // end if length > 0
            } // end if image available

            Thread.Sleep(3);
        } // end while loop
#endif
    } // end method

    public void RFCameraThread()
    {
#if ENABLE_WINMD_SUPPORT
        bool sendImage = true;

        while (true)
        {
            // Try to get the frame from research mode
            if (researchMode.RFImageUpdated())
            {
                long ts;
                byte [] framePayload = researchMode.GetRFCameraBuffer(out ts);

                // only send every other image
                if (!(sendImage))
                {
                    sendImage = true;
                    continue;
                }
                //debugString = (ts - prev_ts).ToString();

                //prev_ts = ts;
                if (framePayload.Length > 0)
                {
                    // Prepend width and length
                    uint width = 640;
                    uint height = 480;
                    byte[] frameHeader = { 0x1A, 0xCF, 0xFC, 0x1D,
                                        (byte)(((framePayload.Length + 8) & 0xFF000000) >> 24),
                                        (byte)(((framePayload.Length + 8) & 0x00FF0000) >> 16),
                                        (byte)(((framePayload.Length + 8) & 0x0000FF00) >> 8),
                                        (byte)(((framePayload.Length + 8) & 0x000000FF) >> 0),
                                        (byte)((width & 0xFF000000) >> 24),
                                        (byte)((width & 0x00FF0000) >> 16),
                                        (byte)((width & 0x0000FF00) >> 8),
                                        (byte)((width & 0x000000FF) >> 0),
                                        (byte)((height & 0xFF000000) >> 24),
                                        (byte)((height & 0x00FF0000) >> 16),
                                        (byte)((height & 0x0000FF00) >> 8),
                                        (byte)((height & 0x000000FF) >> 0) };

                    byte[] frame = new byte[framePayload.Length + 16];

                    System.Buffer.BlockCopy(frameHeader, 0, frame, 0, frameHeader.Length);
                    System.Buffer.BlockCopy(framePayload, 0, frame, frameHeader.Length, framePayload.Length);

                    // Send the data through the socket.  
                    tcpStream2.Write(frame, 0, frame.Length);
                    tcpStream2.Flush();
                    //sendImage = false;

                } // end if length > 0
            } // end if image available

            Thread.Sleep(3);
        } // end while loop
#endif
    } // end method

    public void LLCameraThread()
    {
#if ENABLE_WINMD_SUPPORT
        bool sendImage = true;

        while (true)
        {
            // Try to get the frame from research mode
            if (researchMode.LLImageUpdated())
            {
                long ts;
                byte [] framePayload = researchMode.GetLLCameraBuffer(out ts);

                //debugString = (ts - prev_ts).ToString();
                // only send every other image
                if (!(sendImage))
                {
                    sendImage = true;
                    continue;
                }
                //prev_ts = ts;
                if (framePayload.Length > 0)
                {
                    // Prepend width and length
                    uint width = 640;
                    uint height = 480;
                    byte[] frameHeader = { 0x1A, 0xCF, 0xFC, 0x1D,
                                        (byte)(((framePayload.Length + 8) & 0xFF000000) >> 24),
                                        (byte)(((framePayload.Length + 8) & 0x00FF0000) >> 16),
                                        (byte)(((framePayload.Length + 8) & 0x0000FF00) >> 8),
                                        (byte)(((framePayload.Length + 8) & 0x000000FF) >> 0),
                                        (byte)((width & 0xFF000000) >> 24),
                                        (byte)((width & 0x00FF0000) >> 16),
                                        (byte)((width & 0x0000FF00) >> 8),
                                        (byte)((width & 0x000000FF) >> 0),
                                        (byte)((height & 0xFF000000) >> 24),
                                        (byte)((height & 0x00FF0000) >> 16),
                                        (byte)((height & 0x0000FF00) >> 8),
                                        (byte)((height & 0x000000FF) >> 0) };

                    byte[] frame = new byte[framePayload.Length + 16];

                    System.Buffer.BlockCopy(frameHeader, 0, frame, 0, frameHeader.Length);
                    System.Buffer.BlockCopy(framePayload, 0, frame, frameHeader.Length, framePayload.Length);

                    // Send the data through the socket.  
                    tcpStream3.Write(frame, 0, frame.Length);
                    tcpStream3.Flush();

                    //sendImage = false;

                } // end if length > 0
            } // end if image available

            Thread.Sleep(3);
        } // end while loop
#endif
    } // end method

    public void RRCameraThread()
    {
#if ENABLE_WINMD_SUPPORT
        bool sendImage = true;

        while (true)
        {
            // Try to get the frame from research mode
            if (researchMode.RRImageUpdated())
            {
                long ts;
                byte [] framePayload = researchMode.GetRRCameraBuffer(out ts);

                // only send every other image
                if (!(sendImage))
                {
                    sendImage = true;
                    continue;
                }

                //debugString = (ts - prev_ts).ToString();

                //prev_ts = ts;
                if (framePayload.Length > 0)
                {
                    // Prepend width and length
                    uint width = 640;
                    uint height = 480;
                    byte[] frameHeader = { 0x1A, 0xCF, 0xFC, 0x1D,
                                        (byte)(((framePayload.Length + 8) & 0xFF000000) >> 24),
                                        (byte)(((framePayload.Length + 8) & 0x00FF0000) >> 16),
                                        (byte)(((framePayload.Length + 8) & 0x0000FF00) >> 8),
                                        (byte)(((framePayload.Length + 8) & 0x000000FF) >> 0),
                                        (byte)((width & 0xFF000000) >> 24),
                                        (byte)((width & 0x00FF0000) >> 16),
                                        (byte)((width & 0x0000FF00) >> 8),
                                        (byte)((width & 0x000000FF) >> 0),
                                        (byte)((height & 0xFF000000) >> 24),
                                        (byte)((height & 0x00FF0000) >> 16),
                                        (byte)((height & 0x0000FF00) >> 8),
                                        (byte)((height & 0x000000FF) >> 0) };

                    byte[] frame = new byte[framePayload.Length + 16];

                    System.Buffer.BlockCopy(frameHeader, 0, frame, 0, frameHeader.Length);
                    System.Buffer.BlockCopy(framePayload, 0, frame, frameHeader.Length, framePayload.Length);

                    // Send the data through the socket.  
                    tcpStream4.Write(frame, 0, frame.Length);
                    tcpStream4.Flush();
                    //sendImage = false;
                } // end if length > 0
            } // end if image available

            Thread.Sleep(3);
        } // end while loop
#endif
    } // end method

}