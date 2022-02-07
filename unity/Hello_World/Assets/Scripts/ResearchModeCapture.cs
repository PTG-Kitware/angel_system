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
    DepthSensorMode depthSensorMode = DepthSensorMode.ShortThrow;
    bool enablePointCloud = false;

    Windows.Perception.Spatial.SpatialCoordinateSystem unityWorldOrigin;
#endif

    // Network stuff
    System.Net.Sockets.TcpClient tcpClientLF;
    NetworkStream tcpStreamLF;
    System.Net.Sockets.TcpClient tcpClientRF;
    NetworkStream tcpStreamRF;
    System.Net.Sockets.TcpClient tcpClientLL;
    NetworkStream tcpStreamLL;
    System.Net.Sockets.TcpClient tcpClientRR;
    NetworkStream tcpStreamRR;
    System.Net.Sockets.TcpClient tcpClientDepth;
    NetworkStream tcpStreamDepth;
    System.Net.Sockets.TcpClient tcpClientAbDepth;
    NetworkStream tcpStreamAbDepth;

    public const int LF_VLC_Tcp_Port = 11000;
    public const int RF_VLC_Tcp_Port = 11001;
    public const int LL_VLC_Tcp_Port = 11002;
    public const int RR_VLC_Tcp_Port = 11003;
    public const int Depth_Tcp_Port = 11004;
    public const int Depth_Ab_Tcp_Port = 11005;
    public const string TcpServerIPAddr = "169.254.103.120";

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
    async void Start()
    {
        this.loggerObject = GameObject.Find("Logger");

        /*
        // Connect to the python TCP servers
        this.tcpClientLF = new System.Net.Sockets.TcpClient();
        this.tcpClientRF = new System.Net.Sockets.TcpClient();
        this.tcpClientLL = new System.Net.Sockets.TcpClient();
        this.tcpClientRR = new System.Net.Sockets.TcpClient();
        this.tcpClientDepth = new System.Net.Sockets.TcpClient();
        this.tcpClientAbDepth = new System.Net.Sockets.TcpClient();
        try
        {
            this.tcpClientLF.Connect(TcpServerIPAddr, LF_VLC_Tcp_Port);
            this.loggerObject.GetComponent<Logger>().LogInfo("TCP client LF connected!");
            this.tcpStreamLF = this.tcpClientLF.GetStream();

            this.tcpClientRF.Connect(TcpServerIPAddr, RF_VLC_Tcp_Port);
            this.loggerObject.GetComponent<Logger>().LogInfo("TCP client RF connected!");
            this.tcpStreamRF = this.tcpClientRF.GetStream();

            this.tcpClientLL.Connect(TcpServerIPAddr, LL_VLC_Tcp_Port);
            this.loggerObject.GetComponent<Logger>().LogInfo("TCP client LL connected!");
            this.tcpStreamLL = this.tcpClientLL.GetStream();

            this.tcpClientRR.Connect(TcpServerIPAddr, RR_VLC_Tcp_Port);
            this.loggerObject.GetComponent<Logger>().LogInfo("TCP client RR connected!");
            this.tcpStreamRR = this.tcpClientRR.GetStream();

            this.tcpClientDepth.Connect(TcpServerIPAddr, Depth_Tcp_Port);
            this.loggerObject.GetComponent<Logger>().LogInfo("TCP client Depth connected!");
            this.tcpStreamDepth = this.tcpClientDepth.GetStream();

            this.tcpClientAbDepth.Connect(TcpServerIPAddr, Depth_Ab_Tcp_Port);
            this.loggerObject.GetComponent<Logger>().LogInfo("TCP client AB Depth connected!");
            this.tcpStreamAbDepth = this.tcpClientAbDepth.GetStream();
        }
        catch (Exception e)
        {
            this.loggerObject.GetComponent<Logger>().LogInfo(e.ToString());
        }
        */

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
        //Thread tLFCameraThread = new Thread(LFCameraThread);
        //tLFCameraThread.Start();
        //Thread tRFCameraThread = new Thread(RFCameraThread);
        //tRFCameraThread.Start();
        //Thread tLLCameraThread = new Thread(LLCameraThread);
        //tLLCameraThread.Start();
        //Thread tRRCameraThread = new Thread(RRCameraThread);
        //tRRCameraThread.Start();
        //Thread tDepthCameraThread = new Thread(DepthCameraThread);
        //tDepthCameraThread.Start();
        //Thread tDepthCameraAbThread = new Thread(DepthCameraAbThread);
        //tDepthCameraAbThread.Start();
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
                    this.loggerObject.GetComponent<Logger>().LogInfo("Update interval: " + observer.UpdateInterval.ToString());
                }
            }
        }

#if ENABLE_WINMD_SUPPORT

        if (researchMode.PrintDebugString() != "")
        {
            this.loggerObject.GetComponent<Logger>().LogInfo(researchMode.PrintDebugString());
        }
#endif

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
                    tcpStreamLF.Write(frame, 0, frame.Length);
                    tcpStreamLF.Flush();
                    //sendImage = false;
                } // end if length > 0
            } // end if image available

            Thread.Sleep(5);
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
                    tcpStreamRF.Write(frame, 0, frame.Length);
                    tcpStreamRF.Flush();
                    //sendImage = false;

                } // end if length > 0
            } // end if image available

            Thread.Sleep(5);
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

                // only send every other image
                if (!(sendImage))
                {
                    sendImage = true;
                    continue;
                }
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
                    tcpStreamLL.Write(frame, 0, frame.Length);
                    tcpStreamLL.Flush();

                    //sendImage = false;

                } // end if length > 0
            } // end if image available

            Thread.Sleep(5);
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
                    tcpStreamRR.Write(frame, 0, frame.Length);
                    tcpStreamRR.Flush();
                    //sendImage = false;
                } // end if length > 0
            } // end if image available

            Thread.Sleep(5);
        } // end while loop
#endif
    } // end method

    public void DepthCameraThread()
    {
#if ENABLE_WINMD_SUPPORT
        while (true)
        {
            // Try to get the frame from research mode
            if (researchMode.DepthMapUpdated())
            {
                //debugString = researchMode.PrintDepthResolution();
                //debugString += researchMode.PrintDepthExtrinsics();

                UInt16 [] depthBuffer = researchMode.GetDepthMapBuffer();
                //byte [] depthTexture = researchMode.GetDepthMapTextureBuffer();
                //UInt16 [] abBuffer = researchMode.GetShortAbImageBuffer();
                //byte [] abTexture = researchMode.GetShortAbImageTextureBuffer();

                if (depthBuffer.Length > 0)
                {
                    //debugString = "\nDepth buffer length" + depthBuffer.Length.ToString();
                    //debugString += "\nDepth texture length" + depthTexture.Length.ToString();
                    //debugString += "\nDepth AB buffer length" + abBuffer.Length.ToString();
                    //debugString += "\nDepth AB texture length" + abTexture.Length.ToString();

                    try
                    {
                        // Prepend width and length
                        uint width = 512;
                        uint height = 512;
                        int depthBufferLength = depthBuffer.Length * 2;
                        byte[] frameHeader = { 0x1A, 0xCF, 0xFC, 0x1D,
                                            (byte)(((depthBufferLength + 8) & 0xFF000000) >> 24),
                                            (byte)(((depthBufferLength + 8) & 0x00FF0000) >> 16),
                                            (byte)(((depthBufferLength + 8) & 0x0000FF00) >> 8),
                                            (byte)(((depthBufferLength + 8) & 0x000000FF) >> 0),
                                            (byte)((width & 0xFF000000) >> 24),
                                            (byte)((width & 0x00FF0000) >> 16),
                                            (byte)((width & 0x0000FF00) >> 8),
                                            (byte)((width & 0x000000FF) >> 0),
                                            (byte)((height & 0xFF000000) >> 24),
                                            (byte)((height & 0x00FF0000) >> 16),
                                            (byte)((height & 0x0000FF00) >> 8),
                                            (byte)((height & 0x000000FF) >> 0) };

                        byte[] frame = new byte[(depthBufferLength) + 16];

                        System.Buffer.BlockCopy(frameHeader, 0, frame, 0, frameHeader.Length);

                        for (int i = 0; i < frame.Length - 16; i += 2)
                        {
                            frame[i + 16] = (byte)((depthBuffer[i / 2] & 0xFF00) >> 8); // extract upper byte
                            frame[i + 17] = (byte) (depthBuffer[i / 2] & 0x00FF);   // extract lower byte
                        }
                        
                        //System.Buffer.BlockCopy(depthTexture, 0, frame, frameHeader.Length, depthTexture.Length);

                        // Send the data through the socket.  
                        tcpStreamDepth.Write(frame, 0, frame.Length);
                        tcpStreamDepth.Flush();
                    }
                    catch (Exception e)
                    {
                        debugString += e.ToString();
                    }
                } // end if length > 0
            } // end if image available

            Thread.Sleep(5);
        } // end while loop
#endif
    } // end method

    public void DepthCameraAbThread()
    {
#if ENABLE_WINMD_SUPPORT
        while (true)
        {
            // Try to get the frame from research mode
            if (researchMode.ShortAbImageUpdated())
            {
                //debugString = researchMode.PrintDepthResolution();
                //debugString += researchMode.PrintDepthExtrinsics();

                UInt16 [] AbBuffer = researchMode.GetShortAbImageBuffer();

                if (AbBuffer.Length > 0)
                {
                    try
                    {
                        // Prepend width and length
                        uint width = 512;
                        uint height = 512;
                        int AbBufferLength = AbBuffer.Length * 2;
                        byte[] frameHeader = { 0x1A, 0xCF, 0xFC, 0x1D,
                                            (byte)(((AbBufferLength + 8) & 0xFF000000) >> 24),
                                            (byte)(((AbBufferLength + 8) & 0x00FF0000) >> 16),
                                            (byte)(((AbBufferLength + 8) & 0x0000FF00) >> 8),
                                            (byte)(((AbBufferLength + 8) & 0x000000FF) >> 0),
                                            (byte)((width & 0xFF000000) >> 24),
                                            (byte)((width & 0x00FF0000) >> 16),
                                            (byte)((width & 0x0000FF00) >> 8),
                                            (byte)((width & 0x000000FF) >> 0),
                                            (byte)((height & 0xFF000000) >> 24),
                                            (byte)((height & 0x00FF0000) >> 16),
                                            (byte)((height & 0x0000FF00) >> 8),
                                            (byte)((height & 0x000000FF) >> 0) };

                        byte[] frame = new byte[(AbBufferLength) + 16];

                        System.Buffer.BlockCopy(frameHeader, 0, frame, 0, frameHeader.Length);

                        for (int i = 0; i < frame.Length - 16; i += 2)
                        {
                            frame[i + 16] = (byte)((AbBuffer[i / 2] & 0xFF00) >> 8); // extract upper byte
                            frame[i + 17] = (byte) (AbBuffer[i / 2] & 0x00FF);   // extract lower byte
                        }
                        
                        // Send the data through the socket.  
                        tcpStreamAbDepth.Write(frame, 0, frame.Length);
                        tcpStreamAbDepth.Flush();
                    }
                    catch (Exception e)
                    {
                        debugString += e.ToString();
                    }
                } // end if length > 0
            } // end if image available

            Thread.Sleep(5);
        } // end while loop
#endif
    } // end method


}