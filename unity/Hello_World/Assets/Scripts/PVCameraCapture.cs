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
using System.Net.NetworkInformation;
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
using System.Runtime.InteropServices.WindowsRuntime;
#endif

public class PVCameraCapture : MonoBehaviour
{

#if ENABLE_WINMD_SUPPORT
    MediaCapture mediaCapture = null;
    private MediaFrameReader frameReader = null;
    private byte[] frameData = null;
#endif

    // Network stuff
    System.Net.Sockets.TcpClient tcpClient;
    System.Net.Sockets.TcpListener tcpServer;
    NetworkStream tcpStream;

    private Logger _logger = null;

    long prev_ts;
    uint framesRcvd;
    string debugString = "";

    public string TcpServerIPAddr = "";
    public const int PVTcpPort = 11008;

    [ComImport]
    [Guid("5B0D3235-4DBA-4D44-865E-8F1D0E4FD04D")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    unsafe interface IMemoryBufferByteAccess
    {
        /// <summary>
        /// Unsafe function to retrieve the pointer and size information of the underlying
        /// buffer object. Must be used within unsafe functions. In addition, the project needs
        /// to be configured as "Allow unsafe code". [internal use]
        /// </summary>
        /// <param name="buffer">byte pointer to the start of the buffer</param>
        /// <param name="capacity">the size of the buffer</param>
        void GetBuffer(out byte* buffer, out uint capacity);
    }

    /// <summary>
    /// Lazy acquire the logger object and return the reference to it.
    /// </summary>
    /// <returns>Logger instance reference.</returns>
    private ref Logger logger()
    {
        if (this._logger == null)
        {
            // TODO: Error handling for null loggerObject?
            this._logger = GameObject.Find("Logger").GetComponent<Logger>();
        }
        return ref this._logger;
    }

    private int GetIPv4AddressString()
    {
        int status = -1;
        NetworkInterface[] interfaces = NetworkInterface.GetAllNetworkInterfaces();
        foreach (NetworkInterface adapter in interfaces)
        {
            if (adapter.Supports(NetworkInterfaceComponent.IPv4) &&
                adapter.OperationalStatus == OperationalStatus.Up &&
                adapter.NetworkInterfaceType == NetworkInterfaceType.Ethernet)
            {
                foreach (UnicastIPAddressInformation ip in adapter.GetIPProperties().UnicastAddresses)
                {
                    if (ip.Address.AddressFamily == System.Net.Sockets.AddressFamily.InterNetwork)
                    {
                        TcpServerIPAddr = ip.Address.ToString();
                        status = 0;
                        break;
                    }
                }
            }
        }

        return status;
    }

    // Start is called before the first frame update
    async void Start()
    {
        Logger log = logger();

        if (GetIPv4AddressString() != 0)
        {
            log.LogInfo("Could not get valid IPv4 address. Exiting.");
            return;
        }

        log.LogInfo("Using IPv4 addr: " + TcpServerIPAddr);

        Thread tPVCapture = new Thread(SetupPVCapture);
        tPVCapture.Start();
        log.LogInfo("Waiting for PV TCP connections");

#if ENABLE_WINMD_SUPPORT
        await InitializeMediaCaptureAsyncTask();

        MediaFrameReaderStartStatus mediaFrameReaderStartStatus = await frameReader.StartAsync();
        if (!(mediaFrameReaderStartStatus == MediaFrameReaderStartStatus.Success))
        {
            log.LogInfo("StartFrameReaderAsyncTask() is not successful, status = " + mediaFrameReaderStartStatus);
        }

        log.LogInfo("Media capture started");
#endif
    }

#if ENABLE_WINMD_SUPPORT
    public async Task<bool> InitializeMediaCaptureAsyncTask()
    {
        int targetVideoWidth, targetVideoHeight;
        float targetVideoFrameRate;
        targetVideoWidth = 1280;
        targetVideoHeight = 720;
        //targetVideoWidth = 1920;
        //targetVideoHeight = 1080;
        targetVideoFrameRate = 30.0f;

        var allGroups = await MediaFrameSourceGroup.FindAllAsync();
        int selectedGroupIndex = -1;
        for (int i = 0; i < allGroups.Count; i++)
        {
            var group = allGroups[i];
            //this.logger().LogInfo(group.DisplayName + ", " + group.Id + " " + allGroups.Count.ToString());
            //this.logger().LogInfo(group.DisplayName);
            if (group.DisplayName == "QC Back Camera")
            {
                selectedGroupIndex = i;
                this.logger().LogInfo("Selected group " + i + " on HoloLens 2");
                break;
            }
        }

        if (selectedGroupIndex == -1)
        {
            this.logger().LogInfo("InitializeMediaCaptureAsyncTask() fails because there is no suitable source group");
            return false;
        }

        // Initialize mediacapture with the source group.
        mediaCapture = new MediaCapture();
        MediaStreamType mediaStreamType = MediaStreamType.VideoPreview;

        string deviceId = allGroups[selectedGroupIndex].Id;

        // Look up for all video profiles
        IReadOnlyList<MediaCaptureVideoProfile> profileList = MediaCapture.FindKnownVideoProfiles(deviceId, KnownVideoProfile.VideoConferencing);

        // Initialize mediacapture with the source group.
        var settings = new MediaCaptureInitializationSettings
        {
            SourceGroup = allGroups[selectedGroupIndex],
            VideoDeviceId = deviceId,
            VideoProfile = profileList[0],

            // This media capture can share streaming with other apps.
            SharingMode = MediaCaptureSharingMode.ExclusiveControl,

            // Only stream video and don't initialize audio capture devices.
            StreamingCaptureMode = StreamingCaptureMode.Video,

            // Set to CPU to ensure frames always contain CPU SoftwareBitmap images
            // instead of preferring GPU D3DSurface images.
            MemoryPreference = MediaCaptureMemoryPreference.Cpu
        };
        
        try
        {
            await mediaCapture.InitializeAsync(settings);

            var mediaFrameSourceVideo = mediaCapture.FrameSources.Values.Single(x => x.Info.MediaStreamType == mediaStreamType);
            MediaFrameFormat targetResFormat = null;
            float framerateDiffMin = 60f;
            foreach (var f in mediaFrameSourceVideo.SupportedFormats.OrderBy(x => x.VideoFormat.Width * x.VideoFormat.Height))
            {
                //this.logger().LogInfo("Format width: " + f.VideoFormat.Width.ToString());
                //this.logger().LogInfo("Format height: " + f.VideoFormat.Height.ToString());

                if (f.VideoFormat.Width == targetVideoWidth && f.VideoFormat.Height == targetVideoHeight)
                {
                    if (targetResFormat == null)
                    {
                        //this.logger().LogInfo("Found matching format");
                        targetResFormat = f;
                        framerateDiffMin = Mathf.Abs(f.FrameRate.Numerator / f.FrameRate.Denominator - targetVideoFrameRate);
                    }
                    else if (Mathf.Abs(f.FrameRate.Numerator / f.FrameRate.Denominator - targetVideoFrameRate) < framerateDiffMin)
                    {
                        targetResFormat = f;
                        //this.logger().LogInfo("Else?");
                        framerateDiffMin = Mathf.Abs(f.FrameRate.Numerator / f.FrameRate.Denominator - targetVideoFrameRate);
                    }
                }
            }
            if (targetResFormat == null)
            {
                targetResFormat = mediaFrameSourceVideo.SupportedFormats[0];
                this.logger().LogInfo("Unable to choose the selected format, fall back");
                targetResFormat = mediaFrameSourceVideo.SupportedFormats.OrderBy(x => x.VideoFormat.Width * x.VideoFormat.Height).FirstOrDefault();
            }

            await mediaFrameSourceVideo.SetFormatAsync(targetResFormat);
            //this.logger().LogInfo("Sub type " + targetResFormat.Subtype);

            frameReader = await mediaCapture.CreateFrameReaderAsync(mediaFrameSourceVideo, targetResFormat.Subtype);
            frameReader.FrameArrived += OnFrameArrived;

            frameData = new byte[(int) (targetResFormat.VideoFormat.Width * targetResFormat.VideoFormat.Height * 1.5) + 16];
            this.logger().LogInfo("FrameReader is successfully initialized, " + targetResFormat.VideoFormat.Width + "x" + targetResFormat.VideoFormat.Height +
                ", Framerate: " + targetResFormat.FrameRate.Numerator + "/" + targetResFormat.FrameRate.Denominator);
        }
        catch (Exception e)
        {
            this.logger().LogInfo("FrameReader is not initialized");
            this.logger().LogInfo("Exception: " + e);
            return false;
        }

        this.logger().LogInfo("InitializeMediaCaptureAsyncTask() is successful");

        return true;
    }

    unsafe private void OnFrameArrived(MediaFrameReader sender, MediaFrameArrivedEventArgs args)
    {
        try
        {
            using (var frame = sender.TryAcquireLatestFrame())
            {
                if (frame != null)
                {
                    /*
                    float[] cameraToWorldMatrixAsFloat = null;                
                    if (HL2TryGetCameraToWorldMatrix(frame, out cameraToWorldMatrixAsFloat) == false)
                    {
                        this.logger().LogInfo("HL2TryGetCameraToWorldMatrix failed");
                        return;
                    }

                    latestLocatableCameraToWorld = ConvertFloatArrayToMatrix4x4(cameraToWorldMatrixAsFloat);
                    */

                    var originalSoftwareBitmap = frame.VideoMediaFrame.SoftwareBitmap;

                    //debugString += "height: " + newSoftwareBitmap.PixelHeight.ToString() + "\n";
                    //debugString += "width: " + newSoftwareBitmap.PixelWidth.ToString() + "\n";
                    //debugString += "pixel f: " + newSoftwareBitmap.BitmapPixelFormat.ToString() + "\n";

                    using (var input = originalSoftwareBitmap.LockBuffer(BitmapBufferAccessMode.Read))
                    using (var inputReference = input.CreateReference())
                    {
                        byte* inputBytes;
                        uint inputCapacity;
                        ((IMemoryBufferByteAccess)inputReference).GetBuffer(out inputBytes, out inputCapacity);

                        //debugString = "capacity: " + inputCapacity.ToString() + "\n";
                        //debugString += "alpha mode: " + originalSoftwareBitmap.BitmapAlphaMode.ToString() + "\n";
                        //debugString = "height: " + originalSoftwareBitmap.PixelHeight.ToString() + "\n";
                        //debugString += "width: " + originalSoftwareBitmap.PixelWidth.ToString() + "\n";

                        // add header
                        byte[] frameHeader = { 0x1A, 0xCF, 0xFC, 0x1D,
                                        (byte)(((inputCapacity + 8) & 0xFF000000) >> 24),
                                        (byte)(((inputCapacity + 8) & 0x00FF0000) >> 16),
                                        (byte)(((inputCapacity + 8) & 0x0000FF00) >> 8),
                                        (byte)(((inputCapacity + 8) & 0x000000FF) >> 0),
                                        (byte)((originalSoftwareBitmap.PixelWidth & 0xFF000000) >> 24),
                                        (byte)((originalSoftwareBitmap.PixelWidth & 0x00FF0000) >> 16),
                                        (byte)((originalSoftwareBitmap.PixelWidth & 0x0000FF00) >> 8),
                                        (byte)((originalSoftwareBitmap.PixelWidth & 0x000000FF) >> 0),
                                        (byte)((originalSoftwareBitmap.PixelHeight & 0xFF000000) >> 24),
                                        (byte)((originalSoftwareBitmap.PixelHeight & 0x00FF0000) >> 16),
                                        (byte)((originalSoftwareBitmap.PixelHeight & 0x0000FF00) >> 8),
                                        (byte)((originalSoftwareBitmap.PixelHeight & 0x000000FF) >> 0) };
                        System.Buffer.BlockCopy(frameHeader, 0, frameData, 0, frameHeader.Length);

                        Marshal.Copy((IntPtr)inputBytes, frameData, 16, (int)inputCapacity);

                        // Send the data through the socket.
                        if (tcpStream != null)
                        {
                            tcpStream.Write(frameData, 0, frameData.Length);
                            tcpStream.Flush();
                        }
                        originalSoftwareBitmap?.Dispose();
                    }
                }
            }
        }
        catch (Exception e)
        {
            debugString += ("Frame Arrived Exception: " + e);
        }
    }
#endif

    void Update()
    {
#if ENABLE_WINMD_SUPPORT
        if (debugString != "")
        {
            //this.logger().LogInfo(debugString);
        }
#endif
    }

    void SetupPVCapture()
    {
#if ENABLE_WINMD_SUPPORT
        try
        {
            IPAddress localAddr = IPAddress.Parse(TcpServerIPAddr);

            // TcpListener server = new TcpListener(port);
            tcpServer = new TcpListener(localAddr, PVTcpPort);

            // Start listening for client requests.
            tcpServer.Start();

            // Perform a blocking call to accept requests.
            // You could also use server.AcceptSocket() here.
            tcpClient = tcpServer.AcceptTcpClient();
            tcpStream = tcpClient.GetStream();
        }
        catch (Exception e)
        {
            debugString += e.ToString();
        }
#endif
    }

}