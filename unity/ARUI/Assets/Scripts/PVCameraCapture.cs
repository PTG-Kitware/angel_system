using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.BuiltinInterfaces;
using RosMessageTypes.Std;
using RosMessageTypes.Sensor;
using RosMessageTypes.Angel;

#if ENABLE_WINMD_SUPPORT
using Microsoft.MixedReality.Toolkit;
using Windows.Graphics.Imaging;
using Windows.Media;
using Windows.Media.Capture;
using Windows.Media.Capture.Frames;
using Windows.Media.MediaProperties;
using Windows.Media.Devices.Core;
using Windows.Perception.Spatial;
using Windows.Perception;
using Windows.Storage.Streams;
using System.Runtime.InteropServices.WindowsRuntime;
#endif

public class PVCameraCapture : MonoBehaviour
{
#if ENABLE_WINMD_SUPPORT
    MediaCapture mediaCapture = null;
    private MediaFrameReader frameReader = null;
    private byte[] frameData = null;
#endif

    // Ros stuff
    ROSConnection ros;
    public string imageTopicName = "PVFramesNV12";
    public string headsetPoseTopicName = "HeadsetPoseData";

    // Default frame size / frame rate - available resolutions:
    // https://docs.microsoft.com/en-us/windows/mixed-reality/develop/advanced-concepts/locatable-camera-overview
    public int targetVideoHeight = 720;
    public int targetVideoWidth = 1280;
    public float targetVideoFrameRate = 30.0f;

    private Logger _logger = null;
    string debugString = "";

    Matrix4x4 projectionMatrix;
    float[] projectionMatrixAsFloat;

    private static Mutex projectionMatrixMut = new Mutex();
    private readonly Task initMediaCaptureTask;

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

    // Start is called before the first frame update
    async void Start()
    {
        Logger log = logger();

        // Wait for media capture initialization to finish
        await this.initMediaCaptureTask;

        // Create the image publisher and headset pose publisher
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<ImageMsg>(imageTopicName);
        ros.RegisterPublisher<HeadsetPoseDataMsg>(headsetPoseTopicName);

#if ENABLE_WINMD_SUPPORT
        // Update the projection matrix
        projectionMatrix = Camera.main.projectionMatrix;
        projectionMatrixAsFloat = PTGUtilities.ConvertUnityMatrixToFloatArray(projectionMatrix);

        if (this.frameReader != null)
        {
            var status = await this.frameReader.StartAsync();
            if (!(status == MediaFrameReaderStartStatus.Success))
            {
                log.LogInfo("StartFrameReaderAsyncTask() is not successful, status = " + status);
            }
            else
            {
                log.LogInfo("StartFrameReaderAsyncTask() is successful");
                this.frameReader.FrameArrived += OnFrameArrived;
                log.LogInfo("Media capture started");
            }
        }
#endif
    }

    public PVCameraCapture()
    {
        // Call this here (rather than in the Start() method, which is executed on the thread pool) to
        // ensure that MediaCapture.InitializeAsync() is called from an STA thread (this constructor must
        // itself be called from an STA thread in order for this to be true). Calls from an MTA thread may
        // result in undefined behavior, per the following documentation:
        // https://docs.microsoft.com/en-us/uwp/api/windows.media.capture.mediacapture.initializeasync
#if ENABLE_WINMD_SUPPORT
        this.initMediaCaptureTask = this.InitializeMediaCaptureAsyncTask();
#endif
    }

    void Update()
    {
        if (debugString != "")
        {
            this.logger().LogInfo(debugString);
            debugString = "";
        }

#if ENABLE_WINMD_SUPPORT
        // Update the projection matrix
        projectionMatrixMut.WaitOne();
        projectionMatrix = Camera.main.projectionMatrix;
        projectionMatrixAsFloat = PTGUtilities.ConvertUnityMatrixToFloatArray(projectionMatrix);
        projectionMatrixMut.ReleaseMutex();
#endif
    }

#if ENABLE_WINMD_SUPPORT
    // The following functions were adapted from:
    // https://github.com/qian256/HoloLensARToolKit/blob/master/HoloLensARToolKit/Assets/ARToolKitUWP/Scripts/ARUWPVideo.cs
    public async Task<bool> InitializeMediaCaptureAsyncTask()
    {
        var allGroups = await MediaFrameSourceGroup.FindAllAsync();
        int selectedGroupIndex = -1;
        for (int i = 0; i < allGroups.Count; i++)
        {
            var group = allGroups[i];
            if (group.DisplayName == "QC Back Camera")
            {
                selectedGroupIndex = i;
                break;
            }
        }

        if (selectedGroupIndex == -1)
        {
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
                if (f.VideoFormat.Width == targetVideoWidth && f.VideoFormat.Height == targetVideoHeight)
                {
                    if (targetResFormat == null)
                    {
                        targetResFormat = f;
                        framerateDiffMin = Mathf.Abs(f.FrameRate.Numerator / f.FrameRate.Denominator - targetVideoFrameRate);
                    }
                    else if (Mathf.Abs(f.FrameRate.Numerator / f.FrameRate.Denominator - targetVideoFrameRate) < framerateDiffMin)
                    {
                        targetResFormat = f;
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

            frameData = new byte[(int) (targetResFormat.VideoFormat.Width * targetResFormat.VideoFormat.Height * 1.5)];

            await mediaFrameSourceVideo.SetFormatAsync(targetResFormat);

            this.frameReader = await mediaCapture.CreateFrameReaderAsync(mediaFrameSourceVideo, targetResFormat.Subtype);
        }
        catch (Exception e)
        {
            return false;
        }

        return true;
    }

    unsafe private void OnFrameArrived(MediaFrameReader sender, MediaFrameArrivedEventArgs args)
    {
        using (var frame = sender.TryAcquireLatestFrame())
        {
            if (frame != null)
            {
                float[] cameraToWorldMatrixAsFloat = null;
                try
                {
                    if (HL2TryGetCameraToWorldMatrix(frame, out cameraToWorldMatrixAsFloat) == false)
                    {
                        debugString += "HL2TryGetCameraToWorldMatrix failed";
                    }
                }
                catch (Exception e)
                {
                    debugString += e.ToString();
                }

                var originalSoftwareBitmap = frame.VideoMediaFrame.SoftwareBitmap;

                using (var input = originalSoftwareBitmap.LockBuffer(BitmapBufferAccessMode.Read))
                using (var inputReference = input.CreateReference())
                {
                    byte* inputBytes;
                    uint inputCapacity;
                    ((IMemoryBufferByteAccess)inputReference).GetBuffer(out inputBytes, out inputCapacity);

                    Marshal.Copy((IntPtr)inputBytes, frameData, 0, (int)inputCapacity);

                    uint width = Convert.ToUInt32(originalSoftwareBitmap.PixelWidth);
                    uint height = Convert.ToUInt32(originalSoftwareBitmap.PixelHeight);

                    HeaderMsg header = PTGUtilities.getROSStdMsgsHeader("PVFramesNV12");

                    ImageMsg image = new ImageMsg(header,
                                                  height,
                                                  width,
                                                  "nv12",
                                                  0,
                                                  Convert.ToUInt32(width * 1.5),
                                                  frameData);

                    ros.Publish(imageTopicName, image);

                    // Build and publish the headpose data message
                    projectionMatrixMut.WaitOne();
                    HeadsetPoseDataMsg pose = new HeadsetPoseDataMsg(header, cameraToWorldMatrixAsFloat,
                                                                     projectionMatrixAsFloat);
                    projectionMatrixMut.ReleaseMutex();

                    ros.Publish(headsetPoseTopicName, pose);
                }

                originalSoftwareBitmap?.Dispose();

            }
        }
    }

	public bool HL2TryGetCameraToWorldMatrix(MediaFrameReference frameReference, out float[] outMatrix)
	{
        if (SpatialMappingCapture.unityCoordinateSystem == null)
        {
            outMatrix = PTGUtilities.GetIdentityMatrixFloatArray();
            return false;
        }

        SpatialCoordinateSystem cameraCoordinateSystem = frameReference.CoordinateSystem;

        if (cameraCoordinateSystem == null)
        {
            outMatrix = PTGUtilities.GetIdentityMatrixFloatArray();
            return false;
        }

        System.Numerics.Matrix4x4 cameraToUnityMatrixNumericsMatrix = (System.Numerics.Matrix4x4) cameraCoordinateSystem.TryGetTransformTo(SpatialMappingCapture.unityCoordinateSystem);

        if (cameraToUnityMatrixNumericsMatrix == null)
        {
            outMatrix = PTGUtilities.GetIdentityMatrixFloatArray();
            return false;
        }

        UnityEngine.Matrix4x4 cameraToUnityMatrixUnityMatrix = SystemNumericsExtensions.ToUnity(cameraToUnityMatrixNumericsMatrix);
        outMatrix = PTGUtilities.ConvertUnityMatrixToFloatArray(cameraToUnityMatrixUnityMatrix);

        return true;
	}
#endif

}