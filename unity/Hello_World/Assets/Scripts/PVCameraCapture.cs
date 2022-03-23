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
using UnityEngine.XR.WindowsMR;
using Unity.Robotics.ROSTCPConnector;
using System.Runtime.InteropServices;
using RosMessageTypes.BuiltinInterfaces;
using RosMessageTypes.Std;
using RosMessageTypes.Sensor;
using RosMessageTypes.Angel;


#if ENABLE_WINMD_SUPPORT
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
    // private byte[] frameData = null;
    SpatialCoordinateSystem worldOrigin;
#endif

    // Ros stuff
    ROSConnection ros;
    public string imageTopicName = "PVFramesNV12";
    public string headsetPoseTopicName = "HeadsetPoseData";

    private Logger _logger = null;
    string debugString = "";

    Matrix4x4 projectionMatrix;
    float[] projectionMatrixAsFloat;

    // For filling in ROS message timestamp
    DateTime origin = new DateTime(1970, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc);

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

        // Create the image publisher and headset pose publisher
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<ImageMsg>(imageTopicName);
        ros.RegisterPublisher<HeadsetPoseDataMsg>(headsetPoseTopicName);

#if ENABLE_WINMD_SUPPORT
        await InitializeMediaCaptureAsyncTask();

        MediaFrameReaderStartStatus mediaFrameReaderStartStatus = await frameReader.StartAsync();
        if (!(mediaFrameReaderStartStatus == MediaFrameReaderStartStatus.Success))
        {
            log.LogInfo("StartFrameReaderAsyncTask() is not successful, status = " + mediaFrameReaderStartStatus);
        }

        log.LogInfo("Media capture started");

        // Assuming the projection matrix never changes
        projectionMatrix = Camera.main.projectionMatrix;
        projectionMatrixAsFloat = ConvertUnityMatrixToFloatArray(projectionMatrix);

        // Set the worldOrigin coordinate system to the same system used by the spatial mapping system
        try
        {
            worldOrigin = Marshal.GetObjectForIUnknown(UnityEngine.XR.WindowsMR.WindowsMREnvironment.OriginSpatialCoordinateSystem) as SpatialCoordinateSystem;
            if (worldOrigin == null)
            {
                log.LogInfo("Unable to get world origin");
            }
        }
        catch (Exception e)
        {
            log.LogInfo(e.ToString());
        }
#endif
    }

    void Update()
    {
        if (debugString != "")
        {
            this.logger().LogInfo(debugString);
            debugString = "";
        }

    }

#if ENABLE_WINMD_SUPPORT
    // The following functions were adapted from:
    // https://github.com/qian256/HoloLensARToolKit/blob/master/HoloLensARToolKit/Assets/ARToolKitUWP/Scripts/ARUWPVideo.cs
    public async Task<bool> InitializeMediaCaptureAsyncTask()
    {
        int targetVideoWidth, targetVideoHeight;
        float targetVideoFrameRate;
        //targetVideoWidth = 1280;
        //targetVideoHeight = 720;
        targetVideoWidth = 1920;
        targetVideoHeight = 1080;
        targetVideoFrameRate = 30.0f;

        var allGroups = await MediaFrameSourceGroup.FindAllAsync();
        int selectedGroupIndex = -1;
        for (int i = 0; i < allGroups.Count; i++)
        {
            var group = allGroups[i];
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

            await mediaFrameSourceVideo.SetFormatAsync(targetResFormat);

            frameReader = await mediaCapture.CreateFrameReaderAsync(mediaFrameSourceVideo, targetResFormat.Subtype);
            frameReader.FrameArrived += OnFrameArrived;

            this.logger().LogInfo("FrameReader is successfully initialized, " + targetResFormat.VideoFormat.Width + "x" + targetResFormat.VideoFormat.Height +
                ", Framerate: " + targetResFormat.FrameRate.Numerator + "/" + targetResFormat.FrameRate.Denominator);
        }
        catch (Exception e)
        {
            this.logger().LogInfo("FrameReader is not initialized");
            this.logger().LogInfo("Exception: " + e);
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

                    byte[] frameData = new byte[(int) inputCapacity];
                    Marshal.Copy((IntPtr)inputBytes, frameData, 0, (int)inputCapacity);

                    uint width = Convert.ToUInt32(originalSoftwareBitmap.PixelWidth);
                    uint height = Convert.ToUInt32(originalSoftwareBitmap.PixelHeight);

                    var currTime = DateTime.Now;
                    TimeSpan diff = currTime.ToUniversalTime() - origin;
                    var sec = Convert.ToInt32(Math.Floor(diff.TotalSeconds));
                    var nsec = Convert.ToUInt32((diff.TotalSeconds - sec) * 1e9f);

                    HeaderMsg header = new HeaderMsg(
                        new TimeMsg(sec, nsec),
                        "PVFramesNV12"
                    );

                    ImageMsg image = new ImageMsg(header,
                                                  width,
                                                  height,
                                                  "nv12",
                                                  0,
                                                  Convert.ToUInt32(width * 1.5),
                                                  frameData);

                    ros.Publish(imageTopicName, image);

                    // Build and publish the headpose data message
                    HeadsetPoseDataMsg pose = new HeadsetPoseDataMsg(header, cameraToWorldMatrixAsFloat, projectionMatrixAsFloat);

                    ros.Publish(headsetPoseTopicName, pose);

                }

                originalSoftwareBitmap?.Dispose();

            }
        }
    }

	public bool HL2TryGetCameraToWorldMatrix(MediaFrameReference frameReference, out float[] outMatrix)
	{
        if (worldOrigin == null)
        {
            outMatrix = GetIdentityMatrixFloatArray();
            return false;
        }

        SpatialCoordinateSystem cameraCoordinateSystem = frameReference.CoordinateSystem;

        if (cameraCoordinateSystem == null)
        {
            outMatrix = GetIdentityMatrixFloatArray();
            return false;
        }

        System.Numerics.Matrix4x4? cameraCoordsToUnityCoordsMatrix = cameraCoordinateSystem.TryGetTransformTo(worldOrigin);

        if (cameraCoordsToUnityCoordsMatrix == null)
        {
            outMatrix = GetIdentityMatrixFloatArray();
            return false;
        }

        System.Numerics.Matrix4x4 cameraCoordsToUnityCoords = System.Numerics.Matrix4x4.Transpose(cameraCoordsToUnityCoordsMatrix.Value);

        // Change from right handed coordinate system to left handed UnityEngine
		cameraCoordsToUnityCoords.M31 *= -1f;
		cameraCoordsToUnityCoords.M32 *= -1f;
		cameraCoordsToUnityCoords.M33 *= -1f;
		cameraCoordsToUnityCoords.M34 *= -1f;

        outMatrix = ConvertMatrixToFloatArray(cameraCoordsToUnityCoords);

        return true;
	}

    private float[] ConvertMatrixToFloatArray(System.Numerics.Matrix4x4 matrix)
    {
        return new float[16] {
            matrix.M11, matrix.M12, matrix.M13, matrix.M14,
            matrix.M21, matrix.M22, matrix.M23, matrix.M24,
            matrix.M31, matrix.M32, matrix.M33, matrix.M34,
            matrix.M41, matrix.M42, matrix.M43, matrix.M44
        };
    }

    private float[] ConvertUnityMatrixToFloatArray(Matrix4x4 matrix)
    {
        return new float[16] {
            matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3],
            matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3],
            matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3],
            matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3]
        };
    }

    static float[] GetIdentityMatrixFloatArray()
	{
		return new float[] { 1f, 0, 0, 0, 0, 1f, 0, 0, 0, 0, 1f, 0, 0, 0, 0, 1f };
	}

	public static UnityEngine.Matrix4x4 ConvertFloatArrayToMatrix4x4(float[] matrixAsArray)
	{
		//There is probably a better way to be doing this but System.Numerics.Matrix4x4 is not available
		//in Unity and we do not include UnityEngine in the plugin.
		UnityEngine.Matrix4x4 m = new UnityEngine.Matrix4x4();
		m.m00 = matrixAsArray[0];
		m.m01 = matrixAsArray[1];
		m.m02 = -matrixAsArray[2];
		m.m03 = matrixAsArray[3];
		m.m10 = matrixAsArray[4];
		m.m11 = matrixAsArray[5];
		m.m12 = -matrixAsArray[6];
		m.m13 = matrixAsArray[7];
		m.m20 = matrixAsArray[8];
		m.m21 = matrixAsArray[9];
		m.m22 = -matrixAsArray[10];
		m.m23 = matrixAsArray[11];
		m.m30 = matrixAsArray[12];
		m.m31 = matrixAsArray[13];
		m.m32 = matrixAsArray[14];
		m.m33 = matrixAsArray[15];

		return m;
	}

    private System.Numerics.Matrix4x4 ConvertByteArrayToMatrix4x4(byte[] matrixAsBytes)
    {
        if (matrixAsBytes == null)
        {
            throw new ArgumentNullException("matrixAsBytes");
        }

        if (matrixAsBytes.Length != 64)
        {
            throw new Exception("Cannot convert byte[] to Matrix4x4. Size of array should be 64, but it is " + matrixAsBytes.Length);
        }

        var m = matrixAsBytes;
        return new System.Numerics.Matrix4x4(
            BitConverter.ToSingle(m, 0),
            BitConverter.ToSingle(m, 4),
            BitConverter.ToSingle(m, 8),
            BitConverter.ToSingle(m, 12),
            BitConverter.ToSingle(m, 16),
            BitConverter.ToSingle(m, 20),
            BitConverter.ToSingle(m, 24),
            BitConverter.ToSingle(m, 28),
            BitConverter.ToSingle(m, 32),
            BitConverter.ToSingle(m, 36),
            BitConverter.ToSingle(m, 40),
            BitConverter.ToSingle(m, 44),
            BitConverter.ToSingle(m, 48),
            BitConverter.ToSingle(m, 52),
            BitConverter.ToSingle(m, 56),
            BitConverter.ToSingle(m, 60)
        );
    }
#endif

}
