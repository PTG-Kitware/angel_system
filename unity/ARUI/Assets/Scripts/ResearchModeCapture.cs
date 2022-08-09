using Microsoft.MixedReality.Toolkit;
using System;
using System.Collections;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.BuiltinInterfaces;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;
using RosMessageTypes.Angel;

#if ENABLE_WINMD_SUPPORT
using HoloLens2ResearchMode;
using Windows.Perception;
using Windows.Perception.Spatial;
using Windows.Perception.Spatial.Preview;
#endif

public class ResearchModeCapture : MonoBehaviour
{
#if ENABLE_WINMD_SUPPORT
    private ResearchModeCameraSensor cameraSensor;
    private ResearchModeSensorDevice sensorDevice;
    private Task<ResearchModeSensorConsent> requestCameraAccessTask;
    private const ushort InvalidAhatValue = 4090;
    private SpatialLocator rigNodeLocator;

    private System.Numerics.Matrix4x4? invertedCameraExtrinsics = null;

    // Camera basis (x - right, y - down, z - forward) relative
    // to the HoloLens basis (x - right, y - up, z - back)
    private static readonly System.Numerics.Matrix4x4 CameraBasis = new System.Numerics.Matrix4x4(
        1,  0,  0,  0,
        0, -1,  0,  0,
        0,  0, -1,  0,
        0,  0,  0,  1
    );

#endif

    // Ros stuff
    ROSConnection ros;
    public string depthMapShortTopicName = "ShortThrowDepthMapImages";
    public string headsetDepthPoseTopicName = "HeadsetDepthPoseData";

    private Logger _logger = null;
    private string debugString = "";

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

    /// <summary>
    /// Initializes the depth camera sensor.
    /// </summary>
    void Awake()
    {
        Logger log = logger();

#if ENABLE_WINMD_SUPPORT
        this.sensorDevice = new ResearchModeSensorDevice();
        this.requestCameraAccessTask = this.sensorDevice.RequestCameraAccessAsync().AsTask();
        this.cameraSensor = (ResearchModeCameraSensor)this.sensorDevice.GetSensor(ResearchModeSensorType.DepthAhat);

        Guid rigNodeGuid = this.sensorDevice.GetRigNodeId();
        this.rigNodeLocator = SpatialGraphInteropPreview.CreateLocatorForNode(rigNodeGuid);
#endif
    }

    /// <summary>
    /// Checks that the depth camera was initialized properly,
    /// creates the ROS publisher for the depth camera,
    /// and starts the camera capturing thread.
    /// </summary>
    void Start()
    {
        Logger log = logger();

#if ENABLE_WINMD_SUPPORT
        var consent = this.requestCameraAccessTask.Result;
#endif

        // Create the image publisher and headset pose publisher
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<ImageMsg>(depthMapShortTopicName);
        ros.RegisterPublisher<HeadsetPoseDataMsg>(headsetDepthPoseTopicName);

        // Start the depth camera publisher thread
        Thread tDepthCameraPublisher = new Thread(DepthCameraThread);
        tDepthCameraPublisher.Start();
    }

    void Update()
    {
        if (debugString != "")
        {
            this.logger().LogInfo(debugString);
            debugString = "";
        }
    }

    /// <summary>
    /// Thread that continously extracts depth camera frames and publishes them
    /// as a ROS Image message.
    /// </summary>
    private void DepthCameraThread()
    {
#if ENABLE_WINMD_SUPPORT
        // Open the depth camera stream
        this.cameraSensor.OpenStream();

        while (true)
        {
            // Block until next frame is available
            var sensorFrame = this.cameraSensor.GetNextBuffer();

            // Extract frame metadata
            var frameTicks = sensorFrame.GetTimeStamp().HostTicks;
            var resolution = sensorFrame.GetResolution();
            int imageWidth = (int)resolution.Width;
            int imageHeight = (int)resolution.Height;

            var depthFrame = sensorFrame as ResearchModeSensorDepthFrame;
            UInt16[] depthBuffer = depthFrame.GetBuffer();
            byte[] depthBufferByteArray = new byte[depthBuffer.Length];

            // Check for invalid values and convert to byte values
            for (var i = 0; i < depthBuffer.Length; i++)
            {
                if (depthBuffer[i] > InvalidAhatValue)
                {
                    depthBufferByteArray[i] = 0;
                }
                else
                {
                    depthBufferByteArray[i] = (byte)((float)depthBuffer[i] / 1000 * 255);
                }
            }

            // Get the camera pose info
            var timestamp = PerceptionTimestampHelper.FromSystemRelativeTargetTime(TimeSpan.FromTicks((long)frameTicks));
            var rigNodeLocation = this.rigNodeLocator.TryLocateAtTimestamp(timestamp, SpatialMappingCapture.unityCoordinateSystem);

            // The rig node may not always be locatable, so we need a null check
            float[] cameraPose = null;
            if (rigNodeLocation != null)
            {
                // Compute the camera pose from the rig node location
                cameraPose = this.ToCameraPose(rigNodeLocation);
            }
            else
            {
                debugString += "rig location is null";
            }

            HeaderMsg header = PTGUtilities.getROSStdMsgsHeader("shortThrowDepthMap");
            ImageMsg depthImage = new ImageMsg(
                                      header,
                                      Convert.ToUInt32(imageHeight), // height
                                      Convert.ToUInt32(imageWidth), // width
                                      "mono8", // encoding
                                      0, // is_bigendian
                                      512, // step size (bytes)
                                      depthBufferByteArray
                                  );

            ros.Publish(depthMapShortTopicName, depthImage);

            // Build and publish the headpose data message
            // TODO: If we need the camera projection matrix, we'll have to fill that in here
            HeadsetPoseDataMsg pose = new HeadsetPoseDataMsg(header, cameraPose, new float[0]);
            ros.Publish(headsetDepthPoseTopicName, pose);
        } // end while loop
#endif
    }

#if ENABLE_WINMD_SUPPORT
    /// <summary>
    /// Converts the rig node location to the camera pose.
    /// Based on the Microsoft/psi repository:
    /// https://github.com/microsoft/psi/blob/master/Sources/MixedReality/Microsoft.Psi.MixedReality.UniversalWindows/ResearchModeCamera.cs
    /// </summary>
    /// <param name="rigNodeLocation">The rig node location.</param>
    /// <returns>The 4x4 matrix representing the camera pose as a float array.</returns>
    private float[] ToCameraPose(SpatialLocation rigNodeLocation)
    {
        var q = rigNodeLocation.Orientation;
        var m = System.Numerics.Matrix4x4.CreateFromQuaternion(q);
        var p = rigNodeLocation.Position;
        m.Translation = p;

        // Extrinsics of the camera relative to the rig node
        if (!this.invertedCameraExtrinsics.HasValue)
        {
            System.Numerics.Matrix4x4.Invert(this.cameraSensor.GetCameraExtrinsicsMatrix(), out var invertedMatrix);
            this.invertedCameraExtrinsics = invertedMatrix;
        }

        // Transform the rig node location to camera pose in world coordinates
        System.Numerics.Matrix4x4 cameraPose = CameraBasis * this.invertedCameraExtrinsics.Value * m;

        UnityEngine.Matrix4x4 mUnity = SystemNumericsExtensions.ToUnity(cameraPose);
        float[] mArray = PTGUtilities.ConvertUnityMatrixToFloatArray(mUnity);

        return mArray;
    }
#endif
}
