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

#if ENABLE_WINMD_SUPPORT
using HoloLens2ResearchMode;
#endif

public class ResearchModeCapture : MonoBehaviour
{
#if ENABLE_WINMD_SUPPORT
    private ResearchModeCameraSensor cameraSensor;
    private ResearchModeSensorDevice sensorDevice;
    private Task<ResearchModeSensorConsent> requestCameraAccessTask;
    private const ushort InvalidAhatValue = 4090;
#endif

    // Ros stuff
    ROSConnection ros;
    public string depthMapShortTopicName = "ShortThrowDepthMapImages";

    // For filling in ROS message timestamp
    DateTime timeOrigin = new DateTime(1970, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc);

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

    void Awake()
    {
        Logger log = logger();

#if ENABLE_WINMD_SUPPORT
        this.sensorDevice = new ResearchModeSensorDevice();
        this.requestCameraAccessTask = this.sensorDevice.RequestCameraAccessAsync().AsTask();
        this.cameraSensor = (ResearchModeCameraSensor)this.sensorDevice.GetSensor(ResearchModeSensorType.DepthAhat);
#endif
    }

    // Start is called before the first frame update
    void Start()
    {
        Logger log = logger();

#if ENABLE_WINMD_SUPPORT
        var consent = this.requestCameraAccessTask.Result;
        log.LogInfo("access reqeust " + (consent == ResearchModeSensorConsent.Allowed).ToString());
#endif

        // Create the image publisher and headset pose publisher
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<ImageMsg>(depthMapShortTopicName);

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

            // Check for invalid values
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

            var currTime = DateTime.Now;
            TimeSpan diff = currTime.ToUniversalTime() - timeOrigin;
            var sec = Convert.ToInt32(Math.Floor(diff.TotalSeconds));
            var nsecRos = Convert.ToUInt32((diff.TotalSeconds - sec) * 1e9f);

            HeaderMsg header = new HeaderMsg(
                new TimeMsg(sec, nsecRos),
                "ShortThrowDepthMap"
            );
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
        } // end while loop
#endif
    } // end method


}
