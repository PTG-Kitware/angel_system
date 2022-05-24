using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.BuiltinInterfaces;
using RosMessageTypes.Std;
using RosMessageTypes.Geometry;
using RosMessageTypes.Angel;


public class EyeGazeCapture : MonoBehaviour
{
    private Logger _logger = null;

    // For filling in ROS message timestamp
    DateTime timeOrigin = new DateTime(1970, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc);

    // Ros stuff
    ROSConnection ros;
    public string eyeGazeTopicName = "EyeGazeData";

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
    void Start()
    {
        // Create the eye gaze publisher
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<EyeGazeDataMsg>(eyeGazeTopicName);
    }

    // Update is called once per frame
    void Update()
    {
        Logger log = logger();

        var eyeGazeProvider = CoreServices.InputSystem?.EyeGazeProvider;
        if (eyeGazeProvider != null)
        {
            if (eyeGazeProvider.IsEyeTrackingEnabledAndValid)
            {
                // Get the latest eye tracking data and form it into a ROS message
                PointMsg gazeOrigin = new PointMsg(eyeGazeProvider.GazeOrigin.x,
                                                   eyeGazeProvider.GazeOrigin.y,
                                                   eyeGazeProvider.GazeOrigin.z);
                Vector3Msg gazeDirection = new Vector3Msg(eyeGazeProvider.GazeDirection.x,
                                                          eyeGazeProvider.GazeDirection.y,
                                                          eyeGazeProvider.GazeDirection.z);
                Vector3Msg headMovementDirection = new Vector3Msg(eyeGazeProvider.HeadMovementDirection.x,
                                                                  eyeGazeProvider.HeadMovementDirection.y,
                                                                  eyeGazeProvider.HeadMovementDirection.z);
                Vector3Msg headVelocity = new Vector3Msg(eyeGazeProvider.HeadVelocity.x,
                                                         eyeGazeProvider.HeadVelocity.y,
                                                         eyeGazeProvider.HeadVelocity.z);

                // Create the ROS std_msgs header with the EyeGazeProvider's timestamps
                var currTime = eyeGazeProvider.Timestamp;
                TimeSpan diff = currTime.ToUniversalTime() - timeOrigin;
                var sec = Convert.ToInt32(Math.Floor(diff.TotalSeconds));
                var nsecRos = Convert.ToUInt32((diff.TotalSeconds - sec) * 1e9f);

                HeaderMsg header = new HeaderMsg(
                    new TimeMsg(sec, nsecRos),
                    "EyeGazeData"
                );

                EyeGazeDataMsg eyeGazeDataMsg = new EyeGazeDataMsg(header, gazeOrigin, gazeDirection,
                                                                   headMovementDirection, headVelocity);

                ros.Publish(eyeGazeTopicName, eyeGazeDataMsg);

                log.LogInfo("hit valid: " + eyeGazeProvider.HitInfo.raycastValid.ToString());
                log.LogInfo("hit point: " + eyeGazeProvider.HitInfo.point.ToString());

            }
        }
    }
}
