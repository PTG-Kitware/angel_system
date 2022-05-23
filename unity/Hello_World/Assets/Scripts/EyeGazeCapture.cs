using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;

using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.BuiltinInterfaces;
using RosMessageTypes.Std;
using RosMessageTypes.Sensor;
using RosMessageTypes.Angel;


public class EyeGazeCapture : MonoBehaviour
{
    private float defaultDistanceInMeters = 2f;

    private Logger _logger = null;

    // Ros stuff
    ROSConnection ros;
    public string eyeGazeTopic = "EyeGazeData";

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
        //ros = ROSConnection.GetOrCreateInstance();
        //ros.RegisterPublisher<HandJointPosesUpdateMsg>(eyeGazeTopic);
    }

    // Update is called once per frame
    void Update()
    {
        Logger log = logger();

        var eyeGazeProvider = CoreServices.InputSystem?.EyeGazeProvider;
        if (eyeGazeProvider != null)
        {
            //log.LogInfo("Eye valid: " + eyeGazeProvider.IsEyeTrackingEnabledAndValid.ToString());
            //log.LogInfo("Eye data valid: " + eyeGazeProvider.IsEyeTrackingDataValid.ToString());
            //log.LogInfo("Eye tracking enabled: " + eyeGazeProvider.IsEyeTrackingEnabled.ToString());
            //log.LogInfo("Eye calibration valid: " + eyeGazeProvider.IsEyeCalibrationValid.ToString());
            //log.LogInfo("Eye enabled: " + eyeGazeProvider.Enabled.ToString());
            log.LogInfo("Head movement dir: " + eyeGazeProvider.HeadMovementDirection.ToString());
            //log.LogInfo("Head velocity: " + eyeGazeProvider.HeadVelocity.ToString());
            //log.LogInfo("Eye origin: " + eyeGazeProvider.GazeOrigin.ToString());
            log.LogInfo("Eye direction: " + eyeGazeProvider.GazeDirection.ToString());
        }
    }
}
