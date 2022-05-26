using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;

using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.BuiltinInterfaces;
using RosMessageTypes.Std;
using RosMessageTypes.Angel;
using RosMessageTypes.Geometry;


public class HandCapture : MonoBehaviour, IMixedRealityHandJointHandler
{
    private Logger _logger = null;

    // Ros stuff
    ROSConnection ros;
    public string handJointPoseTopic = "HandJointPoseData";

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
        // Create the hand joint pose publisher
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<HandJointPosesUpdateMsg>(handJointPoseTopic);
    }

    // Register the hand joint event handler with the InputSystem
    protected void OnEnable()
    {
        CoreServices.InputSystem.RegisterHandler<IMixedRealityHandJointHandler>(this);
    }

    // Unregister the hand joint event handler with the InputSystem
    protected void OnDisable()
    {
        CoreServices.InputSystem.RegisterHandler<IMixedRealityHandJointHandler>(this);
    }

    // Callback for when updated hand joint information is received
    void IMixedRealityHandJointHandler.OnHandJointsUpdated(InputEventData<IDictionary<TrackedHandJoint, MixedRealityPose>> eventData)
    {
        List<HandJointPoseMsg> jointPoses = new List<HandJointPoseMsg>();
        foreach (var joint in eventData.InputData)
        {
            // Create the hand joint message for this joint
            HandJointPoseMsg jointPose = new HandJointPoseMsg(
                                            joint.Key.ToString(), // Joint name string
                                            new PoseMsg(
                                                new PointMsg( // Joint position
                                                    joint.Value.Position.x,
                                                    joint.Value.Position.y,
                                                    joint.Value.Position.z
                                                ),
                                                new QuaternionMsg( // Joint rotation
                                                    joint.Value.Rotation.x,
                                                    joint.Value.Rotation.y,
                                                    joint.Value.Rotation.z,
                                                    joint.Value.Rotation.w
                                                )
                                             )
                                          );
            jointPoses.Add(jointPose);
        }

        // Create the hand joints message containing all of the hand joint poses
        HeaderMsg header = PTGUtilities.getROSStdMsgsHeader("HandJointPosesUpdate");
        HandJointPosesUpdateMsg handJointsMsg = new HandJointPosesUpdateMsg(header,
                                                                            eventData.Handedness.ToString(),
                                                                            jointPoses.ToArray());
        ros.Publish(handJointPoseTopic, handJointsMsg);
    }
}
