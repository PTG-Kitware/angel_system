using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;


public class HandCapture : MonoBehaviour, IMixedRealityHandJointHandler
{
    private Logger _logger = null;

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
        Logger log = logger();

        var observer = CoreServices.GetInputSystemDataProvider<IMixedRealityInputDeviceManager>();
        log.LogInfo("observer: " + observer.Name);
    }

    // Update is called once per frame
    void Update()
    {

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
        Logger log = logger();
        log.LogInfo("joint updated!" + eventData.Handedness.ToString());

        foreach (var item in eventData.InputData)
        {
            log.LogInfo("key: " + item.Key.ToString());
            log.LogInfo("value: " + item.Value.ToString());

        }


    }
}
