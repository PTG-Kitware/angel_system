using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Angel;

/// <summary>
/// The bridge between the ANGEL system and the ANGEL ARUI.
/// Subscribes to AruiUpdate messages and calls the appropriate Angel ARUI
/// functions.
/// </summary>
public class AngelARUIBridge : MonoBehaviour
{
    // Ros stuff
    ROSConnection ros;
    public string aruiUpdateTopicName = "AruiUpdates";

    private Logger _logger = null;

    // TODO: this should be replaced with the results of the QueryTaskGraph call
    private string[,] tasks =
    {
        {"0", "Pour 12 ounces of water into liquid measuring cup"},
        {"0", "Pour the water from the liquid measuring cup into the electric kettle"},
        {"0", "Turn on the kettle"},
        {"0", "Place the dripper on top of the mug"},
        {"0", "Take the coffee filter and fold it in half to create a semi circle"},
        {"0", "Fold the filter in half again to create a quarter circle"},
        {"0", "Place the folded filter into the dripper such that the point of the quarter circle rests in the center of the dripper"},
        {"0", "Spread the filter open to create a cone inside the dripper"},
        {"0", "Turn on kitchen scale"},
        {"0", "Place bowl on scale"},
        {"0", "Zero scale"},
        {"0", "Add coffee beans until scale reads 25 grams"},
        {"0", "Pour coffee beans into coffee grinder"}
    };

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

    void Start()
    {
        // TODO: get the list of tasks from the QueryTaskGraph call
        AngelARUI.Instance.SetTasks(tasks);

        // Create the AruiUpdate subscriber
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<AruiUpdateMsg>(aruiUpdateTopicName, AruiUpdateCallback);
    }

    void Update()
    {

    }

    /// <summary>
    /// Callback function for the AruiUpdate subscriber.
    /// Updates the AngelARUI instance with the latest info.
    /// </summary>
    /// <param name="msg"></param>
    private void AruiUpdateCallback(AruiUpdateMsg msg)
    {
        Logger log = logger();

        AngelARUI.Instance.SetCurrentTaskID(msg.task_update.current_step_id);
    }
}
