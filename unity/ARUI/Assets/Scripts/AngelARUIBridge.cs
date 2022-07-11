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

    // TODO: this should be replace with the results of the QueryTaskGraph call
    private string[,] tasks =
    {
        {"0", "Boil 12 ounces of water"},
        {"1", "Measure 12 ounces of water in the liquid measuring cup"},
        {"1", "Pour the water from the liquid measuring cup into  the electric kettle"},
        {"1", "Turn on the electric kettle by pushing the button underneath the handle"},
        {"1", "Boil the water. The water is done boiling when the button underneath the handle pops up"},

        {"0","Assemble the filter cone"},
        {"1", "While the water is boiling, assemble the filter cone. Place the dripper on top of a coffee mug"},
        {"1", "Prepare the filter insert by folding the paper filter in half to create a semi-circle, and in half again to create a quarter-circle. Place the paper filter in the dripper and spread open to create a cone."},
        {"1", "Take the coffee filter and fold it in half to create a semi-circle"},
        {"1", "Folder the filter in half again to create a quarter-circle"},
        {"1", "Place the folded filter into the dripper such that the the point of the quarter-circle rests in the center of the dripper"},
        {"1", "Spread the filter open to create a cone inside the dripper"},
        {"1", "Place the dripper on top of the mug"},

        {"0","Ground the coffee and add to the filter cone"},
        {"1","Weigh the coffee beans and grind until the coffee grounds are the consistency of coarse sand, about 20 seconds. Transfer the grounds to the filter cone."},
        {"1","Turn on the kitchen scale"},
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

    private void AruiUpdateCallback(AruiUpdateMsg msg)
    {
        Logger log = logger();
        log.LogInfo("Got update: " + msg.current_task_uid);

        // Call ARUI set task

        // TODO - need to map the UID to the step index
        //AngelARUI.Instance.SetCurrentTaskID();

    }
}
