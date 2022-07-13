using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.BuiltinInterfaces;
using RosMessageTypes.Std;
using RosMessageTypes.Shape;
using RosMessageTypes.Geometry;
using RosMessageTypes.Angel;


/// <summary>
/// Class responsible for keeping track of the user's task progress and updating
/// the task AR display.
/// </summary>
public class TaskManager : MonoBehaviour
{
    // Ros stuff
    ROSConnection ros;
    public string taskUpdatesTopic = "TaskUpdates";

    private Logger _logger = null;
    private TaskLogger _taskLogger = null;

    private string _debugString = "";

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
    /// Lazy acquire the task logger object and return the reference to it.
    /// </summary>
    /// <returns>Logger instance reference.</returns>
    private ref TaskLogger taskLogger()
    {
        if (this._taskLogger == null)
        {
            this._taskLogger = GameObject.Find("TaskLogger").GetComponent<TaskLogger>();
        }
        return ref this._taskLogger;
    }

    // Start is called before the first frame update
    protected void Start()
    {
        Logger log = logger();
        TaskLogger taskLog = taskLogger();

        // Create placeholder task update message to display in the UI before
        // the first task update message is received
        TaskUpdateMsg taskUpdateMessage = new TaskUpdateMsg(new HeaderMsg(), // ROS std header
                                                           "Waiting for task", // task item
                                                           "Task description goes here.", // task description
                                                           new TaskItemMsg[0], // task items
                                                           new string[0], // task steps
                                                           0, // current step index
                                                           "N/A", // current step
                                                           "N/A", // previous step
                                                           "N/A", // current activity
                                                           "N/A", // next activity
                                                           -1 // time remaining
                                                           );

        log.LogInfo(taskUpdateMessage.task_name);
        taskLog.UpdateTaskDisplay(taskUpdateMessage);

        // Create the task update subscriber and register the callback
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<TaskUpdateMsg>(taskUpdatesTopic, TaskUpdateCallback);
    }

    // Update is called once per frame
    void Update()
    {
        if (_debugString != "")
        {
            this.logger().LogInfo(_debugString);
            _debugString = "";
        }
    }

    /// <summary>
    /// Callback function for the task updates topic subscription.
    /// Updates the task logger display when a new TaskUpdateMsg message is received.
    /// </summary>
    void TaskUpdateCallback(TaskUpdateMsg msg)
    {
        this.taskLogger().UpdateTaskDisplay(msg);
    }

}