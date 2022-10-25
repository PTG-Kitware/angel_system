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
    public string querytaskgraphTopicName = "query_task_graph";

    private Logger _logger = null;

    private bool taskGraphInitialized = false;
    private int loopIdx = 0;

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
        // Create the AruiUpdate subscriber
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<AruiUpdateMsg>(aruiUpdateTopicName, AruiUpdateCallback);

        // Register the QueryTaskGraph service
        ros.RegisterRosService<QueryTaskGraphRequest, QueryTaskGraphResponse>(querytaskgraphTopicName);
    }

    void Update()
    {
        // Check for a task graph every 5 seconds
        // TODO: probably a better way to do this
        if (taskGraphInitialized == false && (loopIdx % 300 == 0))
        {
            // Send message to ROS
            QueryTaskGraphRequest queryTaskGraphRequest = new QueryTaskGraphRequest();
            ros.SendServiceMessage<QueryTaskGraphResponse>(querytaskgraphTopicName, queryTaskGraphRequest, QueryTaskGraphCallback);
        }

        loopIdx++;
    }

    /// <summary>
    /// Callback function for the AruiUpdate subscriber.
    /// Updates the AngelARUI instance with the latest info.
    /// </summary>
    /// <param name="msg"></param>
    private void AruiUpdateCallback(AruiUpdateMsg msg)
    {
        // TaskUpdate message current step ID is the most recently completed
        // step ID, so set the ARUI current ID to one greater than that.
        AngelARUI.Instance.SetCurrentTaskID(msg.task_update.current_step_id + 1);
    }

    /// <summary>
    /// Callback function for the QueryTaskGraph service.
    /// Sets the ARUI task list with the task graph.
    /// </summary>
    /// <param name="msg"></param>
    void QueryTaskGraphCallback(QueryTaskGraphResponse msg)
    {
        Logger log = logger();

        string[,] final_tasks = new string[msg.task_graph.task_steps.Length, 2];

        for (int i = 0; i < msg.task_graph.task_steps.Length; i++)
        {
            final_tasks[i, 0] = msg.task_graph.task_levels[i].ToString();
            final_tasks[i, 1] = msg.task_graph.task_steps[i];
        }

        AngelARUI.Instance.SetTasks(final_tasks);

        taskGraphInitialized = true;
    }
}
