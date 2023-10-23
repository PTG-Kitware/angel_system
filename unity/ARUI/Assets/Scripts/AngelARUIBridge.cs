using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Angel;
using Newtonsoft.Json;

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
    public string confirmedUserIntentTopicName = "ConfirmedUserIntents";
    public string debugMsg = "";

    private Logger _logger = null;

    private bool taskGraphInitialized = false;
    private int loopIdx = 0;

    //private QueryTaskGraphResponse taskGraph = null;

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
        Logger log = logger();

        // Create the AruiUpdate subscriber
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<AruiUpdateMsg>(aruiUpdateTopicName, AruiUpdateCallback);

        // Register the QueryTaskGraph service
        ros.RegisterRosService<QueryTaskGraphRequest, QueryTaskGraphResponse>(querytaskgraphTopicName);
    }

    void Update()
    {
        Logger log = logger();

        // Check for a task graph every 5 seconds
        // TODO: probably a better way to do this
        if (taskGraphInitialized == false && (loopIdx % 300 == 0))
        {
            // Send message to ROS
            QueryTaskGraphRequest queryTaskGraphRequest = new QueryTaskGraphRequest();
            ros.SendServiceMessage<QueryTaskGraphResponse>(querytaskgraphTopicName, queryTaskGraphRequest, QueryTaskGraphCallback);
        }

        if (debugMsg != "")
        {
            log.LogInfo(debugMsg);
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
        try
        {
            AngelARUI.Instance.SetCurrentObservedTask(msg.task_update.task_name);
            AngelARUI.Instance.GoToStep(msg.task_update.task_name, msg.task_update.current_step_id);
        }
        catch (Exception e)
        {
            debugMsg += e.ToString();
        }
    }

    /// <summary>
    /// Callback function for the QueryTaskGraph service.
    /// Sets the ARUI task list with the task graph.
    /// </summary>
    /// <param name="msg"></param>
    void QueryTaskGraphCallback(QueryTaskGraphResponse msg)
    {
        // Create dictionary of TaskName:TaskStepList
        Dictionary<string, string> tasks = new Dictionary<string, string>();

        for (int i = 0; i < msg.task_titles.Length; i++)
        {
            Dictionary<string, object> taskDict = new Dictionary<string, object>();

            taskDict.Add("Name", msg.task_titles[i]);

            // Create list of steps dicts
            List<Dictionary<string, object>> taskSteps = new List<Dictionary<string, object>>();
            for (int j = 0; j < msg.task_graphs[i].task_steps.Length; j++)
            {
                Dictionary<string, object> taskStep = new Dictionary<string, object>
                {
                    { "StepDesc", msg.task_graphs[i].task_steps[j] },
                    { "RequiredItems", Array.Empty<string>() },
                    { "SubSteps", Array.Empty<string>() },
                    { "CurrSubStepIndex", -1 },
                };
                taskSteps.Add(taskStep);
            }
            taskDict.Add("Steps", taskSteps);
            taskDict.Add("CurrStepIndex", 0);
            taskDict.Add("PrevStepIndex", -1);
            taskDict.Add("NextStepIndex", 1);

            string taskDictStr = JsonConvert.SerializeObject(taskDict);
            tasks.Add(msg.task_titles[i], taskDictStr);
        }

        /*
        string path = Path.Combine(Application.persistentDataPath, "MyFile.txt");
        using (TextWriter writer = File.CreateText(path))
        {
            foreach (KeyValuePair<string, string> entry in tasks)
            {
                writer.WriteLine(entry.Key + ": " + entry.Value);
            }
        }
        */

        AngelARUI.Instance.InitManual(tasks);
        taskGraphInitialized = true;
    }
}
