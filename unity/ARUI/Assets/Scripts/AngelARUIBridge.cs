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
    public string systemCommandName = "SystemCommands";
    public string debugMsg = "";

    private bool taskGraphInitialized = false;
    private int loopIdx = 0;

    private bool _showLogger = false;

    void Start()
    {
        // Create the AruiUpdate subscriber
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<AruiUpdateMsg>(aruiUpdateTopicName, AruiUpdateCallback);

        // Register the QueryTaskGraph service
        ros.RegisterRosService<QueryTaskGraphRequest, QueryTaskGraphResponse>(querytaskgraphTopicName);

        // Register the QueryTaskGraph service
        ros.RegisterPublisher<SystemCommandsMsg>(systemCommandName);

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
        // Update task status
        AngelARUI.Instance.SetCurrentObservedTask(msg.task_update.task_name);

        // NOTE: There is a current mismatch between the task lists used by the ARUI
        // and the task monitor. The ARUI does not have a concept of a background step
        // at step 0, so step_id = 0 is not the same between the task monitor and the ARUI.
        // Hence, the logic here to set the ARUI to go to step_id + 1.
        if (msg.task_update.current_step_id == 0 && (msg.task_update.current_step == "background"))
        {
            // Handle special case going back to background
            AngelARUI.Instance.GoToStep(msg.task_update.task_name, msg.task_update.current_step_id);
        }
        else
        {
            AngelARUI.Instance.GoToStep(msg.task_update.task_name, msg.task_update.current_step_id + 1);
        }

        for (int i = 0; i < msg.notifications.Length; i++)
        {
            if (msg.notifications[i].context.Equals(AruiUserNotificationMsg.N_CONTEXT_TASK_ERROR)) {
                AngelARUI.Instance.DebugLogMessage("Show skipped step dialogue to user", true);
                AngelARUI.Instance.TryGetUserConfirmation("We noticed you skipped a step, do you want to go back?",
                    () => { SendGoToPrevious(); },
                    null, 20, true);
            } else if (msg.notifications[i].context.Equals(AruiUserNotificationMsg.N_CONTEXT_USER_MODELING))
            {
                AngelARUI.Instance.DebugLogMessage("Show skipped step dialogue to user", true);
                if (msg.notifications[i].title.Length==0 && msg.notifications[i].description.ToLower().Contains("thinking"))
                    AngelARUI.Instance.SetAgentThinking(true);
                else
                {
                    AngelARUI.Instance.SetAgentThinking(false);
                    AngelARUI.Instance.PlayDialogueAtAgent(msg.notifications[i].title, msg.notifications[i].description);
                }
            }
        }
        
    }
    private void SendGoToPrevious()
    {
        SystemCommandsMsg msg = new SystemCommandsMsg();
        msg.previous_step = true;
        ros.Publish(systemCommandName, msg);
        AngelARUI.Instance.DebugLogMessage("Sending message to backend to go to previous step.", true);
    }

    private void SendGoToNext()
    {
        SystemCommandsMsg msg = new SystemCommandsMsg();
        msg.next_step = true;
        ros.Publish(systemCommandName, msg);
        AngelARUI.Instance.DebugLogMessage("Sending message to backend to go to next step.", true);
    }
    private void SendRestart()
    {
        SystemCommandsMsg msg = new SystemCommandsMsg();
        msg.task_index = 1;
        ros.Publish(systemCommandName, msg);
        AngelARUI.Instance.DebugLogMessage("Sending message to go to the first step", true);
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

        AngelARUI.Instance.InitManual(tasks);
        taskGraphInitialized = true;

        AngelARUI.Instance.RegisterKeyword("previous step", () => { SendGoToPrevious(); });
        AngelARUI.Instance.RegisterKeyword("next step", () => { SendGoToNext(); });

        AngelARUI.Instance.RegisterKeyword("restart", () => { SendRestart(); });

        AngelARUI.Instance.RegisterKeyword("toggle debug", () => { AngelARUI.Instance.SetLoggerVisible(!Logger.Instance.IsVisible); });
        AngelARUI.Instance.RegisterKeyword("angel", () => { AngelARUI.Instance.CallAgentToUser(); });
    }
}
