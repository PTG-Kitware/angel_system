using DilmerGames.Core.Singletons;
using System;
using UnityEngine;

public enum msgCat
{ //Basically the priority in which the notification should be treated.
    N_CAT_DANGER = 0,
    N_CAT_WARNING = 1,
    N_CAT_CAUTION = 2,
    N_CAT_NOTICE = 3,
}

public enum msgContext
{
    N_CONTEXT_TASK_ERROR = 0, //# There is some error the user performed in the task
    N_CONTEXT_ENV_ATTENTION = 1, //# There is something in the environment this notification pertains to. Likely spatial in nature.
    N_CONTEXT_USER_MODELING = 2 //# This notification is in regards to the user modeling (e.g. user frustration).
}

public class AngelARUI : Singleton<AngelARUI>
{    
    [HideInInspector]
    public Camera mainCamera;

    // If true, ARUI debug messages are shown in the unity console and scene Logger (if available)
    public bool showARUIDebugMessages = true;
    public bool showEyeGazeTarget = false;

    private void Awake()
    {
        //Get persistant reference to ar cam
        mainCamera = Camera.main;

        //Instantiate audio manager
        new GameObject("AudioManager").AddComponent<AudioManager>();

        GameObject eyeTarget = Instantiate(Resources.Load(StringResources.eyeTarget_path)) as GameObject;
        FollowEyeTarget follow = eyeTarget.AddComponent<FollowEyeTarget>();
        FollowEyeTarget.Instance.ShowDebugTarget(showEyeGazeTarget);

        //Instantiate orb
        GameObject orb = Instantiate(Resources.Load(StringResources.orb_path)) as GameObject;
        orb.transform.parent = transform;
        orb.AddComponent<Orb>();

        //Instantiate empty tasklist
        GameObject taskListPrefab = Instantiate(Resources.Load(StringResources.taskList_path)) as GameObject;
        taskListPrefab.AddComponent<TaskListManager>();
    }

    #region Tasks
    /// <summary>
    /// Set the task list and set the current task id to 0 (the first task)
    /// </summary>
    /// <param name="tasks">2D array off tasks</param>
    public void SetTasks(string[,] tasks)
    {
        TaskListManager.Instance.SetTasklist(tasks);
        TaskListManager.Instance.SetCurrentTask(0);
    }

    /// <summary>
    /// Set the current task the user has to do.
    /// If taskID is >= 0 and < the number of tasks, then the task with the given taskID is highlighted. 
    /// </summary>
    /// <param name="taskID">index of the current task, in the task</param>
    public void SetCurrentTaskID(int taskID) => TaskListManager.Instance.SetCurrentTask(taskID);

    /// <summary>
    /// Turn the task list on or off.
    /// </summary>
    public void SetTaskListActive(bool isActive) => TaskListManager.Instance.SetTaskListActive(isActive);

    /// <summary>
    /// Toggles the task list. If on, the task list is positioned in front of the user's current gaze.
    /// </summary>
    public void ToggleTasklist() => TaskListManager.Instance.ToggleTasklist();

    #endregion


    /// <summary>
    /// For debugging, print ARUI logging messages
    /// </summary>
    /// <param name="message"></param>
    /// <param name="showInLogger"></param>
    public void LogDebugMessage(string message, bool showInLogger)
    {
        if (showARUIDebugMessages)
        {
            if (showInLogger && FindObjectOfType<Logger>() != null)
                Logger.Instance.LogInfo("***ARUI: " + message);
            Debug.Log("***ARUI: " + message);
        }
    }

    public void SetAllTasksDone() => TaskListManager.Instance.SetAllTasksDone();
}
