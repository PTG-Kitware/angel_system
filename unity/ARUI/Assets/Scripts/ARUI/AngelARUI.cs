using DilmerGames.Core.Singletons;
using UnityEngine;

public class AngelARUI : Singleton<AngelARUI>
{
    [HideInInspector]
    public Camera mainCamera;

    [Tooltip("If true, ARUI debug messages are shown in the unity console and scene Logger (if available)")]
    public bool showARUIDebugMessages = true;

    [Tooltip("If true, the eye gaze target is shown")]
    public bool showEyeGazeTarget = false;

    [Tooltip("Set a custom Skip Notification Message. Can not be empty.")]
    public string SkipNotificationMessage = "You are skipping the current task:";

    [Tooltip("Turn textToSpeech on or off")]
    public bool textToSpeechOn = false;

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
        TaskListManager.Instance.SetCurrentTask(0, textToSpeechOn);
    }

    /// <summary>
    /// Set the current task the user has to do.
    /// If taskID is >= 0 and < the number of tasks, then the task with the given taskID is highlighted. 
    /// </summary>
    /// <param name="taskID">index of the current task, in the task</param>
    public void SetCurrentTaskID(int taskID) => TaskListManager.Instance.SetCurrentTask(taskID, textToSpeechOn);

    /// <summary>
    /// Turn the task list on or off.
    /// </summary>
    public void SetTaskListActive(bool isActive) => TaskListManager.Instance.SetTaskListActive(isActive);

    /// <summary>
    /// Set all tasks in the task list as done. The orb will show a "done" message
    /// </summary>
    public void SetAllTasksDone() => TaskListManager.Instance.SetAllTasksDone(textToSpeechOn);

    /// <summary>
    /// Toggles the task list. If on, the task list is positioned in front of the user's current gaze.
    /// </summary>
    public void ToggleTasklist() => TaskListManager.Instance.ToggleTasklist();

    #endregion

    #region Notifications
    /// <summary>
    /// If given paramter is true, the orb will show message to the user that the system detected an attempt to skip the current task. 
    /// The message will disappear if "SetCurrentTaskID(..)" is called, or ShowSkipNotification(false)
    /// </summary>
    /// <param name="show">if true, the orb will show a skip notification, if false, the notification will disappear</param>
    public void ShowSkipNotification(bool show)
    {
        if (TaskListManager.Instance.GetTaskCount() <= 0 || TaskListManager.Instance.IsDone()) return;

        if (show)
        {
            if (SkipNotificationMessage==null || SkipNotificationMessage.Length==0)
                SkipNotificationMessage = "You are skipping the current task:";

            Orb.Instance.SetNotificationMessage(SkipNotificationMessage);
        }
        else
            Orb.Instance.SetNotificationMessage("");
    }

    #endregion

    /// <summary>
    /// ********FOR DEBUGGING ONLY, prints ARUI logging messages
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

}
