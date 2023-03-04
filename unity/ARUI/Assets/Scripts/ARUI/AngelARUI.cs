using DilmerGames.Core.Singletons;
using UnityEngine;
using UnityEngine.Events;

public class AngelARUI : Singleton<AngelARUI>
{
    private Camera _arCamera;
    public Camera ARCamera
    {
        get { return _arCamera; }
    }

    ///****** Debug Settings
    private bool showARUIDebugMessages = true; /// <If true, ARUI debug messages are shown in the unity console and scene Logger (if available)
    private bool showEyeGazeTarget = true; /// <If true, the eye gaze target is shown if the eye ray hits UI elements (white small cube)

    ///****** Guidance Settings
    private bool useViewManagement = true; /// <If true, the ARUI avoids placing UI eleements in front of salient regions 
    public bool IsVMActiv
    {
        get { return useViewManagement; }
    }

    [Tooltip("Set a custom Skip Notification Message. Can not be empty.")]
    public string SkipNotificationMessage = "You are skipping the current task:";

    ///****** Confirmation Dialogue 
    private UnityAction onUserIntentConfirmedAction = null;     /// <Action invoked if the user accepts the confirmation dialogue
    private ConfirmationDialogue confirmationWindow = null;     /// <Reference to confirmation dialogue
    private GameObject confirmationWindowPrefab = null;

    private void Awake()
    {
        //Get persistant reference to ar cam
        _arCamera = Camera.main;

        //Instantiate audio manager, for audio feedback
        new GameObject("AudioManager").AddComponent<AudioManager>();

        //Instantiate eye target wrapper
        GameObject eyeTarget = Instantiate(Resources.Load(StringResources.eyeTarget_path)) as GameObject;
        eyeTarget.AddComponent<FollowEyeTarget>();
        FollowEyeTarget.Instance.ShowDebugTarget(showEyeGazeTarget);

        //Instantiate the main system menu - the orb
        GameObject orb = Instantiate(Resources.Load(StringResources.orb_path)) as GameObject;
        orb.transform.parent = transform;
        orb.AddComponent<Orb>();

        //Start View Management, if enabled
        if (useViewManagement)
            ARCamera.gameObject.AddComponent<ViewManagement>();

        //Instantiate empty tasklist
        GameObject taskListPrefab = Instantiate(Resources.Load(StringResources.taskList_path)) as GameObject;
        taskListPrefab.AddComponent<TaskListManager>();

        //Load resources for UI elements
        confirmationWindowPrefab = Resources.Load(StringResources.confNotification_path) as GameObject;

    }

    #region Task Guidance
    /// <summary>
    /// Set the task list and set the current task id to 0 (first in the given list)
    /// </summary>
    /// <param name="tasks">2D array tasks</param>
    public void SetTasks(string[,] tasks)
    {
        TaskListManager.Instance.SetTasklist(tasks);
        TaskListManager.Instance.SetCurrentTask(0);
    }

    /// <summary>
    /// Set the current task the user has to do.
    /// If taskID is >= 0 and < the number of tasks, the orb won't react. 
    /// If taskID is the same as the current one, the ARUI won't react.
    /// If taskID has subtasks, the orb shows the first subtask as the current task
    /// </summary>
    /// <param name="taskID">index of the current task that should be highlighted in the UI</param>
    public void SetCurrentTaskID(int taskID) => TaskListManager.Instance.SetCurrentTask(taskID);

    /// <summary>
    /// Enable/Disable Tasklist
    /// </summary>
    public void SetTaskListActive(bool isActive) => TaskListManager.Instance.SetTaskListActive(isActive);

    /// <summary>
    /// Set all tasks in the tasklist as done. The orb will show a "All Done" message
    /// </summary>
    public void SetAllTasksDone() => TaskListManager.Instance.SetAllTasksDone();

    /// <summary>
    /// Toggles the task list. If on, the task list is positioned in front of the user's current gaze.
    /// </summary>
    public void ToggleTasklist() => TaskListManager.Instance.ToggleTasklist();

    /// <summary>
    /// Mute voice feedback for task guidance. ONLY influences task guidance. 
    /// </summary>
    /// <param name="mute">if true, the user will hear the tasks, in addition to text.</param>
    public void MuteAudio(bool mute) => AudioManager.Instance.MuteAudio(mute);

    /// <summary>
    /// Enable or disable view management. enabled by default 
    /// </summary>
    /// <param name="enabled"></param>
    public void SetViewManagement(bool enabled)
    {
        if (useViewManagement != enabled)
        {
            if (enabled)
            {
                ARCamera.gameObject.AddComponent<ViewManagement>();
                useViewManagement = enabled;
            }
            else if (ARCamera.GetComponent<ViewManagement>() != null)
            {
                Destroy(ARCamera.GetComponent<ViewManagement>());
                useViewManagement = enabled;
            }
        }
    }

    #endregion

    #region Notifications
    /// <summary>
    /// Set the callback function that is invoked if the user confirms the confirmation dialogue
    /// </summary>
    public void SetUserIntentCallback(UnityAction userIntentCallBack) => onUserIntentConfirmedAction = userIntentCallBack;

    /// <summary>
    /// If confirmation action is set - SetUserIntentCallback(...) - and no confirmation window is active at the moment, the user is shown a 
    /// timed confirmation window. Recommended text: "Did you mean ...". If the user confirms the dialogue, the onUserIntentConfirmedAction action is invoked. 
    /// </summary>
    /// <param name="msg">message that is shown in the confirmation dialogue</param>
    public void TryGetUserFeedbackOnUserIntent(string msg)
    {
        if (onUserIntentConfirmedAction == null || confirmationWindow != null) return;

        GameObject window = Instantiate(confirmationWindowPrefab, transform);
        confirmationWindow = window.AddComponent<ConfirmationDialogue>();
        confirmationWindow.InitializeConfirmationNotification(msg, onUserIntentConfirmedAction);
    }

    /// <summary>
    /// If given paramter is true, the orb will show message to the user that the system detected an attempt to skip the current task. 
    /// The message will disappear if "SetCurrentTaskID(..)" is called, or ShowSkipNotification(false)
    /// </summary>
    /// <param name="show">if true, the orb will show a skip notification, if false, the notification will disappear</param>
    public void ShowSkipNotification(bool show)
    {
        if (TaskListManager.Instance.GetTaskCount() <= 0 || TaskListManager.Instance.IsDone) return;

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

    #region Logging
    /// <summary>
    /// Set if debug information is shown in the logger window
    /// </summary>
    /// <param name="show">if true, ARUI debug messages are shown in the unity console and scene Logger (if available)</param>
    public void ShowDebugMessagesInLogger(bool show) => showARUIDebugMessages = show;

    /// <summary>
    /// Set if debug information is shown about the users eye gaze
    /// </summary>
    /// <param name="show">if true and the user is looking at a virtual UI element, a debug cube that represents the eye target is shown</param>
    public void ShowDebugEyeGazeTarget(bool show)
    {
        showEyeGazeTarget = show;
        FollowEyeTarget.Instance.ShowDebugTarget(showEyeGazeTarget);
    }

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

    #endregion
}
