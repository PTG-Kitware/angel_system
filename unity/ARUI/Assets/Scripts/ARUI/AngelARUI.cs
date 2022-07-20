using DilmerGames.Core.Singletons;
using Microsoft.MixedReality.Toolkit.Examples.Demos.EyeTracking;
using Microsoft.MixedReality.Toolkit.Input;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum UpdateType
{
    add=0,
    change=1,
    remove=2,
}

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

    private void Awake()
    {
        mainCamera = Camera.main;

        //Instantiate database
        GameObject registry = new GameObject("EntityManager").AddComponent<EntityManager>().gameObject;
        registry.transform.parent = transform;

        //Instantiate audio manager
        new GameObject("AudioManager").AddComponent<AudioManager>();

        //Instantiate orb
        GameObject orb = Instantiate(Resources.Load(StringResources.orb_path)) as GameObject;
        orb.transform.parent = transform;
        orb.AddComponent<Orb>();

        //Instantiate empty tasklist
        GameObject taskListPrefab = Instantiate(Resources.Load(StringResources.taskList_path)) as GameObject;
        taskListPrefab.AddComponent<TaskListManager>();
    }

    private void Start()
    {
        if (Logger.Instance==null)
            PointerUtils.SetHandRayPointerBehavior(PointerBehavior.AlwaysOff);
    }

    /// <summary>
    /// Update the entity database according to the given parameters
    /// //STILL WORK IN PROGRESS
    /// </summary>
    /// <param name="id">unique identifier of the entity to be update/add/removed</param>
    /// <param name="update">type of update</param>
    /// <param name="position">the position in world space related to the entity</param>
    /// <param name="label">text label of updated entity</param>
    public void UpdateDatabase(string id, UpdateType update, Vector3 position, string label)
    {
        if (update.Equals(UpdateType.add)) {
            AddObject(id, position, label);

        } else if (update.Equals(UpdateType.remove)) {
            //TODO
        } else if (update.Equals(UpdateType.change)) {
            //TODO
        }
    }

    /// <summary>
    /// Set the task list and set the current task id to 0 (the first task)
    /// </summary>
    /// <param name="tasks">2D array off tasks</param>
    public void SetTasks(string[,] tasks)
    {
        TaskListManager.Instance.SetTasklist(tasks);
        TaskListManager.Instance.SetCurrentTask(0);

        Orb.Instance.SetMessageActive(true);
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

    /// <summary>
    /// Create Entity that was added to the database.
    /// </summary>
    /// <param name="id">unique id of the entity</param>
    /// <param name="worldPos">position in world space</param>
    /// <param name="text">text label</param>
    /// <returns>The reference to the created entity</returns>
    private DetectedEntity AddObject(string id, Vector3 worldPos, string text)
    {
        DetectedEntity DetectedObj = EntityManager.Instance.AddDetectedEntity(id,text);
        DetectedObj.transform.SetParent(EntityManager.Instance.transform);
        DetectedObj.InitEntity(id, worldPos, text, true);

        return DetectedObj;
    }

    public void PrintDebugMessage(string message, bool showInLogger)
    {
        if (showARUIDebugMessages)
        {
            if (showInLogger && FindObjectOfType<Logger>()!=null)
                Logger.Instance.LogInfo("***ARUI: "+ message);
            Debug.Log("***ARUI: " + message);
        }
    }

}
