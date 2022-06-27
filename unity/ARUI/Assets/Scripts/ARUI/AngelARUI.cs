using DilmerGames.Core.Singletons;
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
    private string[,] tasks;

    [HideInInspector]
    public Camera mainCamera;

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
    /// Set the task list and set the current task id to the given value
    /// </summary>
    /// <param name="tasks">2D array off tasks</param>
    /// <param name="id">id of current task id</param>
    public void SetTasks(string[,] tasks, int id)
    {
        this.tasks = tasks;
        TaskListManager.Instance.InitTasklist(tasks);
        SetCurrentTaskID(id);
    }

    /// <summary>
    /// Set the current task the user has to do.
    /// If taskID is < than 0, the entire task list is shown and every task is set as NOT done.
    /// If taskID is >= 0 and < the number of tasks, then the task with the given taskID is highlighted. 
    /// If taskID is > than the number of tasks, the entire task list is set as DONE.
    /// </summary>
    /// <param name="taskID">index of the current task, in the task</param>
    public void SetCurrentTaskID(int taskID)
    {
        if (tasks == null)
        {
            Debug.Log("Trying to set 'currentTaskID' : " + taskID + " failed. Tasklist was: " + tasks);
            return;
        }

        TaskListManager.Instance.SetCurrentTask(taskID);

        if (taskID < tasks.GetLength(0) && taskID >= 0)
        {
            Orb.Instance.SetMessage(tasks[taskID, 1]);
            Debug.Log("Set current task ID to: " + taskID);
        } else
        {
            Orb.Instance.SetMessage("");
            Debug.Log("Set current task ID to end.");
        }
    }

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
        DetectedEntity DetectedObj = EntityManager.Instance.AddDetectedEntity(text);
        DetectedObj.transform.SetParent(EntityManager.Instance.transform);
        DetectedObj.InitEntity(id, worldPos, text, true);

        Logger.Instance.LogInfo("Placed " + text + " at " + worldPos);
        return DetectedObj;
    }

}
