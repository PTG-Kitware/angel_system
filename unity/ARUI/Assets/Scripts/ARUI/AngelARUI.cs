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
    private int currentTask = 0;

    public Camera mainCamera;

    private AudioSource taskListsound;

    public void Start()
    {
        mainCamera = Camera.main;

        //Init database
        GameObject registry = new GameObject("EntityManager").AddComponent<EntityManager>().gameObject;
        registry.transform.parent = transform;

        //Instantiate audio
        new GameObject("AudioManager").AddComponent<AudioManager>();

        //Instantiate orb
        GameObject orb = Instantiate(Resources.Load(StringResources.orb_path)) as GameObject;
        orb.AddComponent<Orb>();
        orb.transform.parent = transform;

        //Instantiate tasklist
        GameObject taskListPrefab = Instantiate(Resources.Load(StringResources.taskList_path)) as GameObject;
        taskListPrefab.AddComponent<TaskListManager>();
    }

    public void UpdateDatabase(string id, UpdateType update, Vector3 position, string label)
    {
        if (update.Equals(UpdateType.add)) {
            AddObject(position, label, false);

        } else if (update.Equals(UpdateType.remove)) {
            //TODO
        } else if (update.Equals(UpdateType.change)) {
            //TODO
        }
    }

    public void SetExpertiseLevel(int level)
    {
        //TODO
    }

    public void SetTasks(string[,] tasks)
    {
        this.tasks = tasks;
        TaskListManager.Instance.InitTasklist(tasks);
    }

    public void SetCurrentTaskID(int taskID)
    {
        currentTask = taskID;
        TaskListManager.Instance.SetCurrentTask(taskID);
    }

    public void ToggleTasklist()
    {
        TaskListManager.Instance.ToggleTaskList();
        taskListsound.Play();
        
    }

    /// <summary>
    /// Message communicating a single notification for the ARUI to present to the user.
    /// </summary>
    /// <param name="cat">Notification category.</param>
    /// <param name="context">Different notifications can have different contexts from which they are emitted that the UI will want to handle differently.</param>
    /// <param name="title">A short message about this notification.</param>
    /// <param name="description">Potentially longer description of this notification.</param>
    /// <param name="objIDs">0 or more  3D objects this notification is associated with, by UID.</param>
    /// <param name="bb">0 or more spatial polygons this notification is associated with.</param>
    public void SendARMessage(msgCat cat, msgContext context, string title, string description, string[] objIDs, List<Vector3[]> bb)
    {
        //TODO
    }

    public void SetDialog(string message, AudioClip speech)
    {
        //TODO
    }

    public void GuideTheUserTo(string id, bool isOn, bool flat)
    {
        ((DetectedEntity)EntityManager.Instance.Get(name)).SetHaloOn(isOn, flat);
    }

    private DetectedEntity AddObject(Vector3 worldPos, string text, bool showDetectedObj)
    {
        DetectedEntity DetectedObj = EntityManager.Instance.AddDetectedEntity(text);
        DetectedObj.transform.SetParent(EntityManager.Instance.transform);
        DetectedObj.InitEntity(text, worldPos, text, showDetectedObj);

        Logger.Instance.LogInfo("Placed " + text + " at " + worldPos);
        return DetectedObj;
    }

}
