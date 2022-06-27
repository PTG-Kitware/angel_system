using DilmerGames.Core.Singletons;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.Toolkit.Utilities.Solvers;
using System;
using HoloToolkit.Unity;
using Microsoft.MixedReality.Toolkit;

public class TaskListManager : Singleton<TaskListManager>
{
    //Reference to background panel, list container and taskprefabs
    private GameObject list;
    private GameObject taskContainer;
    private GameObject bg;
    private GameObject taskPrefab;
    private GameObject subtaskPrefab;

    //Height of single task entries in the list, based on the height of the prefabs
    private float yOffset = 0.025f;
    
    //List of all current taskelements
    private List<TaskListElement> allTasks;

    private bool isProcessing = false;

    private void Awake()
    {
        list = transform.GetChild(0).gameObject;
        bg = list.transform.GetChild(0).gameObject;

        taskPrefab = Resources.Load(StringResources.taskprefab_path) as GameObject;
        subtaskPrefab = Resources.Load(StringResources.subtaskprefab_path) as GameObject;

        taskContainer = GameObject.Find("TaskContainer");

        list.SetActive(false);

        allTasks = new List<TaskListElement>();

        gameObject.AddComponent<Billboard>();
    }

    /// <summary>
    /// Instantiate and initialize the task list and it's content
    /// </summary>
    /// <param name="tasklist"></param>
    public void InitTasklist(string[,] tasklist)
    {
        if (allTasks.Count > 0)
            FlushTaskList();

        for (int i = 0; i < tasklist.Length/2; i++)
        {
            GameObject task;
            if (tasklist[i, 0].Equals("0"))
                task = Instantiate(taskPrefab, taskContainer.transform);
            else
                task = Instantiate(subtaskPrefab, taskContainer.transform);

            task.transform.position = new Vector3(task.transform.position.x,
                    task.transform.position.y + (i * yOffset *-1), task.transform.position.z);

            TaskListElement t = task.gameObject.AddComponent<TaskListElement>();
            t.InitElement(tasklist[i, 1]);
            allTasks.Add(t);
        }

        float windowheight = yOffset * ((tasklist.Length+2)/2);
        taskContainer.transform.position = new Vector3(
            taskContainer.transform.position.x,
            bg.transform.position.y - (windowheight / 2f) * -1,
            taskContainer.transform.position.z);

        bg.transform.SetYScale(windowheight);   
    }

    private void FlushTaskList()
    {
        for (int i = 0; i < taskContainer.transform.childCount; i++)
            Destroy(taskContainer.transform.GetChild(i).gameObject);

        allTasks = new List<TaskListElement>();
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="taskID"></param>
    public void SetCurrentTask(int taskID)
    {
        if (taskID < 0)
        {
            for (int i = 0; i < taskID; i++)
                allTasks[i].SetAsCurrent();

        } else if (taskID == 0)
        {
            allTasks[taskID].SetAsCurrent();

        } else if (taskID >= allTasks.Count)
        {
            allTasks[allTasks.Count-1].SetIsDone(true);
        }
        else
        {
            for (int i = 0; i < taskID; i++)
                allTasks[i].SetIsDone(true);

            allTasks[taskID].SetAsCurrent();
        }

        if (taskID >= 0)
        {
            for (int i = taskID + 1; i < allTasks.Count; i++)
                allTasks[i].SetIsDone(false);
        }

    }

    public void ToggleTasklist() => SetTaskListActive(!list.activeInHierarchy);

    public void SetTaskListActive(bool isActive)
    {
        if (isProcessing) return;
        Debug.Log("Show Task list: " + isActive);

        if (isActive)
        {
            isProcessing = true;
            StartCoroutine(ShowTaskList());
        } else
        {
            list.SetActive(false);
            AudioManager.Instance.PlaySound(transform.position, SoundType.notification);
        }
    }

    private IEnumerator ShowTaskList()
    {
        Vector3 direction = AngelARUI.Instance.mainCamera.transform.forward;
        var eyeGazeProvider = CoreServices.InputSystem?.EyeGazeProvider;
        if (eyeGazeProvider != null && eyeGazeProvider.IsEyeTrackingEnabledAndValid && eyeGazeProvider.IsEyeCalibrationValid.Value)
        {
            direction = eyeGazeProvider.GazeDirection;
        }
                
        transform.position = AngelARUI.Instance.mainCamera.transform.position + Vector3.Scale(
            direction,
            new Vector3(1.1f, 1.1f, 1.1f));
        transform.SetYPos(AngelARUI.Instance.mainCamera.transform.position.y);

        yield return new WaitForEndOfFrame();

        list.SetActive(true);
        AudioManager.Instance.PlaySound(transform.position, SoundType.notification);

        isProcessing = false;
    }
}
