using DilmerGames.Core.Singletons;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.Toolkit.Utilities.Solvers;
using System;

public class TaskListManager : Singleton<TaskListManager>
{
    private GameObject list;
    private GameObject bg;

    private GameObject taskPrefab;
    private GameObject subtaskPrefab;

    private float yOffset = 0.04f;
    private GameObject taskContainer;
    private List<TaskListElement> allTasks;

    private Follow followSolver;

    private bool isOpening = false;

    // Start is called before the first frame update
    void Start()
    {
        list = transform.GetChild(0).gameObject;
        bg = list.transform.GetChild(0).gameObject;

        taskPrefab = Resources.Load(StringResources.taskprefab_path) as GameObject;
        subtaskPrefab = Resources.Load(StringResources.subtaskprefab_path) as GameObject;

        taskContainer = GameObject.Find("TaskContainer");

        list.SetActive(false);

        followSolver = gameObject.GetComponent<Follow>();
        followSolver.MinDistance = 1.3f;
        followSolver.enabled = false;

        allTasks = new List<TaskListElement>();
    }


    public void SetAllTasks(string[,] tasklist)
    {
        //TODO: delete all and regenerate?

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

        float windowheight = yOffset * (allTasks.Count + 1);
        bg.transform.SetYPos(windowheight / 2 * -1);
        bg.transform.SetYScale(windowheight);

    }

    public void SetCurrentTask(int taskID)
    {
        Debug.Log("id was: " + taskID + " and list count: " + allTasks.Count);
        if (taskID <= 0)
        {
            taskID = 0;
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

        for (int i = taskID + 1; i < allTasks.Count; i++)
            allTasks[i].SetIsDone(false);
    }

    public void ToggleTaskList()
    {
        if (isOpening) return;
        Debug.Log("Show Task list: " + !list.activeSelf);

        if (!list.activeSelf)
        {
            isOpening = true;
            StartCoroutine(ShowTaskList());
        } else
        {
            followSolver.enabled = false;
            list.SetActive(false);
        }
    }

    private IEnumerator ShowTaskList()
    {
        followSolver.enabled = true;
        list.SetActive(true);
    
        yield return new WaitForSeconds(0.5f);

        followSolver.enabled = false;
        isOpening = false;
    }

}
