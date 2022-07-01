using DilmerGames.Core.Singletons;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.Toolkit.Utilities.Solvers;
using System;
using HoloToolkit.Unity;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using UnityEngine.UI;

public class TaskListManager : Singleton<TaskListManager>
{
    private string[,] tasks;
    private Dictionary<int, int> taskToParent;
    private Dictionary<int, TaskListElement> taskToElement;
    private bool taskListGenerated = false;
    private List<TaskListElement> currentTaskIDOnList;
    private Dictionary<int, List<TaskListElement>> mainToSubTasks;

    private bool isProcessingOpening = false;

    //Reference to background panel, list container and taskprefabs
    private GameObject list;
    private RectTransform taskContainer;
    private GameObject taskPrefab;

    /// Must be >=1 and an odd number
    private int numTasks = 7;

    private Shapes.Line progressLine;
    private GameObject topPointsParent;
    private GameObject bottomPointsParent;

    // Eye-gaze based updates
    private bool isCurrentlyLooking = false;
    private bool isVisible = false;
    private bool isFading = false;
    private Material bgMat;
    private EyeTrackingTarget listEyeTarget;
    private Color activeColor = new Color(0.06f, 0.06f, 0.06f, 0.5f);
    private float step = 0.005f;

    private void Awake()
    {
        list = transform.GetChild(0).gameObject;

        taskContainer = GameObject.Find("TaskContainer").GetComponent<RectTransform>();
        bgMat = taskContainer.GetComponent<Image>().material;
        taskPrefab = Resources.Load(StringResources.taskprefab_path) as GameObject;

        listEyeTarget = GetComponentInChildren<EyeTrackingTarget>();
        listEyeTarget.OnLookAtStart.AddListener(delegate { IsLookingAt(true); });
        listEyeTarget.OnLookAway.AddListener(delegate { IsLookingAt(false); });

        progressLine = GetComponentInChildren<Shapes.Line>();
        bottomPointsParent = GameObject.Find("BottomPoints");
        bottomPointsParent.transform.parent = progressLine.transform;
        topPointsParent = GameObject.Find("TopPoints");
        topPointsParent.transform.parent = progressLine.transform;
        bottomPointsParent.SetActive(false);
        topPointsParent.SetActive(false);

        list.SetActive(false);

        gameObject.AddComponent<Billboard>();
    }

    public bool IsTaskListActive() => list.activeInHierarchy;

    private void Update()
    {
        if (!taskListGenerated) return;
        
        if (isCurrentlyLooking && !isVisible && !isFading)
        {
            bgMat.color = activeColor;
            for (int i = 0; i < currentTaskIDOnList.Count; i++)
                currentTaskIDOnList[i].SetAlpha(1f);
            isVisible = true;
        }
        else if (isCurrentlyLooking && isVisible && isFading)
        {
            StopCoroutine(FadeOut());

            isFading = false;
            bgMat.color = activeColor;
            for (int i = 0; i < currentTaskIDOnList.Count; i++)
                currentTaskIDOnList[i].SetAlpha(1f);
        }
        else if (!isCurrentlyLooking && isVisible && !isFading)
        {
            StartCoroutine(FadeOut());
        }
    }

    private IEnumerator FadeOut()
    {
        isFading = true;

        yield return new WaitForSeconds(1.0f);

        float shade = activeColor.r;
        float alpha = 1f;

        while (isFading && shade > 0)
        {
            alpha -= (step * 20);
            shade -= step;

            bgMat.color = new Color(shade, shade, shade);

            if (alpha >= 0)
            {
                for (int i = 0; i < currentTaskIDOnList.Count; i++)
                    currentTaskIDOnList[i].SetAlpha(Mathf.Max(0, alpha));
            }

            yield return new WaitForEndOfFrame();
        }

        isFading = false;
        isVisible = false;
    }

    /// <summary>
    /// Instantiate and initialize the task list and it's content
    /// </summary>
    public void SetCurrentTask(int currentTaskID)
    {
        if (tasks == null) return;

        if (currentTaskID < 0 || currentTaskID >= tasks.GetLength(0))
        {
            AngelARUI.Instance.PringDebugMessage("TaskID was invalid: id " + currentTaskID + ", task list length: " + tasks.GetLength(0), false);
            Orb.Instance.SetMessage("");
            return;
        }

        StartCoroutine(SetCurrentTaskAsync(currentTaskID));
    }

    private IEnumerator SetCurrentTaskAsync(int currentTaskID)
    {
        while (!taskListGenerated)
            yield return new WaitForEndOfFrame();

        AngelARUI.Instance.PringDebugMessage("TaskID was valid: " + currentTaskID + ", task list length: " + tasks.GetLength(0), false);

        bool isSubTask = false;
        if (tasks[currentTaskID, 0].Equals("1"))
            isSubTask = true;

        bool isMainTaskAndHasChildren = false;
        if (tasks[currentTaskID, 0].Equals("0") && currentTaskID + 1 < tasks.GetLength(0) && tasks[currentTaskID + 1, 0].Equals("1")) {
            currentTaskID += 1;
            isMainTaskAndHasChildren = true;
        }

        //Deactivate previous task list elements
        for (int i = 0; i < currentTaskIDOnList.Count; i++)
        {
            taskToElement[i].gameObject.SetActive(false);
            currentTaskIDOnList.Remove(taskToElement[i]);
        }

        //Adapt begin and end list index in the UI based on main/subtask relationship
        int startIndex = currentTaskID - (numTasks + 1) / 2;
        if (startIndex < 0)
            startIndex = 0;

        if (startIndex > 0)
            topPointsParent.SetActive(true);
        else
            topPointsParent.SetActive(false);

        int endIndex = startIndex + numTasks;
        if (endIndex > tasks.GetLength(0))
            endIndex = tasks.GetLength(0);

        if (currentTaskID >= tasks.GetLength(0))
            bottomPointsParent.SetActive(false);
        else
            bottomPointsParent.SetActive(true);

        for (int i = startIndex; i < endIndex; i++)
        {
            TaskListElement current = taskToElement[i];
            if ((isSubTask || isMainTaskAndHasChildren) && i == (currentTaskID - 1))
            {
                current = taskToElement[taskToParent[currentTaskID]];
                int subTasksDone = currentTaskID - current.id - 1;
                current.SetAsCurrent(subTasksDone + "/" + mainToSubTasks[taskToParent[currentTaskID]].Count);

            } else
            {
                if (i < currentTaskID)
                    current.SetIsDone(true);

                else if (i == currentTaskID)
                    current.SetAsCurrent("");
                else
                    current.SetIsDone(false);
            }

            current.gameObject.SetActive(true);
            currentTaskIDOnList.Add(current);
        }

        Orb.Instance.SetMessage(tasks[currentTaskID, 1]);
    }

    public void SetTasklist(string[,] tasks)
    {
        if (tasks != null)
        {
            this.tasks = tasks;
            taskListGenerated = false;

            taskToParent = new Dictionary<int, int>();

            int lastParent = 0;
            int lastGrandparent = 0;
            for (int i = 0; i < tasks.GetLength(0); i++)
            {
                if (tasks[i, 0].Equals("0"))
                {
                    lastGrandparent = i;
                }

                else if (tasks[i, 0].Equals("1"))
                {
                    taskToParent.Add(i, lastGrandparent);
                    lastParent = i;
                }

                else if (tasks[i, 0].Equals("2"))
                    taskToParent.Add(i, lastParent);
            }

            StartCoroutine(GenerateTaskListElementsAsync(tasks));

            if (tasks.GetLength(0) > numTasks)
                bottomPointsParent.SetActive(true);
            else
                bottomPointsParent.SetActive(false);

        }
    }

    private IEnumerator GenerateTaskListElementsAsync(string[,] tasks)
    {
        AngelARUI.Instance.PringDebugMessage("Generate template for task list.", false);

        SetTaskListActive(false);

        list.SetActive(false);

        if (taskToElement != null)
        {
            for (int i = 0; i < taskToElement.Count; i++)
                Destroy(taskToElement[i].gameObject);

            taskToElement = null;
        }

        taskToElement = new Dictionary<int, TaskListElement>();
        currentTaskIDOnList = new List<TaskListElement>();
        mainToSubTasks = new Dictionary<int, List<TaskListElement>>();

        int lastMainIndex = 0;
        for (int i = 0; i < tasks.GetLength(0); i++)
        {
            GameObject task = Instantiate(taskPrefab, taskContainer.transform);
            TaskListElement t = task.gameObject.AddComponent<TaskListElement>();
            t.SetText(i, tasks[i, 1], Int32.Parse(tasks[i, 0]));
            t.SetIsDone(false);
            t.gameObject.SetActive(false);

            taskToElement.Add(i, t);

            if (tasks[i, 0].Equals("0"))
                lastMainIndex = i;
            else if (tasks[i, 0].Equals("1"))
            {
                if (!mainToSubTasks.ContainsKey(lastMainIndex))
                    mainToSubTasks.Add(lastMainIndex, new List<TaskListElement>());
                mainToSubTasks[lastMainIndex].Add(t);
            }

            if (i % 10 == 0)
                yield return new WaitForEndOfFrame();
        }

        yield return new WaitForEndOfFrame();

        taskListGenerated = true;

        Orb.Instance.SetTaskListButtonActive(true);
        AngelARUI.Instance.PringDebugMessage("Finished generating task list", false);
    }



    private void LateUpdate()
    {
        progressLine.Start = new Vector3(0, ((taskContainer.rect.y)) * -1, 0);
        progressLine.End = new Vector3(0, ((taskContainer.rect.y)), 0);

        topPointsParent.transform.position = progressLine.transform.TransformPoint(progressLine.Start);
        bottomPointsParent.transform.position = progressLine.transform.TransformPoint(progressLine.End);

    }

    private void IsLookingAt(bool isLooking) => isCurrentlyLooking = isLooking;

    public void ToggleTasklist() => SetTaskListActive(!list.activeInHierarchy);

    public void SetTaskListActive(bool isActive)
    {
        if (isProcessingOpening || !taskListGenerated) return;
        AngelARUI.Instance.PringDebugMessage("Show Task list: " + isActive, false);

        if (isActive)
        {
            isProcessingOpening = true;
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

        isProcessingOpening = false;
    }

}
