using DilmerGames.Core.Singletons;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using UnityEngine.UI;

public class TaskListManager : Singleton<TaskListManager>
{
    private string[,] tasks;
    private Dictionary<int, int> taskToParent;
    private Dictionary<int, TaskListElement> taskToElement;
    private bool taskListGenerated = false;
    private List<TaskListElement> currentTasksOnList;
    private Dictionary<int, List<TaskListElement>> mainToSubTasks;

    private bool isProcessingOpening = false;
    private bool isProcessingRepositioning = false;

    //Reference to background panel, list container and taskprefabs
    private GameObject list;
    private RectTransform taskContainer;
    private GameObject taskPrefab;

    /// Must be >=1 and an odd number
    private int maxNumTasksOnList = 7;
    private int currentTaskIDInList = 0;

    private Shapes.Line progressLine;
    private GameObject topPointsParent;
    private GameObject bottomPointsParent;

    // Eye-gaze based updates
    private bool isLookingAtTaskList = false;
    private bool isVisible = false;
    private bool isFading = false;
    private bool isDragging = false;
    private Material bgMat;
    private EyeTrackingTarget listEyeTarget;
    private Color activeColor = new Color(0.06f, 0.06f, 0.06f, 0.5f);
    private float step = 0.001f;

    private BoxCollider taskListCollider;
    private Vector3 openCollidersize;
    private Vector3 openColliderCenter;

    private Vector3 closedCollidersize;
    private Vector3 closedColliderCenter;

    private float minDistance = 0.6f;
    private float maxDistance = 1f;

    private void Awake()
    {
        list = transform.GetChild(0).gameObject;

        taskContainer = GameObject.Find("TaskContainer").GetComponent<RectTransform>();
        bgMat = taskContainer.GetComponent<Image>().material;
        taskPrefab = Resources.Load(StringResources.taskprefab_path) as GameObject;

        listEyeTarget = GetComponentInChildren<EyeTrackingTarget>();
        //listEyeTarget.OnLookAtStart.AddListener(delegate { SetIsLookingAt(true); });
        //listEyeTarget.OnLookAway.AddListener(delegate { SetIsLookingAt(false); });

        progressLine = GetComponentInChildren<Shapes.Line>();
        bottomPointsParent = GameObject.Find("BottomPoints");
        bottomPointsParent.transform.parent = progressLine.transform;
        topPointsParent = GameObject.Find("TopPoints");
        topPointsParent.transform.parent = progressLine.transform;
        bottomPointsParent.SetActive(false);
        topPointsParent.SetActive(false);

        list.SetActive(false);

        taskListCollider = gameObject.GetComponent<BoxCollider>();
        openCollidersize = taskListCollider.size;
        openColliderCenter = taskListCollider.center;
        closedCollidersize = new Vector3 (0.1f, 0.3f, 0.03f);
        closedColliderCenter = new Vector3(-0.21f,0,0);

        StartCoroutine(RunDistanceUpdate());
    }

    #region Generate tasklist at runtime

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

            if (tasks.GetLength(0) > maxNumTasksOnList)
                bottomPointsParent.SetActive(true);
            else
                bottomPointsParent.SetActive(false);

        }
    }

    private IEnumerator GenerateTaskListElementsAsync(string[,] tasks)
    {
        AngelARUI.Instance.LogDebugMessage("Generate template for task list.", false);

        SetTaskListActive(false);

        list.SetActive(false);
        taskListCollider.enabled = false;

        if (taskToElement != null)
        {
            for (int i = 0; i < taskToElement.Count; i++)
                Destroy(taskToElement[i].gameObject);

            taskToElement = null;
        }

        taskToElement = new Dictionary<int, TaskListElement>();
        currentTasksOnList = new List<TaskListElement>();
        mainToSubTasks = new Dictionary<int, List<TaskListElement>>();

        int lastMainIndex = 0;
        for (int i = 0; i < tasks.GetLength(0); i++)
        {
            GameObject task = Instantiate(taskPrefab, taskContainer.transform);
            TaskListElement t = task.gameObject.AddComponent<TaskListElement>();
            t.InitText(i, tasks[i, 1], Int32.Parse(tasks[i, 0]));
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
        AngelARUI.Instance.LogDebugMessage("Finished generating task list", false);
    }

    #endregion

    private void Update()
    {
        if (!taskListGenerated || !GetIsTaskListActive()) return;
        
        //**Tasklist is active

        // Update eye tracking flag
        if (isLookingAtTaskList && FollowEyeTarget.Instance.currentHit != EyeTarget.tasklist)
        {
            isLookingAtTaskList = false;
            Orb.Instance.SetSticky(false);
        }
        else if (!isLookingAtTaskList && FollowEyeTarget.Instance.currentHit == EyeTarget.tasklist)
        {
            isLookingAtTaskList = true;
            Orb.Instance.SetSticky(true);
        }

        if (isLookingAtTaskList && !isVisible && !isFading)
        {
            isVisible = true;
            taskListCollider.center = openColliderCenter;
            taskListCollider.size = openCollidersize;

            bgMat.color = activeColor;
            for (int i = 0; i < currentTasksOnList.Count; i++)
                currentTasksOnList[i].SetAlpha(1f);
            
            taskContainer.gameObject.SetActive(true);
        }
        else if (isLookingAtTaskList && isVisible && isFading)
        {
            StopCoroutine(FadeOut());

            isFading = false;
            taskListCollider.center = openColliderCenter;
            taskListCollider.size = openCollidersize;

            bgMat.color = activeColor;
            for (int i = 0; i < currentTasksOnList.Count; i++)
                currentTasksOnList[i].SetAlpha(1f);
        }
        else if (!isLookingAtTaskList && isVisible && !isFading)
        {
            StartCoroutine(FadeOut());
        } 

        if (!isLookingAtTaskList && !isProcessingOpening && !isDragging && !isProcessingOpening && !isProcessingRepositioning)
            UpdatePosition();

        transform.rotation = Quaternion.LookRotation(transform.position - AngelARUI.Instance.mainCamera.transform.position, Vector3.up);
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
                for (int i = 0; i < currentTasksOnList.Count; i++)
                    currentTasksOnList[i].SetAlpha(Mathf.Max(0, alpha));
            }

            yield return new WaitForEndOfFrame();
        }

        if(isFading)
        {
            isFading = false;
            isVisible = false;

            taskContainer.gameObject.SetActive(false);
            taskListCollider.center = closedColliderCenter;
            taskListCollider.size = closedCollidersize;
        }
    }

    /// <summary>
    /// Instantiate and initialize the task list and it's content
    /// </summary>
    public void SetCurrentTask(int currentTaskID)
    {
        if (tasks == null) return;

        if (currentTaskID < 0)
        {
            AngelARUI.Instance.LogDebugMessage("TaskID was invalid: id " + currentTaskID + ", task list length: " + tasks.GetLength(0), false);
            Orb.Instance.SetTaskMessage("");
            return;
        }

        string orbTaskMessage = "All Done!";
        if (currentTaskID < GetTaskCount())
            orbTaskMessage = tasks[currentTaskID, 1];

        currentTaskIDInList = Mathf.Min(GetTaskCount()-1, currentTaskID);
        StartCoroutine(SetCurrentTaskAsync(currentTaskID, orbTaskMessage));
    }

    private IEnumerator SetCurrentTaskAsync(int currentTaskID, string orbMessage)
    {
        while (!taskListGenerated)
            yield return new WaitForEndOfFrame();

        bool allDone = false;
        if (currentTaskID >= tasks.GetLength(0)) {
            allDone = true;
            currentTaskID = tasks.GetLength(0) - 1;
        }

        AngelARUI.Instance.LogDebugMessage("TaskID was valid: " + currentTaskID + ", task list length: " + tasks.GetLength(0), true);

        bool isSubTask = false;
        if (tasks[currentTaskID, 0].Equals("1"))
            isSubTask = true;

        bool isMainTaskAndHasChildren = false;
        if (tasks[currentTaskID, 0].Equals("0") && currentTaskID + 1 < tasks.GetLength(0) && tasks[currentTaskID + 1, 0].Equals("1")) {
            currentTaskID += 1;
            isMainTaskAndHasChildren = true;
        }

        //Deactivate previous task list elements
        for (int i = 0; i < taskToElement.Count; i++)
        {
            taskToElement[i].Reset();
            taskToElement[i].gameObject.SetActive(false);
            currentTasksOnList.Remove(taskToElement[i]);
        }

        //Adapt begin and end list index in the UI based on main/subtask relationship
        int startIndex = currentTaskID - (maxNumTasksOnList + 1) / 2;
        if (startIndex < 0)
            startIndex = 0;

        if (startIndex > 0)
            topPointsParent.SetActive(true);
        else
            topPointsParent.SetActive(false);

        int endIndex = startIndex + maxNumTasksOnList;
        if (endIndex > tasks.GetLength(0))
            endIndex = tasks.GetLength(0);

        if (allDone)
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
                if (i < currentTaskID || allDone)
                    current.SetIsDone(true);

                else if (i == currentTaskID && !allDone)
                    current.SetAsCurrent("");
                else
                    current.SetIsDone(false);
            }

            current.gameObject.SetActive(true);
            currentTasksOnList.Add(current);
        }

        Orb.Instance.SetTaskMessage(orbMessage);
        AudioManager.Instance.PlaySound(Orb.Instance.transform.position, SoundType.taskDone);
    }



    private void LateUpdate()
    {
        progressLine.Start = new Vector3(0, ((taskContainer.rect.y)) * -1, 0);
        progressLine.End = new Vector3(0, ((taskContainer.rect.y)), 0);

        topPointsParent.transform.position = progressLine.transform.TransformPoint(progressLine.Start);
        bottomPointsParent.transform.position = progressLine.transform.TransformPoint(progressLine.End);

        taskListCollider.size = new Vector3(taskListCollider.size.x, taskContainer.rect.height, taskListCollider.size.z);
    }

    public void ToggleTasklist()
    {
        if (list!=null)
            SetTaskListActive(!list.activeInHierarchy);
    }

    #region pose update

    /// <summary>
    /// Update the maximum distance of the tasklist based on the environment collider
    /// </summary>
    private IEnumerator RunDistanceUpdate()
    {
        Debug.Log("maxDist was: " + maxDistance);

        while (true)
        {
            if (taskListCollider.enabled)
            {
                float dist = Utils.GetCameraToPosDist(transform.position);

                if (dist != -1)
                    maxDistance = Mathf.Max(minDistance + 0.02f, Mathf.Min(dist - 0.08f, 1.0f));

                Debug.Log("maxDist was: " + maxDistance);
            }

            yield return new WaitForSeconds(1f);
        }
    }


    private void UpdatePosition()
    {
        if ((Vector3.Distance(AngelARUI.Instance.mainCamera.transform.position, transform.position) > maxDistance
            || Vector3.Angle(transform.position - AngelARUI.Instance.mainCamera.transform.position, AngelARUI.Instance.mainCamera.transform.forward) > 90f))
        {
            StartCoroutine(UpdatePosAndRot(true));
        }
    }

    private IEnumerator UpdatePosAndRot(bool slow)
    {
        isProcessingRepositioning = true;

        Vector3 direction = AngelARUI.Instance.mainCamera.transform.forward;
        var eyeGazeProvider = CoreServices.InputSystem?.EyeGazeProvider;
        if (eyeGazeProvider != null && eyeGazeProvider.IsEyeTrackingEnabledAndValid && eyeGazeProvider.IsEyeCalibrationValid.Value)
            direction = eyeGazeProvider.GazeDirection;

        Vector3 targetPos = AngelARUI.Instance.mainCamera.transform.position + Vector3.Scale(
            direction,
            new Vector3(maxDistance, maxDistance, maxDistance));

        if (slow)
        {
            Vector3 startPos = transform.position;

            float timeElapsed = 0;
            float lerpDuration = 1;
            while (timeElapsed < lerpDuration)
            {
                // Set our position as a fraction of the distance between the markers.
                transform.position = Vector3.Lerp(startPos, targetPos, timeElapsed / lerpDuration);
                transform.SetYPos(AngelARUI.Instance.mainCamera.transform.position.y);

                timeElapsed += Time.deltaTime;
                yield return new WaitForEndOfFrame();
            }

            transform.position = targetPos;
            transform.SetYPos(AngelARUI.Instance.mainCamera.transform.position.y);
        } 
        else
        {
            transform.position = targetPos;
            transform.SetYPos(AngelARUI.Instance.mainCamera.transform.position.y);
        }

        isProcessingRepositioning = false;
    }

    #endregion

    private IEnumerator ShowTaskList()
    {
        isProcessingOpening = true;

        StartCoroutine(UpdatePosAndRot(false));

        while (isProcessingRepositioning)
        {
            yield return new WaitForEndOfFrame();
        }
        
        list.SetActive(true);
        taskListCollider.enabled = true;

        AudioManager.Instance.PlaySound(transform.position, SoundType.notification);

        isProcessingOpening = false;
    }


    #region Getter and Setter

    public bool GetIsTaskListActive() => list.activeInHierarchy;

    public int GetTaskCount()
    {
        if (tasks != null)
            return tasks.GetLength(0);
        else
            return 0;
    }

    public int GetCurrentTaskID() => currentTaskIDInList + 1;

    public void SetTaskListActive(bool isActive)
    {
        if (isProcessingOpening || !taskListGenerated) return;
        AngelARUI.Instance.LogDebugMessage("Show Task list: " + isActive, true);

        if (isActive)
        {
            StartCoroutine(ShowTaskList());
        }
        else
        {
            list.SetActive(false);
            taskListCollider.enabled = false;

            AudioManager.Instance.PlaySound(transform.position, SoundType.notification);
        }
    }

    public void SetIsDragging(bool isDragging)
    {
        this.isDragging = isDragging;
    }

    public void SetNearHover(bool isHovering)
    {
        //TODO
    }

    public void SetAllTasksDone() => SetCurrentTask(GetTaskCount() + 2);

    #endregion
}
