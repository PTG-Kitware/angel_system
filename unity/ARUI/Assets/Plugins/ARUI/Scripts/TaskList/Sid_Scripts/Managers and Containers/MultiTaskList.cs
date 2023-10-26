using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Shapes;
using System;

public class MultiTaskList : Singleton<MultiTaskList>
{
    private List<GameObject> _allTasklists = new List<GameObject>();
    private Dictionary<string, TaskOverviewContainerRepo> _containers;

    private Line _overviewHandle;

    private GameObject _taskOverviewContainer;

    private int _currIndex = 0;
    public int CurrentIndex => _currIndex;

    private float delta = 0;

    [SerializeField]
    private float disableDelay = 1.0f;

    public void Start()
    {
        //Set up child objects
        _overviewHandle = transform.GetChild(0).gameObject.GetComponent<Line>();
        _taskOverviewContainer = transform.GetChild(1).gameObject;

        //Register subscribers
        DataProvider.Instance.RegisterDataSubscriber(() => HandleDataUpdateEvent(), SusbcriberType.TaskListChanged);
        DataProvider.Instance.RegisterDataSubscriber(() => HandleDataUpdateEvent(), SusbcriberType.ObservedTaskChanged);
        DataProvider.Instance.RegisterDataSubscriber(() => HandleDataUpdateEvent(), SusbcriberType.CurrentStepChanged);
        //Set inactive by default
        ToggleOverview(false);
    }

    public void HandleDataUpdateEvent()
    {
        MultiTaskList.Instance.UpdateAllSteps(DataProvider.Instance.CurrentSelectedTasks, DataProvider.Instance.CurrentObservedTask);
    }

    private void Update()
    {
        //if eye gaze not on task objects then do fade out currentindex
        if (EyeGazeManager.Instance != null)
        {
            if (EyeGazeManager.Instance.CurrentHit != EyeTarget.listmenuButton_tasks)
            {
                if (delta > disableDelay)
                {
                    StartCoroutine(FadeOut());
                }
                else
                    delta += Time.deltaTime;
            }
        }

        // Snap orb
        Orb.Instance.SnapToTaskList(transform.GetChild(2).transform.position,
Utils.InFOV(transform.position, AngelARUI.Instance.ARCamera) ||
Utils.InFOV(_taskOverviewContainer.transform.position, AngelARUI.Instance.ARCamera));

        // Scale task list with distance to user 
        float distance = Vector3.Distance(transform.position, AngelARUI.Instance.ARCamera.transform.position);
        float scaleValue = Mathf.Max(0.4f, distance * 0.8f);
        transform.localScale = new Vector3(scaleValue, scaleValue, scaleValue);

        // The canvas should always face the user
        var lookPos = AngelARUI.Instance.ARCamera.transform.position - _taskOverviewContainer.transform.position;
        lookPos.y = 0;
        var rotation = Quaternion.LookRotation(lookPos);
        //_taskOverviewContainer.transform.rotation = rotation * Quaternion.Euler(0, -1, 0);
    }

    #region Managing the main task line 
    /// <summary>
    /// Set the end coordinates of the main task line
    /// </summary>
    /// <param name="EndCords"></param>
    public void SetLineEnd(Vector3 EndCords)
    {
        Vector3 finalCords = _overviewHandle.transform.InverseTransformPoint(EndCords);
        //OverviewLine.End = new Vector3(OverviewLine.End.x, finalCords.y, OverviewLine.End.z);
        _overviewHandle.End = finalCords;
    }
    /// <summary>
    /// Set the start coordinates of the main task line
    /// </summary>
    /// <param name="EndCords"></param>
    public void SetLineStart(Vector3 EndCords)
    {
        Vector3 finalCords = _overviewHandle.transform.InverseTransformPoint(EndCords);
        //OverviewLine.End = new Vector3(OverviewLine.End.x, finalCords.y, OverviewLine.End.z);
        _overviewHandle.Start = finalCords;
    }
    #endregion

    #region Setting inidvidual recipe menus active/inative
    /// <summary>
    /// Sets the overview menu defined by index active
    /// An index of 0 represents the main task while
    /// other indeces are secondary tasks
    /// </summary>
    /// <param name="index"></param>
    public void SetMenuActive(int index)
    {
        TasklistPositionManager.Instance.SetIsLooking(true);
        _currIndex = index;
        for(int i = 0; i < _allTasklists.Count; i++)
        {
            if(i == index)
            {
                _allTasklists[i].SetActive(true);
            } else
            {
                CanvasGroup canvasGroup = _allTasklists[i].GetComponent<CanvasGroup>();
                canvasGroup.alpha = 1.0f;
                _allTasklists[i].SetActive(false);
            }
        }
    }
    #endregion

    #region Managing task overview steps and recipes
    /// <summary>
    /// Takes in all the current tasks stored, key of the current task 
    /// and updates the task overview based on data provided
    /// </summary>
    /// <param name="tasks"></param>
    /// <param name="currTask"></param>
    public void UpdateAllSteps(Dictionary<string, TaskList> tasks, string currTask)
    {
        if (tasks == null) return;

        if (_containers == null || (_containers.Count == 0 && tasks.Count>0))
        {
            //Set Containers for the first time
            InitializeAllContainers(tasks, currTask);
        }

        _overviewHandle.Start = new Vector3(_overviewHandle.Start.x, _overviewHandle.Start.y + 0.015f, _overviewHandle.Start.z);

        ToggleOverview(tasks.Count > 0);

        foreach (KeyValuePair<string, TaskList> pair in tasks)
        { 
            _containers[pair.Key].multiListInstance.SetAsCurrent(pair.Key == currTask);
            _containers[pair.Key].multiListInstance.UpdateProgres(Mathf.Min(1f, Mathf.Max(0f, (float)pair.Value.CurrStepIndex / (float)pair.Value.Steps.Count)));

            if (pair.Value.CurrStepIndex >= pair.Value.Steps.Count) 
            {
                //tak is done
                _containers[pair.Key].setupInstance.SetupCurrTask(null, 0);
                _containers[pair.Key].setupInstance.SetupNextTasks(null, 0);
                _containers[pair.Key].setupInstance.SetupPrevTask(null, 0);
            }
            else
            {
                _containers[pair.Key].setupInstance.SetupPrevTask(pair.Value.Steps, pair.Value.PrevStepIndex);
                _containers[pair.Key].setupInstance.SetupCurrTask(pair.Value.Steps, pair.Value.CurrStepIndex);
                _containers[pair.Key].setupInstance.SetupNextTasks(pair.Value.Steps, pair.Value.NextStepIndex);
            }
        }
    }

    private void InitializeAllContainers(Dictionary<string, TaskList> tasks, string currTask)
    {
        _containers = new Dictionary<string, TaskOverviewContainerRepo>();

        int index= 0;
        foreach (KeyValuePair<string, TaskList> pair in tasks)
        {
            GameObject newOverview = Instantiate(Resources.Load(StringResources.TaskOverview_template_path) as GameObject, _taskOverviewContainer.transform);
            newOverview.transform.localPosition = new Vector3(0, -(0.07f * (index+1)), 0);
            _overviewHandle.Start = new Vector3(_overviewHandle.Start.x, _overviewHandle.Start.y - 0.015f, _overviewHandle.Start.z);
            _taskOverviewContainer.transform.localPosition = new Vector3(_taskOverviewContainer.transform.localPosition.x, _taskOverviewContainer.transform.localPosition.y + 0.025f, _taskOverviewContainer.transform.localPosition.z);

            _containers.Add(pair.Key,newOverview.GetComponent<TaskOverviewContainerRepo>());
            TaskOverviewContainerRepo curr = _containers[pair.Key];
            _allTasklists.Add(curr.taskUI);
            curr.multiListInstance.Index = index;
            curr.taskNameText.SetText(pair.Key);
            SetupCurrTaskOverview currSetup = curr.setupInstance;
            _containers[pair.Key].multiListInstance.UpdateProgres(Mathf.Min(1f, Mathf.Max(0f, (float)pair.Value.CurrStepIndex / (float)pair.Value.Steps.Count)));
            index++;
        }
    }

    #endregion

    #region Setting task overview active and inactive
    /// <summary>
    /// Set the overview (containing all task data) active or inactive
    /// based on current state of _followCameraContainer
    /// </summary>
    public void ToggleOverview()
    {
        if (!_taskOverviewContainer.activeSelf)
        {
            _overviewHandle.gameObject.SetActive(true);
            _taskOverviewContainer.SetActive(true);
            TasklistPositionManager.Instance.SnapToCentroid();
        } else
        {
            _overviewHandle.gameObject.SetActive(false);
            _taskOverviewContainer.SetActive(false);
        }
    }
    /// <summary>
    /// Set the overview (containing all task data) active 
    /// or inactive
    /// </summary>
    public void ToggleOverview(bool active)
    {
        if (active)
        {
            _overviewHandle.gameObject.SetActive(true);
            _taskOverviewContainer.SetActive(true);
        }
        else
        {
            _overviewHandle.gameObject.SetActive(false);
            _taskOverviewContainer.SetActive(false);
        }
    }

    /// <summary>
    /// Fades out entire task overview 
    /// once user does not look at it for a certain
    /// period of time
    /// </summary>
    /// <returns></returns>
    private IEnumerator FadeOut()
    {
        GameObject canvas = null;
        if (_currIndex < _allTasklists.Count)
        {
            canvas = _allTasklists[_currIndex];
            CanvasGroup canvasGroup = canvas.GetComponent<CanvasGroup>();
            float counter = 0f;
            float duration = 1.0f;
            float startAlpha = 1.0f;
            float targetAlpha = 0.0f;
            bool broken = false;
            while (counter < duration)
            {
                if (EyeGazeManager.Instance.CurrentHit == EyeTarget.listmenuButton_tasks)
                {
                    broken = true;
                    break;
                }
                counter += Time.deltaTime;
                if (canvasGroup != null)
                {
                    canvasGroup.alpha = Mathf.Lerp(startAlpha, targetAlpha, counter / duration);
                }

                yield return null;
            }
            if (!broken)
            {
                if (canvas != null)
                {
                    canvas.SetActive(false);
                }
                TasklistPositionManager.Instance.SetIsLooking(false);
                if (canvasGroup != null)
                {
                    canvasGroup.alpha = 1.0f;
                }
            }
            else
            {
                delta = 0.0f;
                canvasGroup.alpha = 1.0f;
                canvas.SetActive(true);
            }
        }
        else
            yield return null;

    }
    #endregion
}
