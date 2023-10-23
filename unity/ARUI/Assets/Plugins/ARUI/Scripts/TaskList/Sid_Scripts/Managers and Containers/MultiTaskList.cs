using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Shapes;

public class MultiTaskList : Singleton<MultiTaskList>
{
    private List<GameObject> _allTasklists = new List<GameObject>();
    private Line _overviewHandle;
    private GameObject _followCameraContainer;
    private GameObject _taskOverviewContainer;
    private GameObject _mainTaskContainer;

    private int _numSecondaryTasks = 0;

    //List of containers for each of the current lists
    private List<TaskOverviewContainerRepo> _containers = new List<TaskOverviewContainerRepo>();

    private int _currIndex = 0;
    public int CurrentIndex => _currIndex;

    [SerializeField]
    private bool isMenu = false;

    private float delta;

    [SerializeField]
    private float disableDelay = 1.0f;

    public void Start()
    {
        //Set up child objects
        _overviewHandle = transform.GetChild(0).gameObject.GetComponent<Line>();
        _followCameraContainer = transform.GetChild(1).gameObject;

        _taskOverviewContainer = _followCameraContainer.transform.GetChild(0).gameObject;
        //Add in main task container
        _mainTaskContainer = Instantiate(Resources.Load(StringResources.Sid_MainTaskOverview_Container_path), _taskOverviewContainer.transform) as GameObject;
        _containers.Add(_mainTaskContainer.GetComponent<TaskOverviewContainerRepo>());
        TaskOverviewContainerRepo curr = _containers[0];
        _allTasklists.Add(curr.taskUI);
        curr.multiListInstance.ListContainer = this.gameObject;
        curr.multiListInstance.index = 0;
        //Register subscribers
        DataProvider.Instance.RegisterDataSubscriber(() => HandleDataUpdateEvent(), SusbcriberType.UpdateTask);
        DataProvider.Instance.RegisterDataSubscriber(() => HandleDataUpdateEvent(), SusbcriberType.UpdateActiveTask);
        DataProvider.Instance.RegisterDataSubscriber(() => HandleDataUpdateEvent(), SusbcriberType.UpdateStep);
        //Set inactive by default
        ToggleOverview(false);
    }

    public void HandleDataUpdateEvent()
    {
        MultiTaskList.Instance.UpdateAllSteps(DataProvider.Instance.CurrentSelectedTasks, DataProvider.Instance.CurrentObservedTask);
    }

    void Update()
    {
        if (isMenu)
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
        }
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
        this.GetComponent<TasklistPositionManager>().SetIsLooking(true);
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
        ResetAllTaskOverviews();
        int index = 1;
        if(tasks.Count > 0)
        {
            ToggleOverview(true);
        } else
        {
            ToggleOverview(false);
        }
        foreach(KeyValuePair<string, TaskList> pair in tasks)
        {
            if (pair.Key == currTask)
            {
                _containers[0].taskNameText.SetText(pair.Key);
                SetupCurrTaskOverview currSetup = _containers[0].setupInstance;
                if (pair.Value.CurrStepIndex != -1)
                {
                    currSetup.SetupCurrTask(pair.Value.Steps[pair.Value.CurrStepIndex], this.GetComponent<TasklistPositionManager>());
                }
                if (pair.Value.NextStepIndex != -1)
                {
                    currSetup.SetupNextTask(pair.Value.Steps[pair.Value.NextStepIndex]);
                } else
                {
                    currSetup.DeactivateNextTask();
                }
                if (pair.Value.PrevStepIndex != -1)
                {
                    currSetup.SetupPrevTask(pair.Value.Steps[pair.Value.PrevStepIndex]);
                } else
                {
                    currSetup.DeactivatePrevTask();
                }
            }
            else
            {
                if (_allTasklists.Contains(_containers[_containers.Count - 1].gameObject))
                    return;
                GameObject currOverview = AddNewTaskOverview();
                _containers.Add(currOverview.GetComponent<TaskOverviewContainerRepo>());
                TaskOverviewContainerRepo curr = _containers[_containers.Count - 1];
                _allTasklists.Add(curr.taskUI);
                curr.multiListInstance.ListContainer = this.gameObject;
                curr.multiListInstance.index = index;
                curr.taskNameText.SetText(pair.Key);
                SetupCurrTaskOverview currSetup = curr.setupInstance;
                if (pair.Value.CurrStepIndex != -1)
                {
                    currSetup.SetupCurrTask(pair.Value.Steps[pair.Value.CurrStepIndex]);
                }
                if (pair.Value.NextStepIndex != -1)
                {
                    currSetup.SetupNextTask(pair.Value.Steps[pair.Value.NextStepIndex]);
                } else
                {
                    currSetup.DeactivateNextTask();
                }
                if (pair.Value.PrevStepIndex != -1)
                {
                    currSetup.SetupPrevTask(pair.Value.Steps[pair.Value.PrevStepIndex]);
                } else
                {
                    currSetup.DeactivatePrevTask();
                }
                index++;
            }
        }
    }
    /// <summary>
    /// Removes all secondary tasks in the overview
    /// so that it can be updated
    /// </summary>
    public void ResetAllTaskOverviews()
    {
        if (_containers.Count == 0) return;
        
        TaskOverviewContainerRepo firstCont = _containers[0];
        int count = _containers.Count;
        for (int i = 1; i < count; i++) {
            _allTasklists.RemoveAt(_allTasklists.Count - 1);
            Destroy(_containers[i].gameObject);
            _overviewHandle.Start = new Vector3(_overviewHandle.Start.x, _overviewHandle.Start.y + 0.015f, _overviewHandle.Start.z);
            _followCameraContainer.transform.localPosition = new Vector3(_followCameraContainer.transform.localPosition.x, _followCameraContainer.transform.localPosition.y - 0.025f, _followCameraContainer.transform.localPosition.z);
            _numSecondaryTasks--;
        } 
        _containers.Clear();
        _containers.Add(firstCont);
    }
    /// <summary>
    /// Adds a secondary task to the task overview
    /// </summary>
    /// <returns></returns>
    public GameObject AddNewTaskOverview()
    {
        _numSecondaryTasks++;
        GameObject newOverview = Instantiate(Resources.Load(StringResources.Sid_TaskOverview_Container_path) as GameObject, _taskOverviewContainer.transform) ;
        newOverview.transform.localPosition = new Vector3(_mainTaskContainer.transform.localPosition.x, _mainTaskContainer.transform.localPosition.y - (0.07f * _numSecondaryTasks), _mainTaskContainer.transform.localPosition.z);
        _overviewHandle.Start = new Vector3(_overviewHandle.Start.x, _overviewHandle.Start.y - 0.015f, _overviewHandle.Start.z);
        _followCameraContainer.transform.localPosition = new Vector3(_followCameraContainer.transform.localPosition.x, _followCameraContainer.transform.localPosition.y + 0.025f, _followCameraContainer.transform.localPosition.z);
        return newOverview;
    }
    #endregion

    #region Setting task overview active and inactive
    /// <summary>
    /// Set the overview (containing all task data) active or inactive
    /// based on current state of _followCameraContainer
    /// </summary>
    public void ToggleOverview()
    {
        if (!_followCameraContainer.activeSelf)
        {
            _overviewHandle.gameObject.SetActive(true);
            _followCameraContainer.SetActive(true);
            TasklistPositionManager.Instance.SnapToCentroid();
        } else
        {
            _overviewHandle.gameObject.SetActive(false);
            _followCameraContainer.SetActive(false);
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
            _followCameraContainer.SetActive(true);
        }
        else
        {
            _overviewHandle.gameObject.SetActive(false);
            _followCameraContainer.SetActive(false);
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
                this.GetComponent<TasklistPositionManager>().SetIsLooking(false);
                if (canvasGroup != null)
                {
                    canvasGroup.alpha = 1.0f;
                }
                this.GetComponent<TasklistPositionManager>().DeactivateLines();
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
