using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Shapes;
using System;

public class MultiTaskList : Singleton<MultiTaskList>
{
    private List<CanvasGroup> _allTasklists = new List<CanvasGroup>();
    private List<BoxCollider> _allColliders = new List<BoxCollider>();
    private Dictionary<string, TaskOverviewContainerRepo> _containers;

    private GameObject _taskOverviewContainer;

    private int _currIndex = 0;
    public int CurrentIndex => _currIndex;

    private float delta = 0;

    [SerializeField]
    private float disableDelay = 1.0f;

    private bool _isActive = false;

    private GameObject _eyeGazeTarget;

    public void Start()
    {
        //Set up child objects
        _taskOverviewContainer = transform.GetChild(0).gameObject;
        TasklistPositionManager.Instance.SnapToCentroid();
        _taskOverviewContainer.gameObject.SetActive(false);

        //Register subscribers
        DataProvider.Instance.RegisterDataSubscriber(() => HandleDataUpdateEvent(), SusbcriberType.TaskListChanged);
        DataProvider.Instance.RegisterDataSubscriber(() => HandleDataUpdateEvent(), SusbcriberType.ObservedTaskChanged);
        DataProvider.Instance.RegisterDataSubscriber(() => HandleDataUpdateEvent(), SusbcriberType.CurrentStepChanged);

        _taskOverviewContainer.SetActive(false);
        _isActive = false;

        _eyeGazeTarget = gameObject;
        EyeGazeManager.Instance.RegisterEyeTargetID(_eyeGazeTarget);
    }

    /// <summary>
    /// Listen to data changes
    /// </summary>
    public void HandleDataUpdateEvent()
    {
        MultiTaskList.Instance.UpdateAllSteps(DataProvider.Instance.CurrentActiveTasks, DataProvider.Instance.CurrentObservedTask);
    }

    private void Update()
    {
        if (!_isActive) return;

        //if eye gaze not on task objects then do fade out currentindex
        if (!isLookingAtAnyTaskOverviewObject())
        {
            if (delta > disableDelay)
                StartCoroutine(FadeOut());
            else
                delta += Time.deltaTime;
        }
        
        // Scale task list with distance to user 
        float distance = Vector3.Distance(transform.position, AngelARUI.Instance.ARCamera.transform.position);
        float scaleValue = Mathf.Max(0.4f, distance * 0.7f);
        transform.localScale = new Vector3(scaleValue, scaleValue, scaleValue);

        // The canvas should always face the user
        var lookPos = transform.position - AngelARUI.Instance.ARCamera.transform.position;
        lookPos.y = 0;
        transform.rotation = Quaternion.LookRotation(lookPos, Vector3.up);

        if (_containers == null) return;
        bool anyMenuActive = false;
        foreach (var canvas in _allTasklists)
        {
            if (canvas.alpha<1)
                anyMenuActive = true;
        }

        foreach (var tasklist in _containers.Values)
            tasklist.multiListInstance.Text.gameObject.SetActive(!anyMenuActive);

    }

    private bool isLookingAtAnyTaskOverviewObject()
    {
        foreach (var tasklist in _allColliders)
        {
            if (EyeGazeManager.Instance != null && EyeGazeManager.Instance.CurrentHitID == tasklist.gameObject.GetInstanceID())
            {
                return true;
            }
        }
        
        foreach (var container in _containers.Values)
        {
            if (EyeGazeManager.Instance != null && EyeGazeManager.Instance.CurrentHitID == container.multiListInstance.EyeGazeTarget.gameObject.GetInstanceID())
            {
                return true;
            }
        }

        return false;
    }

    #region Setting inidvidual recipe menus active/inative

    /// <summary>
    /// Sets the overview menu defined by index active
    /// An index of 0 represents the main task while
    /// other indeces are secondary tasks
    /// </summary>
    /// <param name="index"></param>
    public void SetMenuActive(int index)
    {
        TasklistPositionManager.Instance.IsLooking = true;
        _currIndex = index;
        for (int i = 0; i < _allTasklists.Count; i++)
        {
            if (i == index)
            {
                _allTasklists[i].gameObject.SetActive(true);
            }
            else
            {
                CanvasGroup canvasGroup = _allTasklists[i];
                canvasGroup.alpha = 1.0f;
                _allTasklists[i].gameObject.SetActive(false);
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
            InitializeAllContainers(tasks);

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

    /// <summary>
    /// TODO
    /// </summary>
    /// <param name="tasks"></param>
    private void InitializeAllContainers(Dictionary<string, TaskList> tasks)
    {
        _containers = new Dictionary<string, TaskOverviewContainerRepo>();

        int index= 0;
        foreach (KeyValuePair<string, TaskList> pair in tasks)
        {
            GameObject newOverview = Instantiate(Resources.Load(StringResources.TaskOverview_template_path) as GameObject, _taskOverviewContainer.transform);
            newOverview.transform.localPosition = new Vector3(0, -(0.06f * (index+1)), 0);
            _taskOverviewContainer.transform.localPosition = new Vector3(_taskOverviewContainer.transform.localPosition.x, _taskOverviewContainer.transform.localPosition.y + 0.020f, _taskOverviewContainer.transform.localPosition.z);

            _containers.Add(pair.Key,newOverview.GetComponent<TaskOverviewContainerRepo>());
            TaskOverviewContainerRepo curr = _containers[pair.Key];
            _allTasklists.Add(curr.taskUI.GetComponent<CanvasGroup>());
            curr.multiListInstance.Index = index;
            curr.taskNameText.SetText(pair.Key);
            SetupCurrTaskOverview currSetup = curr.setupInstance;
            _containers[pair.Key].multiListInstance.UpdateProgres(Mathf.Min(1f, Mathf.Max(0f, (float)pair.Value.CurrStepIndex / (float)pair.Value.Steps.Count)));
            index++;
        }

        _allColliders = new List<BoxCollider>();
        foreach (var tasklist in _allTasklists)
            _allColliders.AddRange(tasklist.GetComponentsInChildren<BoxCollider>());

        foreach (var tasklist in _allColliders)
            EyeGazeManager.Instance.RegisterEyeTargetID(tasklist.gameObject);
    }

    #endregion

    #region Task overview visibility

    /// <summary>
    /// Set the overview (containing all task data) active or inactive
    /// based on current state of _followCameraContainer
    /// </summary>
    public void ToggleOverview()
    {
        if (!_taskOverviewContainer.activeSelf)
        {
            _taskOverviewContainer.SetActive(true);
            TasklistPositionManager.Instance.SnapToCentroid();
            _isActive = true;

            MultiTaskList.Instance.UpdateAllSteps(DataProvider.Instance.CurrentActiveTasks, DataProvider.Instance.CurrentObservedTask);
        } else
        {
            _taskOverviewContainer.SetActive(false);
            _isActive = false;
        }
    }

    /// <summary>
    /// Set the overview (containing all task data) active or inactive
    /// </summary>
    public void SetTaskOverViewVisibility(bool visible)
    {
        _taskOverviewContainer.SetActive(visible);
        _isActive = visible;

        if (visible)
        {
            TasklistPositionManager.Instance.SnapToCentroid();
            MultiTaskList.Instance.UpdateAllSteps(DataProvider.Instance.CurrentActiveTasks, DataProvider.Instance.CurrentObservedTask);
        }
    }

    /// <summary>
    /// Set the position of the task overview panel. The panel will always face the user.
    /// </summary>
    /// <param name="worldPosition"></param>
    public void SetPosition(Vector3 worldPosition)
    {
        TasklistPositionManager.Instance.SetPosition(worldPosition);
    }

    /// <summary>
    /// Fades out entire task overview 
    /// once user does not look at it for a certain
    /// period of time
    /// </summary>
    /// <returns></returns>
    private IEnumerator FadeOut()
    {
        if (_currIndex < _allTasklists.Count)
        {
            CanvasGroup canvas =  _allTasklists[_currIndex];
            float counter = 0f;
            float duration = 1.0f;
            float startAlpha = 1.0f;
            float targetAlpha = 0.0f;
            bool broken = false;
            while (counter < duration)
            {
                if (isLookingAtAnyTaskOverviewObject())
                {
                    broken = true;
                    break;
                }
                counter += Time.deltaTime;
                if (canvas != null)
                    canvas.alpha = Mathf.Lerp(startAlpha, targetAlpha, counter / duration);

                yield return null;
            }
            if (!broken)
            {
                if (canvas != null)
                    canvas.gameObject.SetActive(false);

                TasklistPositionManager.Instance.IsLooking = false;

                if (canvas != null)
                    canvas.alpha = 1.0f;
            }
            else
            {
                delta = 0.0f;
                canvas.alpha = 1.0f;
                canvas.gameObject.SetActive(true);
            }
        }
        else
            yield return null;

    }

    #endregion
}
