using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using UnityEngine;
using UnityEngine.Events;

public enum SusbcriberType
{
    TaskListChanged, CurrentStepChanged, ObservedTaskChanged
}

public class DataProvider : Singleton<DataProvider>
{
    private Dictionary<string, string> _manual = null; // Don't write to manual, only read! Manual should be only set once.
    public bool ManualInitialized
    {
        get => _manual != null;
    }

    private Dictionary<string, TaskList> _currentSelectedTasks = new Dictionary<string, TaskList>();
    public Dictionary<string, TaskList> CurrentSelectedTasks => _currentSelectedTasks;

    private string _currentObservedTask = "";
    public string CurrentObservedTask => _currentObservedTask;

    private Dictionary<string, CVDetectedObj> DetectedObjects = new Dictionary<string, CVDetectedObj>();

    #region Data Change Event Handling

    private List<UnityEvent> TaskListChangedSubscribers = new List<UnityEvent>(); /// <Events are triggered if task list changed (add or removal of tasks)
    private List<UnityEvent> CurrentStepChangedSubscribers = new List<UnityEvent>();  /// <Events are triggered if step changed at any task list
    private List<UnityEvent> ObservedTaskChangedSubscribers = new List<UnityEvent>(); /// <Events are triggered if active task by user was detected 

    private void PublishToSubscribers(SusbcriberType type)
    {
        if (type.Equals(SusbcriberType.TaskListChanged))
        {
            foreach (var subscriber in TaskListChangedSubscribers)
            {
                subscriber.Invoke();
            }
        } else if (type.Equals(SusbcriberType.ObservedTaskChanged))
        {
            foreach (var subscriber in ObservedTaskChangedSubscribers)
            {
                subscriber.Invoke();
            }
        } else
        {
            foreach (var subscriber in CurrentStepChangedSubscribers)
            {
                subscriber.Invoke();
            }
        }
    }
    /// <summary>
    /// Register a UnityEvent when the given SubscriberType is triggered
    /// </summary>
    /// <param name="subscriberEvent"></param>
    public void RegisterDataSubscriber(UnityAction subscriberEvent, SusbcriberType type) 
    { 
        UnityEvent newDataUpdateEvent = new UnityEvent();
        newDataUpdateEvent.AddListener(subscriberEvent);

        if (type.Equals(SusbcriberType.TaskListChanged))
            TaskListChangedSubscribers.Add(newDataUpdateEvent);

        else if (type.Equals(SusbcriberType.ObservedTaskChanged))
            ObservedTaskChangedSubscribers.Add(newDataUpdateEvent);

        else
            CurrentStepChangedSubscribers.Add(newDataUpdateEvent);
    }

    #endregion

    #region Task List Data 
    /// <summary>
    /// Initialize the task manual. Can only be set once at start and contains the file names of all tasks
    /// jsonTaskLists is a list of json strings 
    /// </summary>
    /// <param name="jsonTaskLists"></param>
    public void InitManual(Dictionary<string, string> jsonTaskLists)
    {
        if (ManualInitialized || jsonTaskLists == null || jsonTaskLists.Keys.Count == 0) 
            return;
        else
            _manual = new Dictionary<string, string>();

        foreach (string taskID in jsonTaskLists.Keys)
        {
            _manual.Add(taskID, jsonTaskLists[taskID]);
            AngelARUI.Instance.DebugLogMessage("DATA PROVIDER: loaded task from json: " + jsonTaskLists[taskID], true);
        }
    }

    /// <summary>
    /// Set the list of all tasks that the has to do.
    ///
    /// Nothing changes if tasksToBe is null,empty orrlonger than the manual or if tasksToBe is the same as 
    /// the current one, else an event is published to all subscribers that the task list changed.
    /// </summary>
    /// <param name="tasksToBe"></param>
    public void SetSelectedTasksFromManual(List<string> tasksToBe)
    {
        if (!ManualInitialized || tasksToBe == null || tasksToBe.Count > _manual.Keys.Count) return;

        Dictionary<string, TaskList> copy = new Dictionary<string, TaskList>(_currentSelectedTasks);

        bool listChanged = false;
        //Add potential new ones that have not been selected before and existing tasks
        foreach (string taskID in tasksToBe)
        {
            if (_manual.Keys.Contains(taskID)) //check if it exists in manual
            {
                if (!copy.Keys.Contains(taskID)) //check if task was already selected
                {
                    TaskList task = JsonUtility.FromJson<TaskList>(_manual[taskID]);
                    copy.Add(taskID, task);
                    listChanged = true;
                }
            }
        }

        //Check if tasks were removed from new list
        foreach (string taskID in _currentSelectedTasks.Keys)
        {
            if (!tasksToBe.Contains(taskID))
            {//check if it exists in manual 
                copy.Remove(taskID);
                listChanged = true;
            }
        }

        if (!listChanged) return; //Do nothing if the currently observed task list did not change

        //Set new currently selected task
        _currentSelectedTasks = copy;

        //Update currently observed task
        if (copy.Keys.Count > 0 && (_currentObservedTask.Equals("") || !copy.ContainsKey(_currentObservedTask))) //Set the a random initial value for the currentObservedTask
            SetCurrentObservedTask(copy.First().Key);

        string debug = "DATA PROVIDER: selected tasks set to: ";
        foreach (string taskID in _currentSelectedTasks.Keys)
            debug += taskID + ", ";
        AngelARUI.Instance.DebugLogMessage(debug, true);
        PublishToSubscribers(SusbcriberType.TaskListChanged);
    }

    /// <summary>
    /// Set the task that is currently observed by the task monitor
    ///
    /// Nothing happens if the taskID is not present in the current list of selected tasks by the user,
    /// or taskID is the same as the currentlyObservedTask, else an event is published to all subscribers that the currenlty observed task changed.
    /// </summary>
    /// <param name="taskID"></param>
    public void SetCurrentObservedTask(string taskID)
    {
        if (!ManualInitialized || _currentSelectedTasks == null 
            || !_currentSelectedTasks.ContainsKey(taskID)
            || _currentObservedTask == taskID) return;

        _currentObservedTask = taskID;

        AngelARUI.Instance.DebugLogMessage("DATA PROVIDER: currently observed event: "+ taskID, true);
        PublishToSubscribers(SusbcriberType.ObservedTaskChanged);
    }

    /// <summary>
    /// Sets the current step that the user has to do of a given task
    /// If step index is < 0, the current step index is set to 0 (first step in the tasklist)
    /// if the step index is => than the number of steps of the given task, the current step index is
    /// set to the last step in the task list.
    ///
    /// Nothing happens if taskID is not present in the the currently selected tasks
    /// or the user did not select any task or 'stepIndex' is the same as the currnetStepIndex of 
    /// the given task, else an event is published to all subscribers
    /// that the current step index changed.
    /// </summary>
    /// <param name="taskID">id of task list</param>
    /// <param name="stepIndex">index of current step in the task list given by taskID</param>
    public void SetCurrentStep(string taskID, int stepIndex)
    {
        if (!ManualInitialized || _currentSelectedTasks == null 
            || !_currentSelectedTasks.ContainsKey(taskID)
            || _currentSelectedTasks[taskID].CurrStepIndex == stepIndex) return;

        if (stepIndex <= 0)
        {
            _currentSelectedTasks[taskID].PrevStepIndex = -1;
            _currentSelectedTasks[taskID].CurrStepIndex = 0;
            _currentSelectedTasks[taskID].NextStepIndex = 1;

        } else if (stepIndex == _currentSelectedTasks[taskID].Steps.Count - 1)
        {
            _currentSelectedTasks[taskID].PrevStepIndex = _currentSelectedTasks[taskID].Steps.Count-2;
            _currentSelectedTasks[taskID].CurrStepIndex = _currentSelectedTasks[taskID].Steps.Count-1;
            _currentSelectedTasks[taskID].NextStepIndex = -1;
        }
        else if (stepIndex > _currentSelectedTasks[taskID].Steps.Count - 1)
        {
            _currentSelectedTasks[taskID].PrevStepIndex = _currentSelectedTasks[taskID].Steps.Count-1;
            _currentSelectedTasks[taskID].CurrStepIndex = _currentSelectedTasks[taskID].Steps.Count;
            _currentSelectedTasks[taskID].NextStepIndex = -1;
        }
        else
        {
            _currentSelectedTasks[taskID].PrevStepIndex = stepIndex - 1;
            _currentSelectedTasks[taskID].CurrStepIndex = stepIndex;
            _currentSelectedTasks[taskID].NextStepIndex = stepIndex + 1;
        }

        AngelARUI.Instance.DebugLogMessage("DATA PROVIDER: current step index changed for: " + taskID +" - ID: "+ stepIndex, true);
        PublishToSubscribers(SusbcriberType.CurrentStepChanged);
    }

    /// <summary>
    /// Proceed to the next step at the given task. Nothing happens if the taskID is not currently selected
    /// </summary>
    /// <param name="taskID"></param>
    private void GoToNextStep(string taskID)
    {
        if (!ManualInitialized || _currentSelectedTasks == null || !_currentSelectedTasks.ContainsKey(taskID)) return;
        int potentialStepIndex = _currentSelectedTasks[taskID].CurrStepIndex + 1;

        SetCurrentStep(taskID, potentialStepIndex);
    }

    /// <summary>
    /// Proceed to the previous step at the given task. Nothing happens if the taskID is not currently selected
    /// </summary>
    /// <param name="taskID"></param>
    private void GoToPreviousStep(string taskID)
    {
        if (_manual == null || _currentSelectedTasks == null || !_currentSelectedTasks.ContainsKey(taskID)) return;
        int potentialStepIndex = _currentSelectedTasks[taskID].CurrStepIndex -1;

        SetCurrentStep(taskID, potentialStepIndex);
    }

    #endregion

    #region CV Detected Objects
    /// <summary>
    /// Add a 3D mesh to view management. BBox should contain a mesh filter
    /// </summary>
    /// <param name="bbox">The position, rotation, scale and mesh of this object should be considered in view management</param>
    /// <param name="ID">ID to identify the gameobject that should be added</param>
    public void AddDetectedObjects(GameObject bbox, string ID)
    {
        if (DetectedObjects.ContainsKey(ID)) return;

        GameObject copy = Instantiate(bbox);
        copy.gameObject.name = "***ARUI-CVDetected-" + ID;

        // destroy mesh renderer, if attached
        if (copy.GetComponent<MeshRenderer>() != null)
            Destroy(copy.GetComponent<MeshRenderer>());

        CVDetectedObj ndetection = copy.AddComponent<CVDetectedObj>();
        DetectedObjects.Add(ID, ndetection);
    }

    /// <summary>
    /// Remove a 3D mesh from view management
    /// </summary>
    /// <param name="ID">ID to identify the gameobject that should be removed</param>
    public void RemoveDetectedObjects(string ID)
    {
        if (!DetectedObjects.ContainsKey(ID)) return;

        StartCoroutine(LateDestroy(DetectedObjects[ID]));
        DetectedObjects.Remove(ID);
    }

    private IEnumerator LateDestroy(CVDetectedObj temp)
    {
        temp.IsDestroyed = true;

        yield return new WaitForSeconds(0.2f);

        Destroy(temp.gameObject);
    }

    #endregion
}
