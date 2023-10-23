using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using UnityEngine;
using UnityEngine.Events;

public enum SusbcriberType
{
    UpdateTask, UpdateStep, UpdateActiveTask
}

public class DataProvider : Singleton<DataProvider>
{
    private Dictionary<string, string> _manual = null; // Don't write to manual, only read! Manual should be only set once.

    private Dictionary<string, TaskList> _currentSelectedTasks = new Dictionary<string, TaskList>();
    public Dictionary<string, TaskList> CurrentSelectedTasks => _currentSelectedTasks;

    private string _currentObservedTask = "";
    public string CurrentObservedTask => _currentObservedTask;

    private Dictionary<string, CVDetectedObj> DetectedObjects = new Dictionary<string, CVDetectedObj>();

    #region Data Update Event Handling

    private List<UnityEvent> UpdateTasksSubscribers = new List<UnityEvent>(); /// <Events are triggered if task list changed (add or removal of tasks)
    private List<UnityEvent> UpdateStepSubscribers = new List<UnityEvent>();  /// <Events are triggered if step changed at any task list
    private List<UnityEvent> UpdateActiveTaskSubscribers = new List<UnityEvent>(); /// <Events are triggered if active task by user was detected 
                                                                                                                                                                 
    private void PublishToSubscribers(SusbcriberType type)
    {
        if (type.Equals(SusbcriberType.UpdateTask))
        {
            foreach (var subscriber in UpdateTasksSubscribers)
            {
                subscriber.Invoke();
            }
        } else if (type.Equals(SusbcriberType.UpdateActiveTask))
        {
            foreach (var subscriber in UpdateActiveTaskSubscribers)
            {
                subscriber.Invoke();
            }
        } else
        {
            foreach (var subscriber in UpdateStepSubscribers)
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

        if (type.Equals(SusbcriberType.UpdateTask))
            UpdateTasksSubscribers.Add(newDataUpdateEvent);

        else if (type.Equals(SusbcriberType.UpdateActiveTask))
            UpdateActiveTaskSubscribers.Add(newDataUpdateEvent);

        else
            UpdateStepSubscribers.Add(newDataUpdateEvent);
    }

    #endregion

    #region Task List Data 
    /// <summary>
    /// Initialize the task manual. Can only be set once at start and contains the file names of all tasks
    /// Go to the resources folder and load a new tasklist
    /// json file. The name of the file should be in the form
    /// (recipename).json and we will look in 'Resources/Text'
    /// </summary>
    /// <param name="filenamesWithoutExtension"></param>
    public void InitManual(List<string> filenamesWithoutExtension)
    {
        if (_manual != null) return;
        else
            _manual = new Dictionary<string, string>();

        foreach (string filename in filenamesWithoutExtension)
        {
            var jsonTextFile = Resources.Load<TextAsset>("Text/" + filename);
            AngelARUI.Instance.LogDebugMessage("DATA PROVIDER: loaded task from json: " + jsonTextFile.text, true);

            _manual.Add(filename, jsonTextFile.text);
        }
    }

    /// <summary>
    /// Set the list of all tasks that the has to do.
    ///
    /// Nothing changes if tasksToBe is null or empty or longer than the manual, else an event
    /// is published to all subscribers that the task list changed.
    /// </summary>
    /// <param name="tasksToBe"></param>
    public void SetSelectedTasksFromManual(List<string> tasksToBe)
    {
        if (_manual==null || tasksToBe == null || tasksToBe.Count > _manual.Keys.Count) return;

        Dictionary<string, TaskList> copy = new Dictionary<string, TaskList>(_currentSelectedTasks);

        //Add potential new ones that have not been selected before and existing tasks
        foreach (string taskID in tasksToBe)
        {
            if (_manual.Keys.Contains(taskID)) //check if it exists in manual
            {
                if (!copy.Keys.Contains(taskID)) //check if task was already selected
                {
                    TaskList task = JsonUtility.FromJson<TaskList>(_manual[taskID]);
                    copy.Add(taskID, task);
                }
            }
        }

        //Check if tasks were removed from new list
        foreach (string taskID in _currentSelectedTasks.Keys)
        {
            if (!tasksToBe.Contains(taskID)) //check if it exists in manual
                copy.Remove(taskID);
        }

        if (copy.Keys.Count>0 && (_currentObservedTask.Equals("") || !copy.ContainsKey(_currentObservedTask))) //Set the a random initial value for the currentObservedTask
            SetCurrentlyObservedTask(copy.First().Key);
            
        _currentSelectedTasks = copy;

        string debug = "DATA PROVIDER: selected tasks changed to: ";
        foreach (string taskID in _currentSelectedTasks.Keys)
            debug += taskID + ", ";
        AngelARUI.Instance.LogDebugMessage(debug, true);
        PublishToSubscribers(SusbcriberType.UpdateTask);
    }

    /// <summary>
    /// Set the task that is currently observed by the task monitor
    ///
    /// Nothing happens if the taskID is not present in the current list of selected tasks by the user,
    /// else an event is published to all subscribers that the currenlty observed task changed.
    /// </summary>
    /// <param name="taskID"></param>
    public void SetCurrentlyObservedTask(string taskID)
    {
        if (_manual == null || _currentSelectedTasks == null || !_currentSelectedTasks.ContainsKey(taskID)) return;
        _currentObservedTask = taskID;

        AngelARUI.Instance.LogDebugMessage("DATA PROVIDER: currently observed event: "+ taskID, true);
        PublishToSubscribers(SusbcriberType.UpdateActiveTask);
    }

    /// <summary>
    /// Sets the current step that the user has to do of a given task
    /// If step index is < 0, the current step index is set to 0 (first step in the tasklist)
    /// if the step index is => than the number of steps of the given task, the current step index is
    /// set to the last step in the task list.
    ///
    /// Nothing happens if taskID is not present in the the currently selected tasks
    /// or the user did not select any task, else an event is published to all subscribers
    /// that the current step index changed.
    /// </summary>
    /// <param name="taskID">id of task list</param>
    /// <param name="stepIndex">index of current step in the task list given by taskID</param>
    public void SetCurrentStep(string taskID, int stepIndex)
    {
        if (_manual == null || _currentSelectedTasks == null || !_currentSelectedTasks.ContainsKey(taskID)) return;

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

        AngelARUI.Instance.LogDebugMessage("DATA PROVIDER: current step index changed for: " + taskID +" - ID: "+ stepIndex, true);
        PublishToSubscribers(SusbcriberType.UpdateStep);
    }

    /// <summary>
    /// Proceed to the next step at the given task. Nothing happens if the taskID is not currently selected
    /// </summary>
    /// <param name="taskID"></param>
    public void GoToNextStep(string taskID)
    {
        if (_manual == null || _currentSelectedTasks == null || !_currentSelectedTasks.ContainsKey(taskID)) return;
        int potentialStepIndex = _currentSelectedTasks[taskID].CurrStepIndex + 1;

        SetCurrentStep(taskID, potentialStepIndex);
    }

    /// <summary>
    /// Proceed to the previous step at the given task. Nothing happens if the taskID is not currently selected
    /// </summary>
    /// <param name="taskID"></param>
    public void GoToPreviousStep(string taskID)
    {
        if (_manual == null || _currentSelectedTasks == null || !_currentSelectedTasks.ContainsKey(taskID)) return;
        int potentialStepIndex = _currentSelectedTasks[taskID].CurrStepIndex -1;

        SetCurrentStep(taskID, potentialStepIndex);
    }

    /// <summary>
    /// Remove the given task from the currently selected tasklist (e.g.,if task is done)
    /// </summary>
    /// <param name="taskID"></param>
    public void RemoveTaskFromSelected(string taskID)
    {
        if (!_currentSelectedTasks.ContainsKey(taskID)) return;

        List<string> copy = new List<string>(_currentSelectedTasks.Keys);
        copy.Remove(taskID);
        SetSelectedTasksFromManual(copy);
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
