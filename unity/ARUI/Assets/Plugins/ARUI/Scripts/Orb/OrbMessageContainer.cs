using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum MessageAnchor
{
    left = 1, //message is left from the orb
    right = 2, //message is right from the orb
}

public class OrbMessageContainer : MonoBehaviour
{
    private List<OrbPie> _allPies = new List<OrbPie>();
    private Dictionary<string, OrbPie> taskNameToOrbPie;

    //** Layout
    private MessageAnchor _currentAnchor = MessageAnchor.right;

    //** States
    private bool _isLookingAtMessage = false;
    public bool IsLookingAtMessage
    {
        get { return _isLookingAtMessage; }
    }

    private bool _isMessageContainerActive = false;
    public bool IsMessageContainerActive
    {
        get { return _isMessageContainerActive; }
        set
        {
            _isMessageContainerActive = value;

            foreach (OrbPie op in _allPies)
            {
                op.SetPieActive(value, DataProvider.Instance.CurrentObservedTask);
                if (value)
                {
                    op.Text.BackgroundColor = ARUISettings.OrbMessageBGColor;
                    op.SetTextAlpha(1f);
                }
            }
                
            if (value)
            {
                UpdateAnchorInstant(_currentAnchor);
            } else
            {
                _isMessageFading = false;
            }

            _taskListbutton.gameObject.SetActive(value);
        }
    }

    private bool _isMessageFading = false;
    public bool IsMessageFading
    {
        get { return _isMessageFading; }
        set { _isMessageFading = value; }
    }

    private bool _messageIsLerping = false;
    protected bool IsMessageLerping
    {
        get { return _messageIsLerping; }
        set { _messageIsLerping = value; }
    }

    private DwellButton _taskListbutton;                     /// <reference to dwell btn above orb ('tasklist button')
    public DwellButton TaskListToggle
    {
        get => _taskListbutton;
    }
    
    public bool IsInteractingWithBtn => TaskListToggle != null && TaskListToggle.IsInteractingWithBtn;


    /// <summary>
    /// Init component, get reference to gameobjects from children
    /// </summary>
    public void InitializeComponents()
    {
        // Init tasklist button
        GameObject taskListbtn = transform.GetChild(0).gameObject;
        _taskListbutton = taskListbtn.AddComponent<DwellButton>();
        _taskListbutton.gameObject.name += "FacetasklistButton";
        _taskListbutton.InitializeButton(EyeTarget.orbtasklistButton, () => MultiTaskList.Instance.ToggleOverview(), null, true, DwellButtonType.Select);

        float _startDegRight = 23;
        float _startDegLeft = 180;
        float degR = _startDegRight;
        float degL = _startDegLeft;

        // Init Pie Menu
        for (int i = 0; i < 5; i++)
        {
            GameObject ob = transform.GetChild(1).GetChild(0).GetChild(0).GetChild(0).GetChild(i).gameObject;
            OrbPie current = ob.AddComponent<OrbPie>();
            current.InitializeComponents(degR, degL);
            degR += -23;
            degL += 23;

            _allPies.Add(current);
        }

        taskNameToOrbPie = new Dictionary<string, OrbPie>();

        IsMessageContainerActive = false;
    }

    public void Update()
    {
        // Update eye tracking flag
        if (_isLookingAtMessage && EyeGazeManager.Instance.CurrentHit != EyeTarget.orbMessage
            && EyeGazeManager.Instance.CurrentHit != EyeTarget.orbtasklistButton
            && EyeGazeManager.Instance.CurrentHit != EyeTarget.pieCollider)
            _isLookingAtMessage = false;

        else if (!_isLookingAtMessage && (EyeGazeManager.Instance.CurrentHit == EyeTarget.orbMessage
            || EyeGazeManager.Instance.CurrentHit == EyeTarget.orbtasklistButton)
            || EyeGazeManager.Instance.CurrentHit == EyeTarget.pieCollider)
            _isLookingAtMessage = true;

        if (!IsMessageContainerActive || IsMessageLerping) return;

        // Update messagebox anchor
        if (ChangeMessageBoxToRight(100))
            UpdateAnchorLerp(MessageAnchor.right);

        else if (ChangeMessageBoxToLeft(100))
            UpdateAnchorLerp(MessageAnchor.left);

        foreach (OrbPie pie in _allPies)
        {
            pie.UpdateMessageVisibility(DataProvider.Instance.CurrentObservedTask);
        }
    }

    public void HandleUpdateActiveTaskEvent(Dictionary<string, TaskList> currentSelectedTasks, string currentTaskID)
    {
        foreach (OrbPie pie in taskNameToOrbPie.Values)
        {
            float ratio = (float)currentSelectedTasks[pie.TaskName].CurrStepIndex / (float)(currentSelectedTasks[pie.TaskName].Steps.Count - 1);
            pie.UpdateCurrentTaskStatus(ratio, currentTaskID);
        }
    }

    public void HandleUpdateTaskListEvent(Dictionary<string, TaskList> currentSelectedTasks, string currentTaskID)
    {
        if (currentSelectedTasks.Count == 0 || currentSelectedTasks.Count > 5) return;

        foreach (OrbPie pie in taskNameToOrbPie.Values)
            pie.ResetPie();

        taskNameToOrbPie = new Dictionary<string, OrbPie>();

        int pieIndex = 0;
        foreach (string taskName in currentSelectedTasks.Keys)
        {
            taskNameToOrbPie.Add(taskName, _allPies[pieIndex]); //assign task to pie
            _allPies[pieIndex].TaskName = currentSelectedTasks[taskName].Name;
            _allPies[pieIndex].SetTaskMessage(currentSelectedTasks[taskName].CurrStepIndex,
                currentSelectedTasks[taskName].Steps.Count,
                currentSelectedTasks[taskName].Steps[currentSelectedTasks[taskName].CurrStepIndex].StepDesc);
            _allPies[pieIndex].UpdateMessageVisibility(currentTaskID);
            pieIndex++;
        }

        HandleUpdateActiveTaskEvent(currentSelectedTasks, currentTaskID);
     
    }

    #region Message and Notification Updates

    public void AddNotification(NotificationType type, string message, OrbFace face)
    {
        //_currentActiveMessage.AddNotification(type, message, face);   
    }

    public void RemoveNotification(NotificationType type, OrbFace face)
    {
        //_currentActiveMessage.RemoveNotification(type, face);
    }

    public void RemoveAllNotifications()
    {
        //_currentActiveMessage.RemoveAllNotifications();
    }

    #endregion

    //private void ToggleOrbTaskList() => SetOrbListActive(!_prevText.gameObject.activeSelf);
    private void SetOrbListActive(bool active)
    {
        //_prevText.gameObject.SetActive(active);
        //_nextText.gameObject.SetActive(active);

        //if (active)
        //{
        //    _textContainer.MessageCollider.size = new Vector3(_textContainer.MessageCollider.size.x, 0.08f, _textContainer.MessageCollider.size.z);
        //}
        //else
        //{
        //    _textContainer.MessageCollider.size = new Vector3(_textContainer.MessageCollider.size.x, 0.05f, _textContainer.MessageCollider.size.z);
        //}
    }

    /// <summary>
    /// Turn on or off message fading
    /// </summary>
    /// <param name="active"></param>
    public void SetFadeOutMessage(bool active)
    {
        if (active)
        {
            StartCoroutine(FadeOutMessage());
        }
        else
        {
            StopCoroutine(FadeOutMessage());
            IsMessageFading = false;
        }
    }

    /// <summary>
    /// Fade out message from the moment the user does not look at the message anymore
    /// </summary>
    /// <returns></returns>
    private IEnumerator FadeOutMessage()
    {
        float fadeOutStep = 0.001f;
        IsMessageFading = true;

        yield return new WaitForSeconds(1.0f);

        float shade = ARUISettings.OrbMessageBGColor.r;
        float alpha = 1f;

        while (IsMessageFading && shade > 0)
        {
            alpha -= (fadeOutStep * 20);
            shade -= fadeOutStep;

            if (alpha < 0)
                alpha = 0;
            if (shade < 0)
                shade = 0;

            foreach (OrbPie op in _allPies)
            {
                op.Text.BackgroundColor = new Color(shade, shade, shade, shade);
                op.SetTextAlpha(alpha);
            }

            yield return new WaitForEndOfFrame();
        }

        IsMessageFading = false;
        IsMessageContainerActive = !(shade <= 0);
    }


    public List<BoxCollider> GetAllColliders()
    {
        //throw new System.NotImplementedException();
        var pieColliders = new List<BoxCollider>();
        foreach (OrbPie pie in _allPies)
        {
            pieColliders.AddRange(pie.GetComponentsInChildren<BoxCollider>());
        }

        return pieColliders;
    }

    public void SetTaskMessage(Dictionary<string, TaskList> currentSelectedTasks, string currentTaskID)
    {
        UpdateAnchorInstant(_currentAnchor);

        foreach (string task in currentSelectedTasks.Keys)
        {
            if (taskNameToOrbPie.ContainsKey(task))
            {
                OrbPie ob = taskNameToOrbPie[task];
                if (currentSelectedTasks[task].CurrStepIndex >= currentSelectedTasks[task].Steps.Count) {
                    ob.SetTaskMessage(currentSelectedTasks[task].Steps.Count -1,
                currentSelectedTasks[task].Steps.Count, "Done");
                    AngelARUI.Instance.TryGetUserFeedbackOnUserIntent("Did you finish task '"+task+"'?", () => SetTaskAsDone(task));
                } else
                {
                    ob.SetTaskMessage(currentSelectedTasks[task].CurrStepIndex,
                currentSelectedTasks[task].Steps.Count,
                    currentSelectedTasks[task].Steps[currentSelectedTasks[task].CurrStepIndex].StepDesc);
                }

                float ratio = Mathf.Min(1,(float)currentSelectedTasks[task].CurrStepIndex / (float)(currentSelectedTasks[task].Steps.Count - 1));
                ob.UpdateCurrentTaskStatus(ratio, currentTaskID);
            }
        }

        if (currentSelectedTasks[currentTaskID].CurrStepIndex < currentSelectedTasks[currentTaskID].Steps.Count)
            AudioManager.Instance.PlayText(currentSelectedTasks[currentTaskID].Steps[currentSelectedTasks[currentTaskID].CurrStepIndex].StepDesc);

    }

    /// <summary>
    /// Mark the task with the given ID as done 
    /// </summary>
    /// <param name="taskID"></param>
    private void SetTaskAsDone(string taskID)
    {
        AudioManager.Instance.PlaySound(transform.position,SoundType.taskDone);
        DataProvider.Instance.RemoveTaskFromSelected(taskID);
    }

    #region Update UI

    /// <summary>
    /// Updates the anchor of the messagebox smoothly
    /// </summary>
    /// <param name="MessageAnchor">The new anchor</param>
    private void UpdateAnchorLerp(MessageAnchor newMessageAnchor)
    {
        if (IsMessageLerping) return;

        if (newMessageAnchor != _currentAnchor)
        {
            IsMessageLerping = true;
            _currentAnchor = newMessageAnchor;

            StartCoroutine(MoveMessageBox(newMessageAnchor != MessageAnchor.right, false));
        }
    }

    public void UpdateAnchorInstant(MessageAnchor anchor)
    {
        _currentAnchor = anchor;
        foreach (OrbPie ob in _allPies)
            ob.UpdateAnchor(anchor);

        StartCoroutine(MoveMessageBox(anchor.Equals(MessageAnchor.left), true));
    }

    /// <summary>
    /// Lerps the message box to the other side
    /// </summary>
    /// <param name="YOffset">y offset of the message box to the orb prefab</param>
    /// <param name="addWidth"> if messagebox on the left, change the signs</param>
    /// <param name="instant">if lerp should be almost instant (need to do this in a coroutine anyway, because we are waiting for the Hgroup to update properly</param>
    /// <returns></returns>
    IEnumerator MoveMessageBox(bool isLeft, bool instant)
    {
        float step = 0.1f;

        if (instant)
            step = 0.5f;

        while (step < 1)
        {
            foreach (OrbPie op in _allPies)
            {
                float XOffset = op.InitialXOffset;
                if (isLeft)
                    XOffset = -op.InitialXOffset - op.Text.MessageCollider.size.x;
                float YOffset = op.Text.transform.localPosition.y;

                op.Text.transform.localPosition = Vector2.Lerp(op.Text.transform.localPosition, new Vector3(XOffset, YOffset, 0), step + Time.deltaTime);
                step += Time.deltaTime;
            }

            yield return new WaitForEndOfFrame();
        }

        //if (isLeft)
            //_taskListbutton.transform.SetLocalXPos(-0.043f);
        //else
            //_taskListbutton.transform.SetLocalXPos(0.043f);

        IsMessageLerping = false;
    }


    /// <summary>
    /// Check if message box should be anchored right
    /// </summary>
    /// <param name="offsetPaddingInPixel"></param>
    /// <returns></returns>
    private bool ChangeMessageBoxToRight(float offsetPaddingInPixel)
    {
        return AngelARUI.Instance.ARCamera.WorldToScreenPoint(transform.position).x < ((AngelARUI.Instance.ARCamera.pixelWidth * 0.5f) - offsetPaddingInPixel);
    }

    /// <summary>
    /// Check if message box should be anchored left
    /// </summary>
    private bool ChangeMessageBoxToLeft(float offsetPaddingInPixel)
    {
        return (AngelARUI.Instance.ARCamera.WorldToScreenPoint(transform.position).x > ((AngelARUI.Instance.ARCamera.pixelWidth * 0.5f) + offsetPaddingInPixel));
    }

    #endregion
}
