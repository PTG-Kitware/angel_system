using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

public enum MessageAlignment
{
    LockLeft = 1,       //message stays on the left side of agent
    LockRight = 2,      //message stays on the right side of the agent
    Auto = 3,           //message adjust dynamically based on agent position in view space
}

public class OrbMessageContainer : MonoBehaviour
{
    private OrbNotificationManager _orbNotificationManager;

    private List<OrbTask> _allTasksPlaceholder = new List<OrbTask>();

    //** OrbTasks after manual is set
    private OrbTask _mainTaskPiePlace;
    private Dictionary<string, OrbTask> _taskNameToOrbPie;

    //** Layout
    private MessageAlignment _currentAlignment = MessageAlignment.Auto;
    private bool _currentAlignmentIsRight = true;

    //** States
    private bool _isLookingAtMessage = false;
    public bool IsLookingAtMessage
    {
        get { return _isLookingAtMessage; }
    }

    private OrbWarning _currentWarning;
    public bool IsWarningActive => _currentWarning.IsSet;


    private TMPro.TextMeshProUGUI _prevText;
    private TMPro.TextMeshProUGUI _nextText;
    private Color _textColor = new Color(0.8f,0.8f,0.8f,1.0f);


    private bool _isMessageContainerActive = false;
    public bool IsMessageContainerActive
    {
        get { return _isMessageContainerActive; }
        set
        {
            _isMessageContainerActive = value;

            foreach (OrbTask op in _taskNameToOrbPie.Values)
            {
                op.SetPieActive(value);
                if (value)
                {
                    op.Text.BackgroundColor = ARUISettings.OrbMessageBGColor;
                    op.SetTextAlpha(1f);
                    SetTextAlphaOthers(1f);
                }
            }
                
            if (value)
            {
                UpdateAnchorInstant();
            } else
            {
                _isMessageFading = false;
            }

            //_taskListbutton.gameObject.SetActive(value);
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
        //_taskListbutton.InitializeButton(EyeTarget.orbtasklistButton, () => MultiTaskList.Instance.ToggleOverview(), null, true, DwellButtonType.Toggle);
        taskListbtn.SetActive(false);

        // Init Pie Menu
        for (int i = 0; i < 4; i++)
        {
            GameObject ob = transform.GetChild(1).GetChild(0).GetChild(0).GetChild(1).GetChild(i).gameObject;
            OrbTask current = ob.AddComponent<OrbTask>();
            current.InitializeComponents(TaskType.secondary);

            _allTasksPlaceholder.Add(current);
        }

        GameObject obMain = transform.GetChild(1).GetChild(0).GetChild(0).GetChild(0).gameObject;
        _mainTaskPiePlace = obMain.AddComponent<OrbTask>();
        _mainTaskPiePlace.InitializeComponents(TaskType.primary);

        _taskNameToOrbPie = new Dictionary<string, OrbTask>();

        IsMessageContainerActive = false;
        _currentWarning = transform.GetChild(2).gameObject.AddComponent<OrbWarning>();
        _currentWarning.Init("", _mainTaskPiePlace.TextRect.height);
        _currentWarning.gameObject.SetActive(false);

        _prevText = obMain.transform.GetChild(1).GetChild(0).GetChild(0).GetChild(0).GetChild(0).GetChild(1).gameObject.GetComponentInChildren<TMPro.TextMeshProUGUI>();
        _prevText.text = "";
        _nextText = obMain.transform.GetChild(1).GetChild(0).GetChild(0).GetChild(0).GetChild(0).GetChild(2).gameObject.GetComponentInChildren<TMPro.TextMeshProUGUI>();
        _nextText.text = "";

        //Init the notification manager at orb
        _orbNotificationManager = transform.GetChild(1).GetChild(0).GetChild(0).GetChild(2).GetComponentInChildren<VerticalLayoutGroup>().gameObject.AddComponent<OrbNotificationManager>();
        _orbNotificationManager.gameObject.name = "***ARUI-" + StringResources.NotificationManager_name;
    }

    /// <summary>
    /// If confirmation action is set - SetUserIntentCallback(...) - and no confirmation window is active at the moment, the user is shown a 
    /// timed confirmation window. Recommended text: "Did you mean ...". If the user confirms the dialogue, the onUserIntentConfirmedAction action is invoked. 
    /// </summary>
    /// <param name="msg">Message that is shown in the Confirmation Dialogue</param>
    /// <param name="actionOnConfirmation">Actions triggerd if the user confirms the dialogue</param>
    /// <param name="actionOnTimeOut">OPTIONAL - Action triggered if notification times out</param>
    public void TryGetUserConfirmation(string msg, List<UnityAction> actionOnConfirmation, UnityAction actionOnTimeOut, float timeout)
    {
        _orbNotificationManager.TryGetUserConfirmation(msg, actionOnConfirmation, actionOnTimeOut, timeout);
    }

    /// <summary>
    /// TODO
    /// </summary>
    /// <param name="selectionMsg"></param>
    /// <param name="choices"></param>
    /// <param name="actionOnSelection"></param>
    /// <param name="actionOnTimeOut"></param>
    /// <param name="timeout"></param>
    public void TryGetUserChoice(string selectionMsg, List<string> choices, List<UnityAction> actionOnSelection, UnityAction actionOnTimeOut, float timeout)
    {
        _orbNotificationManager.TryGetUserChoice(selectionMsg, choices, actionOnSelection, actionOnTimeOut, timeout);
    }

    public void TryGetUserYesNoChoice(string selectionMsg, UnityAction actionOnYes, UnityAction actionOnNo, UnityAction actionOnTimeOut, float timeout)
    {
        _orbNotificationManager.TryGetUserYesNoChoice(selectionMsg, actionOnYes, actionOnNo, actionOnTimeOut, timeout);
    }

    public void Update()
    {
        // Update eye tracking flag
        var lookingAtAnyTask = false;

        foreach (var orbtask in _taskNameToOrbPie.Values)
        {
            if (orbtask != null && orbtask.IsLookingAtTask)
                lookingAtAnyTask = true;
        }

        if (_isLookingAtMessage && lookingAtAnyTask == false)
            _isLookingAtMessage = false;

        else if (!_isLookingAtMessage && lookingAtAnyTask)
            _isLookingAtMessage = true;

        _currentWarning.UpdateSize(_mainTaskPiePlace.TextRect.width / 2);

        Vector2 anchor = _currentWarning.transform.GetComponent<RectTransform>().anchoredPosition;
        _currentWarning.transform.GetComponent<RectTransform>().anchoredPosition = new Vector2(anchor.x, _mainTaskPiePlace.TextRect.height + 0.01f);

        anchor = _prevText.transform.parent.GetComponent<RectTransform>().anchoredPosition;
        _prevText.transform.parent.GetComponent<RectTransform>().anchoredPosition = new Vector2(anchor.x, _mainTaskPiePlace.TextRect.height+0.01f);

        anchor = _nextText.transform.parent.GetComponent<RectTransform>().anchoredPosition;
        _nextText.transform.parent.GetComponent<RectTransform>().anchoredPosition = new Vector2(anchor.x, -(_mainTaskPiePlace.TextRect.height + 0.01f));

        if (!IsMessageContainerActive || IsMessageLerping) return;

        if (_currentAlignment.Equals(MessageAlignment.Auto))
        {
            // Update messagebox anchor
            if (ChangeMessageBoxToRight(100))
                UpdateAnchorLerp(true);

            else if (ChangeMessageBoxToLeft(100))
                UpdateAnchorLerp(false);
        }
    }

    /// <summary>
    /// Handles updates if the currently observed task updates
    /// </summary>
    /// <param name="currentActiveTasks"></param>
    /// <param name="currentTaskID"></param>
    public void HandleUpdateActiveTaskEvent(Dictionary<string, TaskList> currentActiveTasks, string currentTaskID)
    {
        HandleUpdateTaskListEvent(currentActiveTasks, currentTaskID);
    }

    /// <summary>
    /// Handles updates to the task list (e.g., if stepIndex updates)
    /// </summary>
    /// <param name="currentActiveTasks"></param>
    /// <param name="currentTaskID"></param>
    public void HandleUpdateTaskListEvent(Dictionary<string, TaskList> currentActiveTasks, string currentTaskID)
    {
        if (currentActiveTasks.Count == 0 || currentActiveTasks.Count > 5) return;

        foreach (OrbTask pie in _taskNameToOrbPie.Values)
            pie.ResetPie();

        _taskNameToOrbPie = new Dictionary<string, OrbTask>();

        int pieIndex = 0;
        foreach (string taskName in currentActiveTasks.Keys)
        {
            if (taskName.Equals(currentTaskID))
            {
                _taskNameToOrbPie.Add(taskName, _mainTaskPiePlace);
                _mainTaskPiePlace.TaskName = currentActiveTasks[taskName].Name;
            }
            else
            {
                _taskNameToOrbPie.Add(taskName, _allTasksPlaceholder[pieIndex]); //assign task to pie
                _allTasksPlaceholder[pieIndex].TaskName = currentActiveTasks[taskName].Name;
                pieIndex++;
            }
        }

        UpdateAllTaskMessages(currentActiveTasks);
    }

    public void UpdateAllTaskMessages(Dictionary<string, TaskList> currentActiveTasks)
    {
        UpdateAnchorInstant();

        string tempName = "";
        foreach (string taskName in currentActiveTasks.Keys)
        {
            if (_taskNameToOrbPie.ContainsKey(taskName))
            {
                if (currentActiveTasks[taskName].CurrStepIndex >= currentActiveTasks[taskName].Steps.Count)
                {
                    if (_taskNameToOrbPie[taskName].gameObject.activeSelf)
                    {
                        AudioManager.Instance.PlaySound(transform.position, SoundType.confirmation);
                        _taskNameToOrbPie[taskName].gameObject.SetActive(false);
                    }
                }
                else
                {
                    if (!_taskNameToOrbPie[taskName].gameObject.activeSelf)
                        _taskNameToOrbPie[taskName].gameObject.SetActive(true);

                    _taskNameToOrbPie[taskName].SetTaskMessage(currentActiveTasks[taskName].CurrStepIndex,
                currentActiveTasks[taskName].Steps.Count,
                    currentActiveTasks[taskName].Steps[currentActiveTasks[taskName].CurrStepIndex].StepDesc, currentActiveTasks.Count >1);

                    float ratio = Mathf.Min(1, (float)currentActiveTasks[taskName].CurrStepIndex / (float)(currentActiveTasks[taskName].Steps.Count - 1));
                    _taskNameToOrbPie[taskName].UpdateCurrentTaskStatus(ratio);
                }
            }
            tempName = taskName.ToString();
        }

        // Only show the previous and next step at the orb if there is only one task
        if (_taskNameToOrbPie.Count==1)
        {
            _prevText.text = "";
            _nextText.text = "";
            int prevIndex = currentActiveTasks[tempName].PrevStepIndex;
            int nextIndex = currentActiveTasks[tempName].NextStepIndex;

            if (prevIndex>=0)
            {
                string previous = currentActiveTasks[tempName].Steps[prevIndex].StepDesc;
                _prevText.text = "<b>DONE:</b> " + previous;
            }
            if (nextIndex>=0 && nextIndex < currentActiveTasks[tempName].Steps.Count)
            {
                string next = currentActiveTasks[tempName].Steps[nextIndex].StepDesc;
                _nextText.text = "<b>Upcoming:</b> " + next;
            }

            if (currentActiveTasks[tempName].CurrStepIndex == currentActiveTasks[tempName].Steps.Count-1)
            {
                _nextText.text = "<b>Upcoming: All Done!</b> ";
            }
        }
        else
        {
            _prevText.text = "";
            _nextText.text = "";
        }

    }

    #region Warning

    public void AddWarning(string message, OrbFace face)
    {
        _currentWarning.SetMessage(message, ARUISettings.OrbNoteMaxCharCountPerLine);
        _currentWarning.gameObject.SetActive(true);
        _prevText.gameObject.SetActive(false);
        _nextText.gameObject.SetActive(false);

        face.UpdateNotification(IsWarningActive);
    }

    public void RemoveWarning(OrbFace face)
    {
        _currentWarning.SetMessage("", ARUISettings.OrbMessageMaxCharCountPerLine);
        _currentWarning.gameObject.SetActive(false);
        _prevText.gameObject.SetActive(true);
        _nextText.gameObject.SetActive(true);

        if (face)
            face.UpdateNotification(IsWarningActive);
    }

    #endregion

    #region Update UI

    /// <summary>
    /// Turn on or off message fading
    /// </summary>
    /// <param name="active"></param>
    public void SetFadeOutMessageContainer(bool active)
    {
        if (active)
        {
            StartCoroutine(FadeOutAllMessages());
        }
        else
        {
            StopCoroutine(FadeOutAllMessages());
            IsMessageFading = false;
        }
    }

    /// <summary>
    /// Fade out message from the moment the user does not look at the message anymore
    /// </summary>
    /// <returns></returns>
    private IEnumerator FadeOutAllMessages()
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

            foreach (OrbTask op in _taskNameToOrbPie.Values)
            {
                op.Text.BackgroundColor = new Color(shade, shade, shade, shade);
                op.SetTextAlpha(alpha);
                SetTextAlphaOthers(alpha);
            }

            yield return new WaitForEndOfFrame();
        }

        IsMessageFading = false;
        IsMessageContainerActive = !(shade <= 0);
    }

    private void SetTextAlphaOthers(float alpha)
    {
        if (alpha == 0)
        {
            _prevText.color = new Color(0, 0, 0, 0);
            _nextText.color = new Color(0, 0, 0, 0);
        }
        else
        {
            _prevText.color = new Color(_textColor.r, _textColor.g, _textColor.b, alpha);
            _nextText.color = new Color(_textColor.r, _textColor.g, _textColor.b, alpha);
        }
    }

    /// <summary>
    /// Updates the anchor of the messagebox smoothly
    /// </summary>
    /// <param name="MessageAnchor">The new anchor</param>
    private void UpdateAnchorLerp(bool shouldBeRight)
    {
        if (IsMessageLerping) return;

        if (shouldBeRight != _currentAlignmentIsRight)
        {
            IsMessageLerping = true;
            _currentAlignmentIsRight = shouldBeRight;

            StartCoroutine(MoveMessageBox(!_currentAlignmentIsRight, false));
        }
    }

    /// <summary>
    /// Updates the anchor of the messagebox instantly
    /// </summary>
    /// <param name="anchor"></param>
    public void UpdateAnchorInstant()
    {
        foreach (OrbTask ob in _taskNameToOrbPie.Values)
            ob.UpdateAnchor();

        StartCoroutine(MoveMessageBox(!_currentAlignmentIsRight, true));
    }

    public void ChangeAlignmentTo(MessageAlignment newAlignment)
    {
        _currentAlignment = newAlignment;

        StartCoroutine(DelayedLerping());
    }

    private IEnumerator DelayedLerping()
    {
        while (IsMessageLerping)
            yield return null;

        if (_currentAlignment.Equals(MessageAlignment.LockRight))
        {
            UpdateAnchorLerp(true);
        }
        else if (_currentAlignment.Equals(MessageAlignment.LockLeft))
        {
            UpdateAnchorLerp(false);
        }
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
            foreach (OrbTask op in _taskNameToOrbPie.Values)
            {
                float XOffset = op.InitialXOffset;
                if (isLeft)
                    XOffset = -op.Text.MessageCollider.size.x;
                float YOffset = op.Text.transform.localPosition.y;

                if (op.TaskType.Equals(TaskType.primary))
                {
                    if (isLeft)
                        op.transform.localPosition = Vector2.Lerp(op.transform.localPosition, new Vector3(-0.0342f, 0.034f, 0), step + Time.deltaTime);
                    else
                        op.transform.localPosition = Vector2.Lerp(op.transform.localPosition, new Vector3(0.0342f, 0.034f, 0), step + Time.deltaTime);

                    op.Text.transform.localPosition = Vector2.Lerp(op.Text.transform.localPosition, new Vector3(XOffset, YOffset, 0), step + Time.deltaTime);
                } else
                {
                    op.Text.transform.localPosition = Vector2.Lerp(op.Text.transform.localPosition, new Vector3(XOffset, YOffset, 0), step + Time.deltaTime);
                }
                step += Time.deltaTime;
            }

            float XOffsetWarning = _currentWarning.XOffset;
            if (isLeft)
            {
                XOffsetWarning = -_currentWarning.XOffset-0.25f;
                _currentWarning.Text.alignment = TMPro.TextAlignmentOptions.BottomRight;
            } else
                _currentWarning.Text.alignment = TMPro.TextAlignmentOptions.BottomLeft;

            _currentWarning.gameObject.transform.localPosition = Vector2.Lerp(_currentWarning.gameObject.transform.localPosition,
                                                                            new Vector3(XOffsetWarning, _currentWarning.gameObject.transform.localPosition.y, 0), step + Time.deltaTime);

            yield return new WaitForEndOfFrame();
        }

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
