using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public enum MessageAnchor
{
    left = 1, //message is left from the orb
    right = 2, //message is right from the orb
}

public class OrbMessageContainer : MonoBehaviour
{
    private List<OrbTask> _allTasksPlaceholder = new List<OrbTask>();

    //** OrbTasks after manual is set
    private OrbTask _mainTaskPiePlace;
    private Dictionary<string, OrbTask> _taskNameToOrbPie;

    //** Layout
    private MessageAnchor _currentAnchor = MessageAnchor.right;

    //** States
    private bool _isLookingAtMessage = false;
    public bool IsLookingAtMessage
    {
        get { return _isLookingAtMessage; }
    }

    private Notification _currentNote;
    public bool IsNoteActive => _currentNote.IsSet;

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

    public List<BoxCollider> AllColliders
    {
        get 
        {
            var pieColliders = new List<BoxCollider>();
            foreach (OrbTask pie in _taskNameToOrbPie.Values)
                pieColliders.AddRange(pie.GetComponentsInChildren<BoxCollider>());

            return pieColliders;
        }

    }

    /// <summary>
    /// Init component, get reference to gameobjects from children
    /// </summary>
    public void InitializeComponents()
    {
        // Init tasklist button
        GameObject taskListbtn = transform.GetChild(0).gameObject;
        _taskListbutton = taskListbtn.AddComponent<DwellButton>();
        _taskListbutton.gameObject.name += "FacetasklistButton";
        _taskListbutton.InitializeButton(EyeTarget.orbtasklistButton, () => MultiTaskList.Instance.ToggleOverview(), null, true, DwellButtonType.Toggle);

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
        _currentNote = transform.GetChild(2).GetChild(0).gameObject.AddComponent<Notification>();
        _currentNote.init(NotificationType.warning, "", 0);
        _currentNote.gameObject.SetActive(false);
    }

    public void Update()
    {
        // Update eye tracking flag
        if (_isLookingAtMessage && EyeGazeManager.Instance.CurrentHit != EyeTarget.orbMessage
            && EyeGazeManager.Instance.CurrentHit != EyeTarget.orbtasklistButton && EyeGazeManager.Instance.CurrentHit != EyeTarget.pieCollider
            )
            _isLookingAtMessage = false;

        else if (!_isLookingAtMessage && (EyeGazeManager.Instance.CurrentHit == EyeTarget.orbMessage
            || EyeGazeManager.Instance.CurrentHit == EyeTarget.orbtasklistButton || EyeGazeManager.Instance.CurrentHit == EyeTarget.pieCollider))
            
            _isLookingAtMessage = true;

        if (!IsMessageContainerActive || IsMessageLerping) return;

        // Update messagebox anchor
        if (ChangeMessageBoxToRight(100))
            UpdateAnchorLerp(MessageAnchor.right);

        else if (ChangeMessageBoxToLeft(100))
            UpdateAnchorLerp(MessageAnchor.left);
    }

    public void HandleUpdateActiveTaskEvent(Dictionary<string, TaskList> currentSelectedTasks, string currentTaskID)
    {
        //foreach (OrbTask pie in _taskNameToOrbPie.Values)
        //{
        //    float ratio = (float)currentSelectedTasks[pie.TaskName].CurrStepIndex / (float)(currentSelectedTasks[pie.TaskName].Steps.Count - 1);
        //    pie.UpdateCurrentTaskStatus(ratio);
        //}

        HandleUpdateTaskListEvent(currentSelectedTasks, currentTaskID);
    }

    public void HandleUpdateTaskListEvent(Dictionary<string, TaskList> currentSelectedTasks, string currentTaskID)
    {
        if (currentSelectedTasks.Count == 0 || currentSelectedTasks.Count > 5) return;

        foreach (OrbTask pie in _taskNameToOrbPie.Values)
            pie.ResetPie();

        _taskNameToOrbPie = new Dictionary<string, OrbTask>();

        int pieIndex = 0;
        foreach (string taskName in currentSelectedTasks.Keys)
        {
            if (taskName.Equals(currentTaskID))
            {
                _taskNameToOrbPie.Add(taskName, _mainTaskPiePlace);
                _mainTaskPiePlace.TaskName = currentSelectedTasks[taskName].Name;
            }
            else
            {
                _taskNameToOrbPie.Add(taskName, _allTasksPlaceholder[pieIndex]); //assign task to pie
                _allTasksPlaceholder[pieIndex].TaskName = currentSelectedTasks[taskName].Name;
                pieIndex++;
            }
        }

        UpdateAllTaskMessages(currentSelectedTasks);
    }

    public void UpdateAllTaskMessages(Dictionary<string, TaskList> currentSelectedTasks)
    {
        UpdateAnchorInstant(_currentAnchor);

        foreach (string taskName in currentSelectedTasks.Keys)
        {
            if (_taskNameToOrbPie.ContainsKey(taskName))
            {
                if (currentSelectedTasks[taskName].CurrStepIndex >= currentSelectedTasks[taskName].Steps.Count)
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

                    _taskNameToOrbPie[taskName].SetTaskMessage(currentSelectedTasks[taskName].CurrStepIndex,
                currentSelectedTasks[taskName].Steps.Count,
                    currentSelectedTasks[taskName].Steps[currentSelectedTasks[taskName].CurrStepIndex].StepDesc);

                    float ratio = Mathf.Min(1, (float)currentSelectedTasks[taskName].CurrStepIndex / (float)(currentSelectedTasks[taskName].Steps.Count - 1));
                    _taskNameToOrbPie[taskName].UpdateCurrentTaskStatus(ratio);
                }
            }
        }
    }

    #region Notification

    public void AddNotification(string message, OrbFace face)
    {
        _currentNote.SetMessage(message, ARUISettings.OrbNoteMaxCharCountPerLine);
        _currentNote.gameObject.SetActive(true);
        face.UpdateNotification(IsNoteActive);
    }

    public void RemoveNotification(OrbFace face)
    {
        _currentNote.SetMessage("", ARUISettings.OrbMessageMaxCharCountPerLine);
        _currentNote.gameObject.SetActive(false);
        if (face)
            face.UpdateNotification(IsNoteActive);
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
            }

            yield return new WaitForEndOfFrame();
        }

        IsMessageFading = false;
        IsMessageContainerActive = !(shade <= 0);
    }

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
        foreach (OrbTask ob in _taskNameToOrbPie.Values)
            ob.UpdateAnchor();

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
