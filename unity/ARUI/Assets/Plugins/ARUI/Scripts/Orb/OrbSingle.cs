//using System;
//using System.Collections;
//using System.Collections.Generic;
//using UnityEngine;

///// <summary>
///// Represents the message container next to the orb
///// </summary>
//public class OrbSingle : OrbMessage
//{
//    public BoxCollider Collider => _textContainer.MessageCollider;

//    private Color32 _glowColor = Color.white;
//    private float _maxglowAlpha = 0.3f;
//    private Color _activeColorText = Color.white;

//    //*** Reference for notifications
//    private Notification _currentNote;
//    private Notification _currentWarning;

//    public bool IsNoteActive => _currentNote.IsSet;
//    public bool IsWarningActive => _currentWarning.IsSet;

//    private FlexibleTextContainer _textContainer;
//    private GameObject _indicator;
//    private Vector3 _initialIndicatorPos;
//    private float _initialmessageYOffset;

//    private TMPro.TextMeshProUGUI _progressText;

//    private TMPro.TextMeshProUGUI _prevText;
//    private TMPro.TextMeshProUGUI _nextText;

//    private DwellButton _taskListbutton;                     /// <reference to dwell btn above orb ('tasklist button')
//    public DwellButton TaskListToggle
//    {
//        get => _taskListbutton;
//    }

//    public override List<BoxCollider> GetAllColliders()
//    {
//        return new List<BoxCollider> { _taskListbutton.Collider, _textContainer.MessageCollider };
//    }

//    public override bool IsInteractingWithBtn() => TaskListToggle != null && TaskListToggle.IsInteractingWithBtn;

//    private bool _enalbed = false;
//    public override void SetEnabled(bool enabled)
//    {
//        _enalbed = enabled;
//        _textContainer.gameObject.SetActive(enabled);
//        _taskListbutton.gameObject.SetActive(enabled);
//        _indicator.gameObject.SetActive(enabled);
//    }

//    /// <summary>
//    /// Init component, get reference to gameobjects from children
//    /// </summary>
//    public override void InitializeComponents()
//    {
//        messageType = OrbMessageType.single;

//        _textContainer = transform.GetChild(0).gameObject.AddComponent<FlexibleTextContainer>();
//        _textContainer.gameObject.name += "_orb";

//        TMPro.TextMeshProUGUI[] allText = _textContainer.AllTextMeshComponents;

//        _progressText = allText[1].gameObject.GetComponent<TMPro.TextMeshProUGUI>();
//        _progressText.text = "";

//        _prevText = _textContainer.VGroup.GetChild(1).gameObject.GetComponentInChildren<TMPro.TextMeshProUGUI>();
//        _prevText.text = "";
//        _nextText = _textContainer.VGroup.GetChild(2).gameObject.GetComponentInChildren<TMPro.TextMeshProUGUI>();
//        _nextText.text = "";

//        _currentWarning = _textContainer.VGroup.GetChild(3).gameObject.AddComponent<Notification>();
//        _currentWarning.init(NotificationType.warning, "", _textContainer.TextRect.height);

//        _currentNote = _textContainer.VGroup.GetChild(4).gameObject.AddComponent<Notification>();
//        _currentNote.init(NotificationType.note, "", _textContainer.TextRect.height);

//        _initialmessageYOffset = _textContainer.transform.position.x;

//        //message direction indicator
//        _indicator = gameObject.GetComponentInChildren<Shapes.Polyline>().gameObject;
//        _initialIndicatorPos = _indicator.transform.position;

//        _glowColor = _textContainer.GlowColor;

//        // Init tasklist button
//        GameObject taskListbtn = transform.GetChild(2).gameObject;
//        _taskListbutton = taskListbtn.AddComponent<DwellButton>();
//        _taskListbutton.gameObject.name += "FacetasklistButton";
//        _taskListbutton.InitializeButton(EyeTarget.orbtasklistButton, () => ToggleOrbTaskList(),
//            null, true, DwellButtonType.Select);

//        SetIsActive(false, false);
//    }

//    private new void Update()
//    {
//        base.Update();
//        _currentNote.UpdateSize(_textContainer.TextRect.width / 2);
//        _currentWarning.UpdateSize(_textContainer.TextRect.width / 2);

//        _currentNote.UpdateYPos(_textContainer.TextRect.height, _prevText.gameObject.activeSelf);
//        _currentWarning.UpdateYPos(_textContainer.TextRect.height, _prevText.gameObject.activeSelf);

//        _prevText.gameObject.transform.SetLocalYPos(_textContainer.TextRect.height);
//        _nextText.gameObject.transform.SetLocalYPos(-_textContainer.TextRect.height);

//        if (!(IsMessageVisible && _textContainer.gameObject.activeSelf) || IsMessageLerping) return;

//        // Update messagebox anchor
//        if (ChangeMessageBoxToRight(100))
//            UpdateAnchorLerp(MessageAnchor.right);

//        else if (ChangeMessageBoxToLeft(100))
//            UpdateAnchorLerp(MessageAnchor.left);
//    }

//    #region Message and Notification Updates


//    public override void AddNotification(NotificationType type, string message, OrbFace face)
//    {
//        if (type.Equals(NotificationType.note))
//            _currentNote.SetMessage(message, ARUISettings.OrbNoteMaxCharCountPerLine);

//        else if (type.Equals(NotificationType.warning))
//            _currentWarning.SetMessage(message, ARUISettings.OrbMessageMaxCharCountPerLine);

//        SetOrbListActive(false);
//        _taskListbutton.IsDisabled = true;

//        face.UpdateNotification(IsWarningActive, IsNoteActive);
//    }

//    public override void RemoveNotification(NotificationType type, OrbFace face)
//    {
//        if (type.Equals(NotificationType.note))
//            _currentNote.SetMessage("", ARUISettings.OrbMessageMaxCharCountPerLine);
//        else if (type.Equals(NotificationType.warning))
//            _currentWarning.SetMessage("", ARUISettings.OrbMessageMaxCharCountPerLine);

//        SetOrbListActive(!(IsNoteActive || IsWarningActive));
//        _taskListbutton.IsDisabled = (IsNoteActive || IsWarningActive);

//        face.UpdateNotification(IsWarningActive, IsNoteActive);
//    }


//    public override void RemoveAllNotifications()
//    {
//        _currentNote.SetMessage("", ARUISettings.OrbMessageMaxCharCountPerLine);
//        _currentWarning.SetMessage("", ARUISettings.OrbMessageMaxCharCountPerLine);

//        SetOrbListActive(true);
//        _taskListbutton.IsDisabled = false;
//    }

//    /// <summary>
//    /// Turn on or off message fading
//    /// </summary>
//    /// <param name="active"></param>
//    public override void SetFadeOutMessage(bool active)
//    {
//        if (active)
//        {
//            StartCoroutine(FadeOutMessage());
//        }
//        else
//        {
//            StopCoroutine(FadeOutMessage());
//            IsMessageFading = false;
//            _textContainer.BackgroundColor = ARUISettings.OrbMessageBGColor;

//            SetTextAlpha(1f);
//        }
//    }

//    /// <summary>
//    /// Fade out message from the moment the user does not look at the message anymore
//    /// </summary>
//    /// <returns></returns>
//    private IEnumerator FadeOutMessage()
//    {
//        float fadeOutStep = 0.001f;
//        IsMessageFading = true;

//        yield return new WaitForSeconds(1.0f);

//        float shade = ARUISettings.OrbMessageBGColor.r;
//        float alpha = 1f;

//        while (IsMessageFading && shade > 0)
//        {
//            alpha -= (fadeOutStep * 20);
//            shade -= fadeOutStep;

//            if (alpha < 0)
//                alpha = 0;
//            if (shade < 0)
//                shade = 0;

//            _textContainer.BackgroundColor = new Color(shade, shade, shade, shade);
//            SetTextAlpha(alpha);

//            yield return new WaitForEndOfFrame();
//        }

//        IsMessageFading = false;

//        if (shade <= 0)
//        {
//            SetIsActive(false, false);
//            IsMessageVisible = false;
//        }
//    }


//    private IEnumerator FadeNewTaskGlow()
//    {
//        SetFadeOutMessage(false);

//        UserHasSeenNewStep = false;

//        _textContainer.GlowColor = new Color(_glowColor.r, _glowColor.g, _glowColor.b, _maxglowAlpha);

//        while (!IsLookingAtMessage)
//        {
//            yield return new WaitForEndOfFrame();
//        }

//        float step = (_maxglowAlpha / 10);
//        float current = _maxglowAlpha;
//        while (current > 0)
//        {
//            current -= step;
//            _textContainer.GlowColor = new Color(_glowColor.r, _glowColor.g, _glowColor.b, current);
//            yield return new WaitForSeconds(0.1f);
//        }

//        _textContainer.GlowColor = new Color(_glowColor.r, _glowColor.g, _glowColor.b, 0f);

//        UserHasSeenNewStep = true;
//    }

//    #endregion


//    #region Position Updates

//    /// <summary>
//    /// Updates the anchor of the messagebox smoothly
//    /// </summary>
//    /// <param name="MessageAnchor">The new anchor</param>
//    private void UpdateAnchorLerp(MessageAnchor newMessageAnchor)
//    {
//        if (IsMessageLerping) return;

//        if (newMessageAnchor != CurrentAnchor)
//        {
//            IsMessageLerping = true;
//            CurrentAnchor = newMessageAnchor;
//            UpdateBoxIndicatorPos();

//            StartCoroutine(MoveMessageBox(_initialmessageYOffset, newMessageAnchor != MessageAnchor.right, false));
//        }
//    }

//    /// <summary>
//    /// Updates the anchor of the messagebox instantly (still need to run coroutine to allow the Hgroup rect to update properly
//    /// </summary>
//    private void UpdateAnchorInstant()
//    {
//        _textContainer.UpdateAnchorInstant();

//        bool isLeft = false;
//        if (ChangeMessageBoxToLeft(0))
//        {
//            CurrentAnchor = MessageAnchor.left;
//            isLeft = true;
//        }
//        else
//            CurrentAnchor = MessageAnchor.right;

//        UpdateBoxIndicatorPos();
//        StartCoroutine(MoveMessageBox(_initialmessageYOffset, isLeft, true));
//    }

//    /// <summary>
//    /// Updates the position and orientation of the messagebox indicator
//    /// </summary>
//    private void UpdateBoxIndicatorPos()
//    {
//        if (CurrentAnchor == MessageAnchor.right)
//        {
//            _indicator.transform.localPosition = new Vector3(_initialIndicatorPos.x, 0, 0);
//            _indicator.transform.localRotation = Quaternion.identity;
//        }
//        else
//        {
//            _indicator.transform.localPosition = new Vector3(-_initialIndicatorPos.x, 0, 0);
//            _indicator.transform.localRotation = Quaternion.Euler(0, 180, 0);
//        }
//    }

//    /// <summary>
//    /// Lerps the message box to the other side
//    /// </summary>
//    /// <param name="YOffset">y offset of the message box to the orb prefab</param>
//    /// <param name="addWidth"> if messagebox on the left, change the signs</param>
//    /// <param name="instant">if lerp should be almost instant (need to do this in a coroutine anyway, because we are waiting for the Hgroup to update properly</param>
//    /// <returns></returns>
//    IEnumerator MoveMessageBox(float YOffset, bool isLeft, bool instant)
//    {
//        float initialYOffset = YOffset;
//        float step = 0.1f;

//        if (instant)
//            step = 0.5f;

//        while (step < 1)
//        {
//            if (isLeft)
//                YOffset = -initialYOffset - _textContainer.MessageCollider.size.x;

//            _textContainer.transform.localPosition = Vector2.Lerp(_textContainer.transform.localPosition, new Vector3(YOffset, 0, 0), step += Time.deltaTime);
//            step += Time.deltaTime;

//            yield return new WaitForEndOfFrame();
//        }

//        if (isLeft)
//            _taskListbutton.transform.SetLocalXPos(-0.043f);
//        else
//            _taskListbutton.transform.SetLocalXPos(0.043f);

//        IsMessageLerping = false;
//    }

//    private void ToggleOrbTaskList() => SetOrbListActive(!_prevText.gameObject.activeSelf);

//    #endregion

//    #region Getter and Setter

//    /// <summary>
//    /// Actives or disactivates the messagebox of the orb in the hierarchy
//    /// </summary>
//    /// <param name="active"></param>
//    public override void SetIsActive(bool active, bool newTask)
//    {
//        _textContainer.gameObject.SetActive(active);
//        _indicator.SetActive(active);

//        if (active)
//        {
//            UpdateAnchorInstant();
//            _textContainer.BackgroundColor = ARUISettings.OrbMessageBGColor;
//            SetTextAlpha(1f);
//        }
//        else
//            IsMessageFading = false;

//        IsMessageVisible = active;

//        if (newTask)
//        {
//            StartCoroutine(FadeNewTaskGlow());
//            RemoveAllNotifications();
//        }

//        _taskListbutton.gameObject.SetActive(active);
//    }


//    /// <summary>
//    /// Sets the orb task message to the given message and adds line break based on maxCharCountPerLine
//    /// TODO
//    /// </summary>
//    /// <param name="message"></param>
//    public override string SetTaskMessage(TaskList currentTask)
//    {
//        int stepCount = currentTask.Steps.Count;
//        int currentID = currentTask.CurrStepIndex;
//        int previousID = currentTask.PrevStepIndex;
//        int nextID = currentTask.NextStepIndex;
//        string previousMessage = "";
//        string nextMessage = "";
//        string currentMessage = "All Done";
//        if (currentID < stepCount)
//            currentMessage = currentTask.Steps[currentID].StepDesc;

//        if (previousID < stepCount && previousID != currentID)
//        {
//            if (currentID >= 1)
//                previousMessage = currentTask.Steps[previousID].StepDesc;

//            if (currentID + 1 < stepCount && nextID != currentID)
//                nextMessage = UpdateAnchorInstant();currentTask.Steps[nextID].StepDesc;
//        }

//        //Update UI
//        _textContainer.Text = currentMessage;
//        _progressText.text = (currentID + 1) + "/" + stepCount;

//        _prevText.text = "";
//        _nextText.text = "";
//        if (previousMessage.Length > 0)
//            _prevText.text = "<b>DONE:</b> " + Utils.SplitTextIntoLines(previousMessage, ARUISettings.OrbMessageMaxCharCountPerLine);

//        if (nextMessage.Length > 0)
//            _nextText.text = "<b>Upcoming:</b> " + Utils.SplitTextIntoLines(nextMessage, ARUISettings.OrbNoteMaxCharCountPerLine);

//        UpdateAnchorInstant();

//        _progressText.gameObject.SetActive(!currentMessage.Contains("Done"));

//        return currentMessage;
//    }

//    private void SetOrbListActive(bool active)
//    {
//        _prevText.gameObject.SetActive(active);
//        _nextText.gameObject.SetActive(active);

//        if (active)
//        {
//            _textContainer.MessageCollider.size = new Vector3(_textContainer.MessageCollider.size.x, 0.08f, _textContainer.MessageCollider.size.z);
//        }
//        else
//        {
//            _textContainer.MessageCollider.size = new Vector3(_textContainer.MessageCollider.size.x, 0.05f, _textContainer.MessageCollider.size.z);
//        }
//    }

//    /// <summary>
//    /// Update the color of the text based on visibility
//    /// </summary>
//    /// <param name="alpha"></param>
//    private void SetTextAlpha(float alpha)
//    {
//        if (alpha == 0)
//            _textContainer.TextColor = new Color(0, 0, 0, 0);
//        else
//            _textContainer.TextColor = new Color(_activeColorText.r, _activeColorText.g, _activeColorText.b, alpha);
//    }

//    public override void UpdateTaskList(Dictionary<string, TaskList> currentSelectedTasks) { }


//    #endregion
//}