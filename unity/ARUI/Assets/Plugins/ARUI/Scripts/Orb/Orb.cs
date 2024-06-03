using Microsoft.MixedReality.OpenXR;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

public enum MovementBehavior
{
    Follow = 0,
    Fixed = 1,
}

/// <summary>
/// Represents a virtual assistant in the shape of an orb, staying in the FOV of the user and
/// guiding the user through a sequence of tasks
/// </summary>
public class Orb : Singleton<Orb>
{
    ///** Reference to parts of the orb
    private MovementBehavior _orbBehavior = MovementBehavior.Follow;                                   /// <the orb shape itself (part of prefab)
    public MovementBehavior OrbBehavior
    {
        get => _orbBehavior;
    }
    private Vector3 _lastFixedPosition = Vector3.zero;

    ///** Reference to parts of the orb
    private OrbFace _face;                                   /// <the orb shape itself (part of prefab)
    public float MouthScale
    {
        get => _face.MouthScale;
        set => _face.MouthScale = value;
    }

    private OrbGrabbable _grabbable;                         /// <reference to grabbing behavior
    private OrbMessageContainer _messageContainer;                     /// <reference to orb message container (part of prefab)

    private List<BoxCollider> _allOrbColliders;              /// <reference to all collider - will be merged for view management.
    public List<BoxCollider> AllOrbColliders => _allOrbColliders;

    ///** Placement behaviors - overall, orb stays in the FOV of the user
    private OrbFollowerSolver _followSolver;
    public Transform orbTransform
    {
        get => _followSolver.transform;
    }
    private GameObject _eyeGazeTargetBody;

    ///** Flags
    private bool _isLookingAtOrb = false;                    /// <true if the user is currently looking at the orb shape or orb message
    private bool _lazyLookAtRunning = false;                 /// <used for lazy look at disable
    private bool _lazyFollowStarted = false;                 /// <used for lazy following

    private OrbSpeechBubble _dialogue;
    private OrbHandle _orbHandle;

    /// <summary>
    /// Get all orb references from prefab
    /// </summary>
    private void Awake()
    {
        gameObject.name = "***ARUI-Orb";
        _face = transform.GetChild(0).GetChild(0).gameObject.AddComponent<OrbFace>();

        // Get message object in orb prefab
        GameObject messageObj = transform.GetChild(0).GetChild(1).gameObject;
        _messageContainer = messageObj.AddComponent<OrbMessageContainer>();
        _messageContainer.InitializeComponents();

        // Get grabbable and following scripts
        _followSolver = gameObject.GetComponentInChildren<OrbFollowerSolver>();
        _eyeGazeTargetBody = _followSolver.gameObject;
        EyeGazeManager.Instance.RegisterEyeTargetID(_eyeGazeTargetBody);

        _grabbable = gameObject.GetComponentInChildren<OrbGrabbable>();

        BoxCollider taskListBtnCol = transform.GetChild(0).GetComponent<BoxCollider>();

        _dialogue = transform.GetChild(0).GetChild(2).gameObject.AddComponent<OrbSpeechBubble>();
        _dialogue.Init();
        _dialogue.SetText("", "Hello there! How can I help you?", 0);
        _dialogue.gameObject.SetActive(false);

        GameObject handleObj = transform.GetChild(0).GetChild(3).gameObject;
        _orbHandle = handleObj.AddComponent<OrbHandle>();

        // Collect all orb colliders
        _allOrbColliders = new List<BoxCollider>();

        ListenToDataEvents();
    }

    /// <summary>
    /// Update visibility of orb based on eye evets and task manager.
    /// </summary>
    private void Update()
    {
        // Update eye tracking flag
        if (_isLookingAtOrb && EyeGazeManager.Instance.CurrentHitID != _eyeGazeTargetBody.GetInstanceID())
            SetIsLookingAtFace(false);
        else if (!_isLookingAtOrb && EyeGazeManager.Instance.CurrentHitID == _eyeGazeTargetBody.GetInstanceID())
            SetIsLookingAtFace(true);

        if (_isLookingAtOrb || _messageContainer.IsLookingAtMessage)
            _face.MessageNotificationEnabled = false;

        _followSolver.IsPaused = (_orbBehavior == MovementBehavior.Fixed || _face.UserIsGrabbing);

        float distance = Vector3.Distance(_followSolver.transform.position, AngelARUI.Instance.ARCamera.transform.position);
        float scaleValue = Mathf.Max(1f, distance * 1.2f);
        _followSolver.transform.localScale = new Vector3(scaleValue, scaleValue, scaleValue);

        if (DataProvider.Instance.CurrentActiveTasks.Keys.Count > 0)
            UpdateMessageVisibility();
    }

    #region Visibility, Position Updates and eye/collision event handler

    /// <summary>
    /// View management
    /// Update the visibility of the orb message based on eye gaze collisions with the orb collider 
    /// </summary>
    private void UpdateMessageVisibility()
    {
        if ((IsLookingAtOrb(false) && !_messageContainer.IsMessageContainerActive && !_messageContainer.IsMessageFading) 
            || _orbBehavior.Equals(MovementBehavior.Fixed))
        { //Set the message visible!
            _messageContainer.SetFadeOutMessageContainer(false);
            _messageContainer.IsMessageContainerActive = true;
        }
        else if (!_messageContainer.IsLookingAtMessage && !IsLookingAtOrb(false) && _followSolver.IsOutOfFOV)
        {
            _messageContainer.IsMessageContainerActive = false;
        }
        else if ((_messageContainer.IsLookingAtMessage || IsLookingAtOrb(false)) && _messageContainer.IsMessageContainerActive && _messageContainer.IsMessageFading)
        { //Stop Fading
            _messageContainer.SetFadeOutMessageContainer(false);
        }
        else if (!IsLookingAtOrb(false) && _messageContainer.IsMessageContainerActive && !_messageContainer.IsMessageFading
            && !_messageContainer.IsLookingAtMessage)
        { //Start Fading
            _messageContainer.SetFadeOutMessageContainer(true);
        }
    } 

    /// <summary>
    /// If the user drags the orb, the orb will stay in place until it will be out of FOV
    /// </summary>
    private IEnumerator EnableLazyFollow()
    {
        _lazyFollowStarted = true;

        yield return new WaitForEndOfFrame();

        _followSolver.IsPaused = (true);

        while (_grabbable.transform.position.InFOV(AngelARUI.Instance.ARCamera))
            yield return new WaitForSeconds(0.1f);

        _followSolver.IsPaused = (false);
        _lazyFollowStarted = false;
    }

    /// <summary>
    /// Make sure that fast eye movements are not detected as dwelling
    /// </summary>
    /// <returns></returns>
    private IEnumerator StartLazyLookAt()
    {
        yield return new WaitForSeconds(0.2f);

        if (_lazyLookAtRunning)
        {
            _isLookingAtOrb = true;
            _lazyLookAtRunning = false;
            _face.UserIsLooking = true;
        }
    }

    /// <summary>
    /// TODO - 
    /// </summary>
    /// <param name="newBehavior"></param>
    public void UpdateMovementbehavior(MovementBehavior newBehavior)
    {
        _orbBehavior = newBehavior;

        if (newBehavior==MovementBehavior.Fixed) 
            Orb.Instance.SetHandleProgress(1);
        else
        {
            Orb.Instance.SetHandleProgress(0);
            _lastFixedPosition = _followSolver.transform.position;
        }
            
    }

    /// <summary>
    /// Called if input events with hand collider are detected
    /// </summary>
    /// <param name="isDragging"></param>
    public void SetIsDragging(bool isDragging)
    {
        _face.UserIsGrabbing = isDragging;

        if (_orbBehavior == MovementBehavior.Follow)
        {
            if (!isDragging && !_lazyFollowStarted)
                StartCoroutine(EnableLazyFollow());

            if (isDragging && _lazyFollowStarted)
            {
                StopCoroutine(EnableLazyFollow());

                _lazyFollowStarted = false;
                _followSolver.IsPaused = (false);
            }
        }
    }

    /// <summary>
    /// Called if changes in eye events are detected
    /// </summary>
    /// <param name="isLooking"></param>
    private void SetIsLookingAtFace(bool isLooking)
    {
        if (isLooking && !_lazyLookAtRunning)
        {
            _lazyLookAtRunning = true;
            StartCoroutine(StartLazyLookAt());
        }
        else if (!isLooking)
        {
            if (_lazyLookAtRunning)
                StopCoroutine(StartLazyLookAt());

            _isLookingAtOrb = false;
            _lazyLookAtRunning = false;
            _face.UserIsLooking= false;
        }
    }

    /// <summary>
    /// Moves the orb in front of the user
    /// </summary>
    /// <exception cref="NotImplementedException"></exception>
    public void MoveToUser() {

        if (_orbBehavior == MovementBehavior.Fixed)
        {
            UpdateMovementbehavior(MovementBehavior.Follow);
        }

        _followSolver.MoveToCenter();
        AudioManager.Instance.PlayTextIfNotPlaying("mhm");
    }

    
    /// <summary>
    /// 
    /// </summary>
    public void MoveToLastFixedLocation()
    {
        if (_lastFixedPosition.Equals(Vector3.zero)) return;

        _followSolver.transform.position = _lastFixedPosition;
        _grabbable.TransitionToFixedMovement();
    }

    #endregion

    #region Task Message, Warning and Notifications

    public void AddWarning(string message)
    {
        _messageContainer.AddWarning(message, _face);
        AudioManager.Instance.PlaySound(_face.transform.position, SoundType.warning);
    }
    
    public void RemoveWarning() => _messageContainer.RemoveWarning(_face);

    /// <summary>
    /// Set the task messages the orb communicates, if 'message' is less than 2 char, the message is deactivated
    /// </summary>
    /// <param name="message"></param>
    private void SetTaskMessage(Dictionary<string, TaskList> currentActiveTasks)
    {
        _messageContainer.UpdateAllTaskMessages(currentActiveTasks);

        if (_allOrbColliders.Count == 0)
        {
            _allOrbColliders.Add(transform.GetChild(0).GetComponent<BoxCollider>());
        }
    }

    public void TryGetUserConfirmation(string msg, List<UnityAction> actionOnConfirmation, UnityAction actionOnTimeOut, float timeout)
    {
        _messageContainer.TryGetUserConfirmation(msg, actionOnConfirmation, actionOnTimeOut, timeout);
    }

    public void TryGetUserChoice(string selectionMsg, List<string> choices, List<UnityAction> actionOnSelection, UnityAction actionOnTimeOut, float timeout)
    {
        _messageContainer.TryGetUserChoice(selectionMsg,choices, actionOnSelection, actionOnTimeOut, timeout);
    }

    public void TryGetUserYesNoChoice(string selectionMsg, UnityAction actionOnYes, UnityAction actionOnNo, UnityAction actionOnTimeOut, float timeout)
    {
        _messageContainer.TryGetUserYesNoChoice(selectionMsg, actionOnYes, actionOnNo, actionOnTimeOut, timeout);
    }

    #endregion

    #region Getter and Setter

    /// <summary>
    /// Detect hand hovering events
    /// </summary>
    /// <param name="isHovering"></param>
    public void SetNearHover(bool isHovering) => _face.UserIsGrabbing = isHovering;

    /// <summary>
    /// Update the position behavior of the orb
    /// </summary>
    /// <param name="isSticky"></param>
    public void SetSticky(bool isSticky)
    {
        _followSolver.SetSticky(isSticky);

        if (isSticky && _messageContainer.IsMessageContainerActive && !_messageContainer.IsMessageFading)
            _messageContainer.SetFadeOutMessageContainer(true);
    }

    /// <summary>
    /// If true, changes the visual appearance to the agent to a 'thinking' state, else idle
    /// </summary>
    /// <param name="isThinking"></param>
    public void SetOrbThinking(bool isThinking)
    {
        if (isThinking)
            _face.SetOrbState(OrbStates.Loading);
        else
            _face.SetOrbState(OrbStates.Idle);
    }

    /// <summary>
    /// Show the user dialogue at the orb active or not
    /// </summary>
    /// <param name="isActive"></param>
    public void SetDialogueActive(bool isActive) => _dialogue.gameObject.SetActive(isActive);

    /// <summary>
    /// Change the dialogue message at the orb. If 'utterance' is an empty string, only the answer is shown.
    /// </summary>
    /// <param name="utterance"></param>
    /// <param name="answer"></param>
    public void SetDialogueText(string utterance, string answer, float timeout) => _dialogue.SetText(utterance, answer, timeout);

    /// <summary>
    /// Change the visual appearance of the orb handle. 0% is black, 100% progress is white
    /// </summary>
    /// <param name="progress"></param>
    public void SetHandleProgress(float progress) => _orbHandle.SetHandleProgress(progress);

    /// <summary>
    /// Check if user is looking at orb. - includes orb message and task list button if 'any' is true. else only orb face and message
    /// </summary>
    /// <param name="any">if true, subobjects of orb are inlcluded, else only face and message</param>
    /// <returns></returns>
    public bool IsLookingAtOrb(bool any)
    {
        if (any)
            return _isLookingAtOrb || _messageContainer.IsLookingAtMessage || _messageContainer.IsInteractingWithBtn;
        else
            return _isLookingAtOrb || _messageContainer.IsLookingAtMessage;
    }

    /// <summary>
    /// Change alignment of message container next to orb. 
    /// </summary>
    /// <param name="newAlignment"></param>
    public void SetMessageAlignmentTo(MessageAlignment newAlignment) => _messageContainer.ChangeAlignmentTo(newAlignment);

    #endregion

    #region Data Change Listeners

    /// <summary>
    /// Register events that happen in case the task data changes
    /// </summary>
    private void ListenToDataEvents()
    {
        DataProvider.Instance.RegisterDataSubscriber(() => HandleUpdateTaskListEvent(), SusbcriberType.TaskListChanged);
        DataProvider.Instance.RegisterDataSubscriber(() => HandleUpdateActiveTaskEvent(), SusbcriberType.ObservedTaskChanged);
        DataProvider.Instance.RegisterDataSubscriber(() => HandleUpdateActiveStepEvent(), SusbcriberType.CurrentStepChanged);
    }

    /// <summary>
    /// Task List changed (add or removal of task)
    /// </summary>
    private void HandleUpdateTaskListEvent()
    {
        _messageContainer.HandleUpdateTaskListEvent(DataProvider.Instance.CurrentActiveTasks, DataProvider.Instance.CurrentObservedTask);
    }

    /// <summary>
    /// Currently observed task changed. Update orb message
    /// </summary>
    private void HandleUpdateActiveTaskEvent()
    {
        if (DataProvider.Instance.CurrentActiveTasks.Count > 0)
            _messageContainer.HandleUpdateActiveTaskEvent(DataProvider.Instance.CurrentActiveTasks, DataProvider.Instance.CurrentObservedTask);
    }

    /// <summary>
    /// Current step for tasks changed. Update orb message
    /// </summary>
    private void HandleUpdateActiveStepEvent()
    {
        SetTaskMessage(DataProvider.Instance.CurrentActiveTasks);
    }

    #endregion

}
