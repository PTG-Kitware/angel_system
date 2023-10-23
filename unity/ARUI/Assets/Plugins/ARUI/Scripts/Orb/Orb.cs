using Microsoft.MixedReality.Toolkit.Input;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum OrbMovementBehavior
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
    private OrbMovementBehavior _orbBehavior = OrbMovementBehavior.Follow;                                   /// <the orb shape itself (part of prefab)
    public OrbMovementBehavior OrbBehavior
    {
        get => _orbBehavior;
    }

    ///** Reference to parts of the orb
    private OrbFace _face;                                   /// <the orb shape itself (part of prefab)
    public float MouthScale
    {
        get => _face.MouthScale;
        set => _face.MouthScale = value;
    }

    private OrbHandle _orbHandle;
    private OrbGrabbable _grabbable;                         /// <reference to grabbing behavior
    private OrbMessageContainer _messageContainer;                     /// <reference to orb message container (part of prefab)
    public OrbMessageContainer Message => _messageContainer;

    private List<BoxCollider> _allOrbColliders;              /// <reference to all collider - will be merged for view management.
    public List<BoxCollider> AllOrbColliders => _allOrbColliders;

    ///** Placement behaviors - overall, orb stays in the FOV of the user
    private OrbFollowerSolver _followSolver;

    ///** Flags
    private bool _isLookingAtOrb = false;                    /// <true if the user is currently looking at the orb shape or orb message
    private bool _lazyLookAtRunning = false;                 /// <used for lazy look at disable
    private bool _lazyFollowStarted = false;                 /// <used for lazy following

    private bool _guidanceIsActive = true;

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

        // Get handle object in orb prefab
        GameObject handleObj = transform.GetChild(0).GetChild(2).gameObject;
        _orbHandle = handleObj.AddComponent<OrbHandle>();

        // Get grabbable and following scripts
        _followSolver = gameObject.GetComponentInChildren<OrbFollowerSolver>();
        _grabbable = gameObject.GetComponentInChildren<OrbGrabbable>();

        BoxCollider taskListBtnCol = transform.GetChild(0).GetComponent<BoxCollider>();

        // Collect all orb colliders
        _allOrbColliders = new List<BoxCollider>();

        ListenToDataEvents();
    }

    /// <summary>
    /// Update visibility of orb based on eye evets and task manager.
    /// </summary>
    private void Update()
    {
        _orbHandle.IsActive = (_orbBehavior == OrbMovementBehavior.Fixed);
        _followSolver.IsPaused = (_orbBehavior == OrbMovementBehavior.Fixed || _face.UserIsGrabbing);

        float distance = Vector3.Distance(_followSolver.transform.position, AngelARUI.Instance.ARCamera.transform.position);
        if (distance > 0.8)
            _followSolver.transform.localScale = new Vector3(distance * 1.1f, distance * 1.1f, distance * 1.1f);
        else
            _followSolver.transform.localScale = new Vector3(1, 1, 1);

        if (_guidanceIsActive != AngelARUI.Instance.IsGuidanceActive)
            ToggleOrbGuidance();

        if (!_guidanceIsActive) return;

        // Update eye tracking flag
        if (_isLookingAtOrb && EyeGazeManager.Instance.CurrentHit != EyeTarget.orbFace)
            SetIsLookingAtFace(false);
        else if (!_isLookingAtOrb && EyeGazeManager.Instance.CurrentHit == EyeTarget.orbFace)
            SetIsLookingAtFace(true);

        if (_isLookingAtOrb || _messageContainer.IsLookingAtMessage)
            _face.MessageNotificationEnabled = false;

        if (DataProvider.Instance.CurrentSelectedTasks.Keys.Count > 0)
            UpdateMessageVisibility();
    }

    private void ToggleOrbGuidance()
    {
        _face.SetOrbGuidance(AngelARUI.Instance.IsGuidanceActive);
        _messageContainer.gameObject.SetActive(AngelARUI.Instance.IsGuidanceActive);
        _guidanceIsActive = AngelARUI.Instance.IsGuidanceActive;
    }

    #region Visibility, Position Updates and eye/collision event handler

    /// <summary>
    /// View management
    /// Update the visibility of the orb message based on eye gaze collisions with the orb collider 
    /// </summary>
    private void UpdateMessageVisibility()
    {
        if ((IsLookingAtOrb(false) && !_messageContainer.IsMessageContainerActive && !_messageContainer.IsMessageFading && !ManualManager.Instance.MenuActive))
        { //Set the message visible!
            _messageContainer.IsMessageContainerActive = true;
        }
        else if ( (ManualManager.Instance.MenuActive && _messageContainer.IsMessageContainerActive)
            || (!_messageContainer.IsLookingAtMessage && !IsLookingAtOrb(false) && _followSolver.IsOutOfFOV))
        {
            _messageContainer.IsMessageContainerActive = false;
        }
        else if ((_messageContainer.IsLookingAtMessage || IsLookingAtOrb(false)) && _messageContainer.IsMessageContainerActive && _messageContainer.IsMessageFading)
        { //Stop Fading, set the message visible
            _messageContainer.SetFadeOutMessage(false);
        }
        else if (!IsLookingAtOrb(false) && _messageContainer.IsMessageContainerActive && !_messageContainer.IsMessageFading
            && !_messageContainer.IsLookingAtMessage)
        { //Start Fading
            _messageContainer.SetFadeOutMessage(true);
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

    public void UpdateMovementbehavior(OrbMovementBehavior newBehavior)
    {
        _orbBehavior = newBehavior;
    }

    /// <summary>
    /// Called if input events with hand collider are detected
    /// </summary>
    /// <param name="isDragging"></param>
    public void SetIsDragging(bool isDragging)
    {
        _face.UserIsGrabbing = isDragging;

        if (_orbBehavior == OrbMovementBehavior.Follow)
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

    #endregion

    #region Task Messages and Notifications

    public void AddNotification(NotificationType type, string message)
    {
        _messageContainer.AddNotification(type, message, _face);

        AudioManager.Instance.PlaySound(_face.transform.position, SoundType.warning);
    }

    
    public void RemoveNotification(NotificationType type)
    {
        _messageContainer.RemoveNotification(type, _face);
        
    }

    /// <summary>
    /// Set the task messages the orb communicates, if 'message' is less than 2 char, the message is deactivated
    /// </summary>
    /// <param name="message"></param>
    private void SetTaskMessage(Dictionary<string, TaskList> currentSelectedTasks, string currentActiveTask)
    {
        _messageContainer.RemoveAllNotifications();
        _face.UpdateNotification(false,false);

        _messageContainer.SetTaskMessage(currentSelectedTasks, currentActiveTask);

        if (_allOrbColliders.Count == 0)
        {
            _allOrbColliders.Add(transform.GetChild(0).GetComponent<BoxCollider>());
            _allOrbColliders.AddRange(_messageContainer.GetAllColliders());
        }
    }

    #endregion

    #region Getter and Setter

    /// <summary>
    /// Detect hand hovering events
    /// </summary>
    /// <param name="isHovering"></param>
    public void SetNearHover(bool isHovering) => _face.UserIsGrabbing = isHovering;

    /// <summary>
    /// Change the visibility of the tasklist button
    /// </summary>
    /// <param name="isActive"></param>
    //public void SetTaskListButtonActive(bool isActive) => _messageContainer.TaskListToggle.gameObject.SetActive(isActive);

    /// <summary>
    /// Update the position behavior of the orb
    /// </summary>
    /// <param name="isSticky"></param>
    private void SetSticky(bool isSticky)
    {
        _followSolver.SetSticky(isSticky);

        if (isSticky)
            _messageContainer.IsMessageContainerActive = false;
    }

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

    #endregion

    #region Data Change Listeners

    /// <summary>
    /// Register events that happen in case the task data changes
    /// </summary>
    private void ListenToDataEvents()
    {
        DataProvider.Instance.RegisterDataSubscriber(() => HandleUpdateTaskListEvent(), SusbcriberType.UpdateTask);
        DataProvider.Instance.RegisterDataSubscriber(() => HandleUpdateActiveTaskEvent(), SusbcriberType.UpdateActiveTask);
        DataProvider.Instance.RegisterDataSubscriber(() => HandleUpdateActiveStepEvent(), SusbcriberType.UpdateStep);
    }

    /// <summary>
    /// Task List changed (add or removal of task)
    /// </summary>
    private void HandleUpdateTaskListEvent()
    {
        _messageContainer.HandleUpdateTaskListEvent(DataProvider.Instance.CurrentSelectedTasks, DataProvider.Instance.CurrentObservedTask);
    }

    /// <summary>
    /// Currently observed task changed. Update orb message
    /// </summary>
    private void HandleUpdateActiveTaskEvent()
    {
        if (DataProvider.Instance.CurrentSelectedTasks.Count > 0)
            _messageContainer.HandleUpdateActiveTaskEvent(DataProvider.Instance.CurrentSelectedTasks, DataProvider.Instance.CurrentObservedTask);
    }

    /// <summary>
    /// Current step for tasks changed. Update orb message
    /// </summary>
    private void HandleUpdateActiveStepEvent()
    {
        SetTaskMessage(DataProvider.Instance.CurrentSelectedTasks, DataProvider.Instance.CurrentObservedTask);
    }

    #endregion

}
