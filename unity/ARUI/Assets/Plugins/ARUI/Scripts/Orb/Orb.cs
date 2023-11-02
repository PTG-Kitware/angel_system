using Microsoft.MixedReality.Toolkit.Input;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Represents a virtual assistant in the shape of an orb, staying in the FOV of the user and
/// guiding the user through a sequence of tasks
/// </summary>
public class Orb : Singleton<Orb>
{
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

    ///** Flags
    private bool _isLookingAtOrb = false;                    /// <true if the user is currently looking at the orb shape or orb message
    private bool _lazyLookAtRunning = false;                 /// <used for lazy look at disable
    private bool _lazyFollowStarted = false;                 /// <used for lazy following

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
        // Update eye tracking flag
        if (_isLookingAtOrb && EyeGazeManager.Instance.CurrentHit != EyeTarget.orbFace)
            SetIsLookingAtFace(false);
        else if (!_isLookingAtOrb && EyeGazeManager.Instance.CurrentHit == EyeTarget.orbFace)
            SetIsLookingAtFace(true);

        if (_isLookingAtOrb || _messageContainer.IsLookingAtMessage)
            _face.MessageNotificationEnabled = false;

        float distance = Vector3.Distance(_followSolver.transform.position, AngelARUI.Instance.ARCamera.transform.position);
        float scaleValue = Mathf.Max(1f, distance * 1.2f);
        _followSolver.transform.localScale = new Vector3(scaleValue, scaleValue, scaleValue);

        if (DataProvider.Instance.CurrentSelectedTasks.Keys.Count > 0)
            UpdateMessageVisibility();
    }

    #region Visibility, Position Updates and eye/collision event handler

    /// <summary>
    /// View management
    /// Update the visibility of the orb message based on eye gaze collisions with the orb collider 
    /// </summary>
    private void UpdateMessageVisibility()
    {
        if ((IsLookingAtOrb(false) && !_messageContainer.IsMessageContainerActive && !_messageContainer.IsMessageFading))
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
    /// Called if input events with hand collider are detected
    /// </summary>
    /// <param name="isDragging"></param>
    public void SetIsDragging(bool isDragging)
    {
        _face.UserIsGrabbing = isDragging;

        if (!isDragging && !_lazyFollowStarted)
            StartCoroutine(EnableLazyFollow());

        if (isDragging && _lazyFollowStarted)
        {
            StopCoroutine(EnableLazyFollow());

            _lazyFollowStarted = false;
            _followSolver.IsPaused = (false);
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

    public void AddNotification(string message)
    {
        _messageContainer.AddNotification(message, _face);
        AudioManager.Instance.PlaySound(_face.transform.position, SoundType.warning);
    }
    
    public void RemoveNotification() => _messageContainer.RemoveNotification(_face);

    /// <summary>
    /// Set the task messages the orb communicates, if 'message' is less than 2 char, the message is deactivated
    /// </summary>
    /// <param name="message"></param>
    private void SetTaskMessage(Dictionary<string, TaskList> currentSelectedTasks)
    {
        _messageContainer.UpdateAllTaskMessages(currentSelectedTasks);

        if (_allOrbColliders.Count == 0)
        {
            _allOrbColliders.Add(transform.GetChild(0).GetComponent<BoxCollider>());
            _allOrbColliders.AddRange(_messageContainer.AllColliders);
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
        DataProvider.Instance.RegisterDataSubscriber(() => HandleUpdateTaskListEvent(), SusbcriberType.TaskListChanged);
        DataProvider.Instance.RegisterDataSubscriber(() => HandleUpdateActiveTaskEvent(), SusbcriberType.ObservedTaskChanged);
        DataProvider.Instance.RegisterDataSubscriber(() => HandleUpdateActiveStepEvent(), SusbcriberType.CurrentStepChanged);
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
        SetTaskMessage(DataProvider.Instance.CurrentSelectedTasks);
    }

    #endregion

}
