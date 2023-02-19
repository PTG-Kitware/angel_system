using DilmerGames.Core.Singletons;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using System;
using System.Collections;
using System.Diagnostics.Eventing.Reader;
using TMPro;
using UnityEngine;
using UnityEngine.Events;
using static UnityEngine.Timeline.TimelineAsset;

/// <summary>
/// Represents a virtual assistant, guiding the user through a sequence of tasks
/// </summary>
public class Orb : Singleton<Orb>
{
    //Reference to parts of the orb
    private OrbFace face;
    private OrbGrabbable grabbable;
    private OrbMessage messageContainer;
    private DwellButton taskListbutton;

    //Placement behaviors
    private OrbFollowerSolver followSolver;

    //Flags
    private bool isLookingAtOrb = false;
    public bool IsLookingAtOrb
    {
        get { return isLookingAtOrb || messageContainer.IsLookingAtMessage; }
    }
    public bool IsDragging
    {
        get { return grabbable.IsDragging; }
        set { SetIsDragging(value); }
    }

    private bool lazyLookAtRunning = false;
    private bool lazyFollowStarted = false;

    /// <summary>
    /// Get all orb references from prefab
    /// </summary>
    void Awake()
    {
        gameObject.name = "Orb";
        face = transform.GetChild(0).GetChild(0).gameObject.AddComponent<OrbFace>();

        //Get message object in orb prefab
        GameObject messageObj = transform.GetChild(0).GetChild(1).gameObject;
        messageContainer = messageObj.AddComponent<OrbMessage>();

        //Get grabbable and following scripts
        followSolver = gameObject.GetComponentInChildren<OrbFollowerSolver>();
        grabbable = gameObject.GetComponentInChildren<OrbGrabbable>();

        ////Init tasklist button
        GameObject taskListbtn = transform.GetChild(0).GetChild(2).gameObject;
        taskListbutton = taskListbtn.AddComponent<DwellButton>();
        taskListbutton.gameObject.name += "FacetasklistButton";
        taskListbutton.InitializeButton(EyeTarget.orbtasklistButton ,() => TaskListManager.Instance.ToggleTasklist());
        taskListbtn.SetActive(false);
    }

    private void Update()
    {
        // Update eye tracking flag
        if (isLookingAtOrb && FollowEyeTarget.Instance.currentHit != EyeTarget.orbFace)
            SetIsLookingAtFace(false);
        else if (!isLookingAtOrb && FollowEyeTarget.Instance.currentHit == EyeTarget.orbFace)
            SetIsLookingAtFace(true);

        if (messageContainer.UserHasNotSeenNewTask && (messageContainer.IsLookingAtMessage || IsLookingAtOrb)) 
            face.SetNotificationIconActive(false);

        UpdateOrbVisibility();

        if (!taskListbutton.GetIsLookingAtBtn() && TaskListManager.Instance.GetIsTaskListActive())
            taskListbutton.SetSelected(true);
        else if (!taskListbutton.GetIsLookingAtBtn() && !TaskListManager.Instance.GetIsTaskListActive())
            taskListbutton.SetSelected(false);
    }


    #region Visibility, Position Updates and eye/collision event handler

    /// <summary>
    /// luminance-based view management
    /// Update the visibility of the orb message based on eye gaze collisions with the orb collider 
    /// </summary>
    private void UpdateOrbVisibility()
    {
        if (messageContainer.UserHasNotSeenNewTask) return;

        //Debug.Log(IsLookingAtOrb + ", " + messageContainer.isMessageVisible + ", " + messageContainer.isMessageFading); 
        if ((IsLookingAtOrb && !messageContainer.IsMessageVisible && !messageContainer.IsMessageFading))
        { //Set the message visible!
            messageContainer.SetIsActive(true, false);
        } else if (!messageContainer.IsLookingAtMessage && !IsLookingAtOrb && followSolver.IsOutOfFOV){
            messageContainer.SetIsActive(false, false);
        }
        else if ((messageContainer.IsLookingAtMessage || IsLookingAtOrb) && messageContainer.IsMessageVisible && messageContainer.IsMessageFading)
        { //Stop Fading, set the message visible
            messageContainer.SetFadeOutMessage(false);
        }
        else if (!IsLookingAtOrb && messageContainer.IsMessageVisible && !messageContainer.IsMessageFading 
            && !messageContainer.IsLookingAtMessage && !messageContainer.UserHasNotSeenNewTask && !messageContainer.IsNotificationActive)
        { //Start Fading
            messageContainer.SetFadeOutMessage(true);
        }
    }

    /// <summary>
    /// If the user drags the orb, the orb will stay in place until it will be out of FOV
    /// </summary>
    private IEnumerator EnableLazyFollow()
    {
        lazyFollowStarted = true;

        yield return new WaitForEndOfFrame();

        followSolver.IsPaused = (true);

        while (Utils.InFOV(AngelARUI.Instance.ARCamera, grabbable.transform.position))
        {
            yield return new WaitForSeconds(0.1f);
        }

        followSolver.IsPaused = (false);
        lazyFollowStarted = false;
    }

    /// <summary>
    /// Make sure that fast eye movements are not detected as dwelling
    /// </summary>
    /// <returns></returns>
    private IEnumerator StartLazyLookAt()
    {
        yield return new WaitForSeconds(0.2f);

        if (lazyLookAtRunning)
        {
            isLookingAtOrb = true;
            lazyLookAtRunning = false;
        }
    }

    /// <summary>
    /// Called if input events with hand collider are detected
    /// </summary>
    /// <param name="isDragging"></param>
    private void SetIsDragging(bool isDragging)
    {
        face.ChangeDragginColorActive(isDragging);
        followSolver.IsPaused = (isDragging);

        if (!isDragging && !lazyFollowStarted)
        {
            StartCoroutine(EnableLazyFollow());
        }

        if (isDragging && lazyFollowStarted)
        {
            StopCoroutine(EnableLazyFollow());

            lazyFollowStarted = false;
            followSolver.IsPaused = (false);
        }
    }

    /// <summary>
    /// Called if changes in eye events are detected
    /// </summary>
    /// <param name="isLooking"></param>
    private void SetIsLookingAtFace(bool isLooking)
    {
        if (isLooking && !lazyLookAtRunning)
        {
            lazyLookAtRunning = true;
            StartCoroutine(StartLazyLookAt());
        }
        else if (!isLooking)
        {
            if (lazyLookAtRunning)
                StopCoroutine(StartLazyLookAt());

            isLookingAtOrb = false;
            lazyLookAtRunning = false;
            //Debug.Log("Set looking at orb false");
        }

    }

    #endregion

    #region Task Messages and Notifications

    /// <summary>
    /// Set the notification messages the orb communicates, if 'message' is less than 2 char, the message is deactivated
    /// </summary>
    /// <param name="message"></param>
    public void SetNotificationMessage(string message)
    {
        if (message.Length <= 1)
        {
            messageContainer.IsNotificationActive = false ;
            face.ChangeColorToNotificationActive(false);
            followSolver.MoveToCenter(false);
        }
        else
        {
            messageContainer.SetIsActive(true, false);
            messageContainer.IsNotificationActive = true;
            face.ChangeColorToNotificationActive(true);
            followSolver.MoveToCenter(true);

            AudioManager.Instance.PlaySound(transform.position, SoundType.warning);
        }

        messageContainer.SetNotificationMessage(message);
    }

    /// <summary>
    /// Set the task messages the orb communicates, if 'message' is less than 2 char, the message is deactivated
    /// </summary>
    /// <param name="message"></param>
    public void SetTaskMessage(string message)
    {
        if (message.Length <= 1)
            messageContainer.SetIsActive(false, false);
        else
        {
            messageContainer.SetIsActive(true, true);
            face.SetNotificationIconActive(true);

            AudioManager.Instance.PlayText(message);
        }

        messageContainer.SetTaskMessage(message);

        SetNotificationMessage("");
        face.ChangeColorToDone(message.Contains("Done"));
    }

    #endregion

    #region Getter and Setter

    /// <summary>
    /// Access to collider of orb (including task message)
    /// </summary>
    /// <param name="collider"></param>
    /// <returns></returns>
    public bool GetCurrentMessageCollider(ref BoxCollider collider)
    {
        if (messageContainer.GetIsActive() && messageContainer.IsMessageVisible)
        {
            collider = messageContainer.GetMessageCollider();
            return true;
        }
        else
            return false;
    }

    /// <summary>
    /// Detect hand hovering events
    /// </summary>
    /// <param name="isHovering"></param>
    public void SetNearHover(bool isHovering) => face.SetDraggableHandle(isHovering);
    
    /// <summary>
    /// Change the visibility of the tasklist button
    /// </summary>
    /// <param name="isActive"></param>
    public void SetTaskListButtonActive(bool isActive) => taskListbutton.gameObject.SetActive(isActive);

    /// <summary>
    /// Update the position behavior of the orb
    /// </summary>
    /// <param name="isSticky"></param>
    public void SetSticky(bool isSticky)
    {
        followSolver.SetSticky(isSticky);

        if (isSticky)
            messageContainer.SetIsActive(false, false);
    }

    #endregion

}
