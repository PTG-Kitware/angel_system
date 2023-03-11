using DilmerGames.Core.Singletons;
using Microsoft.MixedReality.Toolkit.Input;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

/// <summary>
/// Represents a virtual assistant in the shape of an orb, staying in the FOV of the user and
/// guiding the user through a sequence of tasks
/// </summary>
public class Orb : Singleton<Orb>
{
    ///** Reference to parts of the orb
    private OrbFace face;                   /// <the orb shape itself
 
    private OrbGrabbable grabbable;
    private OrbMessage messageContainer;
    private DwellButton taskListbutton;

    private List<BoxCollider> allOrbColliders;
    public List<BoxCollider> AllOrbColliders { get { return allOrbColliders; } }

    private MainMenu mainMenu;

    ///** Placement behaviors - overall, orb stays in the FOV of the user
    private OrbFollowerSolver followSolver;
   


    //Flags
    private bool isLookingAtOrb = false;

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
        taskListbutton.InitializeButton(EyeTarget.orbtasklistButton, () => TaskListManager.Instance.ToggleTasklist(), false, DwellButtonType.Toggle);
        taskListbtn.SetActive(false);

        mainMenu = GetComponentInChildren<MainMenu>();
        mainMenu.gameObject.SetActive(false);

        BoxCollider taskListBtnCol = transform.GetChild(0).GetComponent<BoxCollider>();
        // Collect all orb colliders
        allOrbColliders = new List<BoxCollider>();
        allOrbColliders.Add(taskListBtnCol);
        allOrbColliders.Add(taskListbutton.Collider);
    }

    private void Update()
    {
        // Update eye tracking flag
        if (isLookingAtOrb && FollowEyeTarget.Instance.currentHit != EyeTarget.orbFace)
            SetIsLookingAtFace(false);
        else if (!isLookingAtOrb && FollowEyeTarget.Instance.currentHit == EyeTarget.orbFace)
            SetIsLookingAtFace(true);

        if (messageContainer.UserHasNotSeenNewTask && IsLookingAtOrb(false)) 
            face.SetNotificationIconActive(false);

        UpdateOrbVisibility();
    }


    #region Visibility, Position Updates and eye/collision event handler

    /// <summary>
    /// View management
    /// Update the visibility of the orb message based on eye gaze collisions with the orb collider 
    /// </summary>
    private void UpdateOrbVisibility()
    {
        if (messageContainer.UserHasNotSeenNewTask) return;

        if (TaskListManager.Instance.GetTaskCount() != 0)
        {
            if ((IsLookingAtOrb(false) && !messageContainer.IsMessageVisible && !messageContainer.IsMessageFading))
            { //Set the message visible!
                messageContainer.SetIsActive(true, false);
            }
            else if (!messageContainer.IsLookingAtMessage && !IsLookingAtOrb(false) && followSolver.IsOutOfFOV)
            {
                messageContainer.SetIsActive(false, false);
            }
            else if ((messageContainer.IsLookingAtMessage || IsLookingAtOrb(false)) && messageContainer.IsMessageVisible && messageContainer.IsMessageFading)
            { //Stop Fading, set the message visible
                messageContainer.SetFadeOutMessage(false);
            }
            else if (!IsLookingAtOrb(false) && messageContainer.IsMessageVisible && !messageContainer.IsMessageFading
                && !messageContainer.IsLookingAtMessage && !messageContainer.UserHasNotSeenNewTask && !messageContainer.IsNotificationActive)
            { //Start Fading
                messageContainer.SetFadeOutMessage(true);
            }
        } 
        
        //else if (TaskListManager.Instance.ShowCookbook())
        //{
        //    if (!IsLookingAtOrb(true) && mainMenu.gameObject.activeSelf && !mainMenu.isFading)
        //    {
        //        StartCoroutine(LazyMenuFade());
        //    }
        //    else if (IsLookingAtOrb(true) && !mainMenu.gameObject.activeSelf)
        //        mainMenu.gameObject.SetActive(true);
        //}

    }

    //private IEnumerator LazyMenuFade()
    //{
    //    mainMenu.isFading = true;
    //    yield return new WaitForSeconds(1f);
    //    if (!IsLookingAtOrb(true))
    //        mainMenu.gameObject.SetActive(false);

    //    mainMenu.isFading = false;
    //}

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

        if (!allOrbColliders.Contains(messageContainer.GetMessageCollider()))
            allOrbColliders.Add(messageContainer.GetMessageCollider());
    }

    #endregion

    #region Getter and Setter

    /// <summary>
    /// Access to collider of orb (including task message)
    /// </summary>
    /// <returns>The box collider of the orb message, if the message is not active, returns null</returns>
    public BoxCollider GetCurrentMessageCollider()
    {
        if (messageContainer.GetIsActive())
            return messageContainer.GetMessageCollider();
        else
            return null;
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
    public void SetTaskListButtonActive(bool isActive)
    {
        //mainMenu.gameObject.SetActive(!isActive);
        taskListbutton.gameObject.SetActive(isActive);
    }

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

    public bool IsLookingAtOrb(bool any)
    {
        if (any)
            return isLookingAtOrb || messageContainer.IsLookingAtMessage || taskListbutton.IsInteractingWithBtn
                || FollowEyeTarget.Instance.currentHit == EyeTarget.recipe;
        else
            return isLookingAtOrb || messageContainer.IsLookingAtMessage;
    }

    #endregion

}
