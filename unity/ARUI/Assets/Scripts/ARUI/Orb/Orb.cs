using DilmerGames.Core.Singletons;
using Microsoft.MixedReality.Toolkit.Input;
using System.Collections;
using UnityEngine;

/// <summary>
/// Represents a virtual assistant, guiding the user through a sequence of tasks
/// </summary>
public class Orb : Singleton<Orb>
{
    private OrbFace face;
    private OrbGrabbable grabbable;
    private OrbMessage messageContainer;

    //Placement behaviors
    private OrbFollowerSolver followSolver;
    private bool lazyFollowStarted = false;

    //Input events 
    private EyeTrackingTarget eyeEvents;
    public bool IsLookingAtOrb = false;

    /// <summary>
    /// Instantiate and Initialise all objects related to the orb.
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

        //Init input events
        eyeEvents = transform.GetChild(0).GetComponent<EyeTrackingTarget>();
        eyeEvents.OnLookAtStart.AddListener(delegate { SetIsLookingAtFace(true); });
        eyeEvents.OnLookAway.AddListener(delegate { SetIsLookingAtFace(false); });
    }

    public void Update()
    {
        if (messageContainer.userHasNotSeenNewTask && (messageContainer.isLookingAtMessage || IsLookingAtOrb))
        {
            followSolver.MoveToEyeTarget(false);
            face.SetNotificationIconActive(false);
        }

        UpdateOrbVisibility();
    }


    #region Visibility and Position Updates

    /// <summary>
    /// luminance-based view management
    /// Update the visibility of the orb message based on eye gaze collisions with the orb collider 
    /// </summary>
    private void UpdateOrbVisibility()
    {
        if (messageContainer.userHasNotSeenNewTask) return;

        if ((IsLookingAtOrb && !messageContainer.isMessageVisible && !messageContainer.isMessageFading))
        { //Set the message visible!
            SetMessageActive(true);
        }
        else if ((messageContainer.isLookingAtMessage || IsLookingAtOrb) && messageContainer.isMessageVisible && messageContainer.isMessageFading)
        { //Stop Fading, set the message visible
            messageContainer.SetFadeOutMessage(false);
        }
        else if (!IsLookingAtOrb && messageContainer.isMessageVisible && !messageContainer.isMessageFading 
            && !messageContainer.isLookingAtMessage && !messageContainer.userHasNotSeenNewTask && !messageContainer.isNotificationActive)
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

        followSolver.SetPaused(true);

        while (Utils.InFOV(AngelARUI.Instance.mainCamera, grabbable.transform.position))
        {
            yield return new WaitForSeconds(0.1f);
        }

        followSolver.SetPaused(false);
        lazyFollowStarted = false;
    }

    #endregion

    #region Messages and Notifications

    /// <summary>
    /// Set the notification messages the orb communicates, if 'message' is less than 2 char, the message is deactivated
    /// </summary>
    /// <param name="message"></param>
    public void SetNotificationMessage(string message)
    {
        if (message.Length <= 1 && messageContainer.isNotificationActive && messageContainer.isActive())
        {
            messageContainer.SetNotificationTextActive(false);
            face.ChangeColorToNotificationActive(false);
        }
        else
        {
            messageContainer.SetNotificationText(message);
            messageContainer.SetNotificationTextActive(true);
            face.ChangeColorToNotificationActive(true);
        }
    }

    /// <summary>
    /// Set the task messages the orb communicates, if 'message' is less than 2 char, the message is deactivated
    /// </summary>
    /// <param name="message"></param>
    public void SetTaskMessage(string message)
    {
        if (message.Length <= 1 && messageContainer.isMessageActive && messageContainer.isActive())
            SetMessageActive(false);
        else
        {
            SetMessageActive(true);
            messageContainer.HandleNewTask();
            //followSolver.MoveToEyeTarget(true);
            face.SetNotificationIconActive(true);
        }

        messageContainer.SetTaskMessage(message);
    }

    private void SetMessageActive(bool isActive)
    {
        messageContainer.isMessageActive = isActive;
        messageContainer.SetActive(messageContainer.isMessageActive);
        messageContainer.SetVisible(isActive);
    }

    #endregion

    #region Getter and Setter

    public bool GetCurrentMessageCollider(ref BoxCollider collider)
    {
        if (messageContainer.isMessageActive && messageContainer.isMessageVisible)
        {
            collider = messageContainer.GetCollider();
            return true;
        }
        else
            return false;
    }

    public bool GetIsUserLookingAtOrb() => IsLookingAtOrb || messageContainer.isLookingAtMessage;

    public bool GetIsDragging() => grabbable.isDragging;

    public void SetIsDragging(bool isDragging)
    {
        face.ChangeDragginColorActive(isDragging);
        followSolver.SetPaused(isDragging);

        if ( !isDragging && !lazyFollowStarted)
        {
            StartCoroutine(EnableLazyFollow());
        }

        if (isDragging && lazyFollowStarted)
        {
            StopCoroutine(EnableLazyFollow());
            lazyFollowStarted = false;
            followSolver.SetPaused(false);
        }
    }

    public void SetNearHover(bool isHovering) => face.SetDraggableHandle(isHovering);

    private void SetIsLookingAtFace(bool isLooking) => IsLookingAtOrb = isLooking;

    #endregion

}
