using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.UI;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Utilities;
using UnityEngine;
using Shapes;
using System;
using System.Collections;
using Microsoft.MixedReality.OpenXR;

/// <summary>
/// Catch pointer and dragging events at orb
/// </summary>
public class OrbGrabbable : MonoBehaviour, IMixedRealityPointerHandler
{
    private ObjectManipulator _grabbable;

    private bool _grabbingAllowed = true;
    public bool IsGrabbingAllowed
    {
        get { return _grabbingAllowed; }
        set { _grabbable.enabled = value; }
    }

    private bool _isProcessingClosedHand = false;

    private void Start()
    {
        _grabbable = gameObject.GetComponent<ObjectManipulator>();

        _grabbable.OnHoverEntered.AddListener(delegate { OnHoverStarted(); });
        _grabbable.OnHoverExited.AddListener(delegate { OnHoverExited(); });
    }

    private void OnHoverStarted() => Orb.Instance.SetNearHover(true);

    private void OnHoverExited() => Orb.Instance.SetNearHover(false);

    public void OnPointerDown(MixedRealityPointerEventData eventData)
    {
        Orb.Instance.SetIsDragging(true);
        AudioManager.Instance.PlaySound(transform.position, SoundType.moveStart);
    }

    public void OnPointerDragged(MixedRealityPointerEventData eventData)
    {
        if (!_isProcessingClosedHand && isUsedHandClosed(eventData))
        {
            //start countdown for or fix
            _isProcessingClosedHand = true;
            StartCoroutine(TransitionToFixedMovement());
        } else if (_isProcessingClosedHand && !isUsedHandClosed(eventData))
        {
            _isProcessingClosedHand = false;
            StopCoroutine(TransitionToFixedMovement());

            Orb.Instance.UpdateMovementbehavior(MovementBehavior.Follow);

        } else if (!_isProcessingClosedHand && !isUsedHandClosed(eventData))
        {
            Orb.Instance.UpdateMovementbehavior(MovementBehavior.Follow);
        }
    }

    public IEnumerator TransitionToFixedMovement()
    {
        Orb.Instance.UpdateMovementbehavior(MovementBehavior.Fixed);

        float duration = 2f; 
        float pastSeconds = 0;

        while (_isProcessingClosedHand && pastSeconds < duration)
        {
            pastSeconds += Time.deltaTime;
            Orb.Instance.SetHandleProgress(pastSeconds / duration);
            yield return new WaitForEndOfFrame();
        }

        _isProcessingClosedHand = false;
    }

    public void OnPointerUp(MixedRealityPointerEventData eventData)
    {
        Orb.Instance.SetIsDragging(false);
        AudioManager.Instance.PlaySound(transform.position, SoundType.moveEnd);
    }

    public void OnPointerClicked(MixedRealityPointerEventData eventData) {}

    /// <summary>
    /// Helper function that returns true if the hand the user interacts with is closed
    /// </summary>
    /// <param name="eventData"></param>
    /// <returns></returns>
    private bool isUsedHandClosed(MixedRealityPointerEventData eventData)
    {
        return (Microsoft.MixedReality.Toolkit.Utilities.Handedness.Right == eventData.Handedness && HandPoseManager.Instance.rightPose == Holofunk.HandPose.HandPose.Closed)
         || (Microsoft.MixedReality.Toolkit.Utilities.Handedness.Left == eventData.Handedness && HandPoseManager.Instance.leftPose == Holofunk.HandPose.HandPose.Closed);
    }
}