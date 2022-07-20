using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.UI;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class OrbGrabbable : MonoBehaviour, IMixedRealityPointerHandler
{
    private ObjectManipulator grabbable;

    public void Start()
    {
        grabbable = gameObject.GetComponent<ObjectManipulator>();

        grabbable.OnHoverEntered.AddListener(delegate { OnHoverStarted(); });
        grabbable.OnHoverExited.AddListener(delegate { OnHoverExited(); });
        
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
        Orb.Instance.SetIsDragging(true);
    }

    public void OnPointerUp(MixedRealityPointerEventData eventData)
    {
        Orb.Instance.SetIsDragging(false);
        AudioManager.Instance.PlaySound(transform.position, SoundType.moveEnd);
    }

    public void OnPointerClicked(MixedRealityPointerEventData eventData)
    {
        
    }
}
