using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.UI;
using UnityEngine;

/// <summary>
/// Catches pointer and dragging events at orb
/// </summary>
public class TaskListGrabbable : MonoBehaviour, IMixedRealityPointerHandler
{
    private ObjectManipulator grabbable;
    public bool isDragging;

    public void Start()
    {
        grabbable = gameObject.GetComponent<ObjectManipulator>();

        grabbable.OnHoverEntered.AddListener(delegate { OnHoverStarted(); });
        grabbable.OnHoverExited.AddListener(delegate { OnHoverExited(); });
    }

    private void OnHoverStarted() => TaskListManager.Instance.SetNearHover(true);

    private void OnHoverExited() => TaskListManager.Instance.SetNearHover(false);

    public void OnPointerDown(MixedRealityPointerEventData eventData)
    {
        TaskListManager.Instance.SetIsDragging(true);
        isDragging = true;
        AudioManager.Instance.PlaySound(transform.position, SoundType.moveStart);
    }

    public void OnPointerDragged(MixedRealityPointerEventData eventData)
    {
        TaskListManager.Instance.SetIsDragging(true);
        isDragging = true;
    }

    public void OnPointerUp(MixedRealityPointerEventData eventData)
    {
        TaskListManager.Instance.SetIsDragging(false);
        isDragging = false;
        AudioManager.Instance.PlaySound(transform.position, SoundType.moveEnd);
    }

    public void OnPointerClicked(MixedRealityPointerEventData eventData) { }
}
