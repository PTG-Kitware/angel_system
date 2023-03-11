using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

/// <summary>
/// The position and scale of a controllable object is defined by the view management component
/// </summary>
public class VMControllable : VMObject
{
    private BoxCollider baseCollider;

    // Start is called before the first frame update
    public void Start()
    {
        baseCollider = gameObject.transform.GetComponentInChildren<BoxCollider>(); 

        //make sure that the AABB is only returned if object is in front of the camera.
        if (transform.InFrontOfCamera(AngelARUI.Instance.ARCamera))
            AABB = transform.RectFromObjs(AngelARUI.Instance.ARCamera, new List<BoxCollider> { baseCollider });
        else
            AABB = Rect.zero;
    }

    public void Update()
    {
        if (transform.InFrontOfCamera(AngelARUI.Instance.ARCamera))
            AABB = transform.RectFromObjs(AngelARUI.Instance.ARCamera, new List<BoxCollider>{ baseCollider });
        else
            AABB = Rect.zero;
    }

    public void UpdateRectBasedOnSubColliders(List<BoxCollider> allColliders)
    {
        if (transform.InFrontOfCamera(AngelARUI.Instance.ARCamera))
        {
            AABB = allColliders[0].transform.RectFromObjs(AngelARUI.Instance.ARCamera, allColliders);
        }
        else
            AABB = Rect.zero;
    }

}
