using UnityEngine;

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
            AABB = transform.RectFromObj(AngelARUI.Instance.ARCamera, baseCollider);
        else
            AABB = Rect.zero;
    }

    public void Update()
    {
        if (transform.InFrontOfCamera(AngelARUI.Instance.ARCamera)) 
            AABB = transform.RectFromObj(AngelARUI.Instance.ARCamera, baseCollider);
        else
            AABB = Rect.zero;
    }

    public void UpdateRectBasedOnSubColliders(BoxCollider box1, BoxCollider box2)
    {
        if (transform.InFrontOfCamera(AngelARUI.Instance.ARCamera))
        {
            if (box1 == null)
                AABB = box2.transform.RectFromObj(AngelARUI.Instance.ARCamera, box2);
            else
                AABB = box2.transform.RectFromObjs(AngelARUI.Instance.ARCamera,new BoxCollider[2] { box1, box2 }, new Transform[2] { box1.transform , box2.transform });
        }
        else
            AABB = Rect.zero;
    }

}
