using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.UI;
using Newtonsoft.Json.Bson;
using UnityEngine;

public class VMNonControllable : VMObject
{
    protected BoxCollider collider;
    
    public void Awake()
    {
        if (gameObject.GetComponent<BoxCollider>() == null)
            collider = gameObject.AddComponent<BoxCollider>();
    }

    public void Start()
    {
        if (transform.InFrontOfCamera(AngelARUI.Instance.ARCamera))
            AABB = transform.RectFromObj(AngelARUI.Instance.ARCamera, collider);
        else
            AABB = Rect.zero;
    }

    private void Update()
    {
        if (collider == null && gameObject.GetComponent<BoxCollider>() == null)
            collider = gameObject.AddComponent<BoxCollider>();

        if (transform.InFrontOfCamera(AngelARUI.Instance.ARCamera))
        {
            AABB = transform.RectFromObj(AngelARUI.Instance.ARCamera, collider);
            ViewManagement.Instance.RegisterNonControllable(this);
        }
        else
            AABB = Rect.zero;
    }

    private void OnDestroy()
    {
        ViewManagement.Instance.DeRegisterNonControllable(this);
    }

    private void OnDisable()
    {
        ViewManagement.Instance.DeRegisterNonControllable(this);
    }
}
