using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.UI;
using Newtonsoft.Json.Bson;
using System.Collections.Generic;
using System.Runtime.Remoting.Messaging;
using UnityEngine;

public class VMNonControllable : VMObject
{
    protected BoxCollider collider;
    
    public void Awake()
    {
        if (gameObject.GetComponent<BoxCollider>() == null)
            collider = gameObject.AddComponent<BoxCollider>();
        else
            collider = gameObject.GetComponent<BoxCollider>();
    }

    public void Start()
    {
        if (transform.InFrontOfCamera(AngelARUI.Instance.ARCamera))
            AABB = transform.RectFromObjs(AngelARUI.Instance.ARCamera, new List<BoxCollider> { collider });
        else
            AABB = Rect.zero;
    }

    private void Update()
    {
        if (!AngelARUI.Instance.IsVMActiv) return;

        if (transform.InFrontOfCamera(AngelARUI.Instance.ARCamera))
        {
            AABB = transform.RectFromObjs(AngelARUI.Instance.ARCamera, new List<BoxCollider> { collider });
            ViewManagement.Instance.RegisterNonControllable(this);
        }
        else
        {
            ViewManagement.Instance.DeRegisterNonControllable(this);
            AABB = Rect.zero;
        }  
    }

    private void OnDestroy()
    {
        if (!AngelARUI.Instance.IsVMActiv) return;
        ViewManagement.Instance.DeRegisterNonControllable(this);
    }

    private void OnDisable()
    {
        if (!AngelARUI.Instance.IsVMActiv) return;
        ViewManagement.Instance.DeRegisterNonControllable(this);
    }
}
