using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.UI;
using Newtonsoft.Json.Bson;
using System.Collections.Generic;
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
        ViewManagement.Instance.DeRegisterNonControllable(this);
    }

    private void OnDisable()
    {
        ViewManagement.Instance.DeRegisterNonControllable(this);
    }
}
