using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.UI;
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

    protected void Update()
    {
        if (!AngelARUI.Instance.IsVMActiv ) return;

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
        if (ViewManagement.Instance != null)
            ViewManagement.Instance.DeRegisterNonControllable(this);
    }

    private void OnDisable()
    {
        if (ViewManagement.Instance!=null)
            ViewManagement.Instance.DeRegisterNonControllable(this);
    }

    private void OnGUI()
    {
        if (!AngelARUI.Instance.PrintVMDebug || this is CVDetectedObj) return;

        GUIStyle tintableText = new GUIStyle(GUI.skin.box);
        tintableText.normal.background = Texture2D.whiteTexture; // must be white to tint properly
        tintableText.normal.textColor = Color.white; // whatever you want

        GUI.backgroundColor = new Color(255, 255, 255, 0.7f);
        int[] rect = Utils.GUIGetCappedGUI(AABB);
        GUI.Box(new Rect(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]), gameObject.name, tintableText);

    }
}
