using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UIElements;

public class VMControllable : VMObject
{
    private BoxCollider _baseCollider;
    public BoxCollider BaseCollider => _baseCollider;
   
    public void Start()
    {
        _baseCollider = gameObject.transform.GetComponentInChildren<BoxCollider>();

        if (AngelARUI.Instance.ARCamera == null) return;

        //make sure that the AABB is only returned if object is in front of the camera.
        if (transform.InFrontOfCamera(AngelARUI.Instance.ARCamera))
            AABB = transform.RectFromObjs(AngelARUI.Instance.ARCamera, new List<BoxCollider> { _baseCollider });
        else
            AABB = Rect.zero;
    }

    public void Update()
    {
        if (transform.InFrontOfCamera(AngelARUI.Instance.ARCamera) &&
            BaseCollider.enabled)
            AABB = transform.RectFromObjs(AngelARUI.Instance.ARCamera, new List<BoxCollider> { _baseCollider });
        else
            AABB = Rect.zero;
    }

    public void UpdateRectBasedOnSubColliders(List<BoxCollider> allColliders)
    {
        if (allColliders.Count !=0 && transform.InFrontOfCamera(AngelARUI.Instance.ARCamera) &&
            BaseCollider.enabled)
        {
            AABB = allColliders[0].transform.RectFromObjs(AngelARUI.Instance.ARCamera, allColliders);
        }
        else
            AABB = Rect.zero;
    }

    private void OnGUI()
    {
        if (!AngelARUI.Instance.PrintVMDebug) return;

        GUIStyle tintableText = new GUIStyle(GUI.skin.box);
        tintableText.normal.background = Texture2D.whiteTexture; // must be white to tint properly
        tintableText.normal.textColor = Color.white; // whatever you want

        GUI.backgroundColor = new Color(255, 255, 255, 0.7f);
        int[] rect = Utils.GUIGetCappedGUI(AABB);
        GUI.Box(new Rect(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]), gameObject.name, tintableText);

    }
}
