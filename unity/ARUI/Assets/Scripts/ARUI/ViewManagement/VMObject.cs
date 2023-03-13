using Microsoft.MixedReality.OpenXR;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class VMObject : MonoBehaviour
{
    public Rect AABB; //AABB: minx, miny, maxX, maxY - GUI coordinate system

    private void Start()
    {
        GameObject vmCopy = new GameObject("VMCopy_" + gameObject.name);
        vmCopy.transform.parent = transform;
        vmCopy.transform.localPosition = new Vector3(0, 0, 0);
        vmCopy.transform.localScale = Vector3.one;
        vmCopy.layer = 30;
    }
}
