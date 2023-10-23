using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FollowCameraCanvas : MonoBehaviour
{
    GameObject mainCamera;

    private void Start()
    {
        mainCamera = GameObject.Find("Main Camera");
    }

    private void Update()
    {
        // The canvas should always face the user
        this.transform.LookAt(this.transform.position + mainCamera.transform.rotation * Vector3.forward, Vector3.up);
    }
}
