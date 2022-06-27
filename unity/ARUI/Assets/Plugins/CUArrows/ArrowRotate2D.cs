using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// [AddComponentMenu("Scripts/AutoRotate")]
public class ArrowRotate2D : MonoBehaviour
{
    //public GameObject obj;

    // Update is called once per frame
    private Vector3 direction = new Vector3(0.0f, 0.0f, 90.0f);
    void Update()
    {
        // make it a child of the current object
        Vector3 displacement = Camera.main.transform.position - transform.position;
        // place it in front of the current object
        //transform.localPosition = Vector3.forward * 5;
        // transform.rotation = Quaternion.identity;
        transform.Rotate(direction * Time.deltaTime);
    }
}
