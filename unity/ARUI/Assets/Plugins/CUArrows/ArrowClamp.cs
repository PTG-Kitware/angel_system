using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.Toolkit.Utilities;


public class ArrowClamp : MonoBehaviour
{
    // Start is called before the first frame update
    public GameObject arrowBody;

    void Start()
    {
        
        Debug.Log("origin:" + arrowBody.transform.position);
        Debug.Log("Point 1: " + arrowBody.GetComponent<ArrowBodyDataProvider>().controlPoints[0].Position);
        Debug.Log("Point 2: " + arrowBody.GetComponent<ArrowBodyDataProvider>().controlPoints[1]);
        Debug.Log("Point 3: " + arrowBody.GetComponent<ArrowBodyDataProvider>().controlPoints[2]);
        Debug.Log("Point 4: " + arrowBody.GetComponent<ArrowBodyDataProvider>().controlPoints[3].Position);
    }

    //private float speed = 0.1f;
    //private Vector3 direction = new Vector3(0.0f, 0.0f, 30.0f);

    // Update is called once per frame
    void Update()
    {
        Vector3 attachPoint = arrowBody.GetComponent<ArrowBodyDataProvider>().controlPoints[0].Position;
        //Debug.Log("control point position: " + attachPoint);
        //Debug.Log("control point transform: " + arrowBody.transform.position);
        transform.position = attachPoint;
        //Debug.Log("calculated attach point: " + attachPoint);
        //Debug.Log("new pos: " + transform.position);

        //transform.Translate(Vector3.forward * Speed * Time.deltaTime);
        //transform.Rotate(direction * Time.deltaTime);
    }
}
