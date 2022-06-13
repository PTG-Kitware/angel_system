using Microsoft.MixedReality.Toolkit.Utilities.Solvers;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ObjectIndicator : MonoBehaviour
{
    private Shapes.Disc haloouter;
    private Shapes.Disc haloInner;
    private GameObject halo;
    private Vector3 objectpos;
    private Vector3 objectscale;
    private float onscreenDepth;

    private Shapes.Disc indicator;

    public float offscreenThres =0.8f;
    public float baseScale = 5f;

    private Shapes.Line line;

    private bool haloLeft = false;
    private Vector3 poiToCam;

    private float offScreenThres = 0.45f;

    // Transforms to act as start and end markers for the journey.
    private float scalingFrames = 15;
    public Vector3 targetScale = new Vector3(1,1,1);
    public float speed = 1f;

    private DirectionalIndicator directionalSolverPos;

    public int interpolationFramesCount = 45; // Number of frames to completely interpolate between the 2 positions
    int elapsedFrames = 0;

    private bool isFlat = false;

    // Start is called before the first frame update
    void Start()
    {
        Shapes.Disc[] discs = GetComponentsInChildren<Shapes.Disc>();
        haloInner = discs[0];
        haloouter = discs[1];
        halo = haloInner.transform.parent.gameObject;
        objectpos = halo.transform.position;
        objectscale = halo.transform.localScale;

        directionalSolverPos = GetComponentInChildren<DirectionalIndicator>();
        indicator = directionalSolverPos.transform.GetComponentInChildren<Shapes.Disc>();
    }

    // Update is called once per frame
    void Update()
    {
        Vector3 poiToCam = transform.position - AngelARUI.Instance.mainCamera.transform.position;
        onscreenDepth = Mathf.Max(1,poiToCam.magnitude);
        float angle = Vector3.Dot(AngelARUI.Instance.mainCamera.transform.forward, Vector3.Normalize(poiToCam));

        float degangle = Vector3.Angle(AngelARUI.Instance.mainCamera.transform.forward, Vector3.Normalize(poiToCam));
        float alpha = Mathf.Max(0,Mathf.Min(1,(1f / 15f) * (degangle-25f)));
        //Debug.Log(degangle + "  " + alpha);
        indicator.ColorInner = new Color(1, 1, 1, alpha);
        haloInner.ColorOuter = new Color(1, 1, 1, 1-alpha);
        haloouter.ColorInner = new Color(1, 1, 1, 1-alpha);

        //if (degangle>45)
        //{
        //    int layerMask = 1 << 4;

        //    RaycastHit hit;
        //    if (Physics.Raycast(halo.transform.position, halo.transform.right, out hit, 100f, layerMask))
        //    {
        //        Debug.Log("Did Hit Right");
        //        Debug.DrawRay(halo.transform.position, hit.point - halo.transform.position, Color.green);
        //    }
        //    else if (Physics.Raycast(halo.transform.position, (-halo.transform.right), out hit, 100f, layerMask))
        //    {

        // on-screen halo faces the user
        if (!isFlat)
            halo.transform.rotation = Quaternion.LookRotation(AngelARUI.Instance.mainCamera.transform.position - halo.transform.position, Vector3.up);
        else
            halo.transform.rotation = Quaternion.LookRotation(Vector3.up, Vector3.right);

        directionalSolverPos.transform.rotation = Quaternion.LookRotation(AngelARUI.Instance.mainCamera.transform.position - halo.transform.position, Vector3.up);

    }

    public void SetFlat(bool isFlat)
    {
        this.isFlat = isFlat;
    }
}
