using System.Collections;
using UnityEngine;
using Holofunk.Core;

public class HandPoseManager : Singleton<HandPoseManager>
{
    public Holofunk.HandPose.HandPoseVisualizer left;
    public Holofunk.HandPose.HandPoseVisualizer right;

    public Holofunk.HandPose.HandPose leftPose;
    public Holofunk.HandPose.HandPose rightPose;

    // Start is called before the first frame update
    void Start()
    {
        left = transform.GetChild(0).GetComponent<Holofunk.HandPose.HandPoseVisualizer>();
        right = transform.GetChild(1).GetComponent<Holofunk.HandPose.HandPoseVisualizer>();
    }

    // Update is called once per frame
    void Update()
    {
        leftPose = left.LastDetectedPose;

        rightPose = right.LastDetectedPose;    
    }
}
