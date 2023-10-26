using Microsoft.MixedReality.Toolkit.Utilities;
using Shapes;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.Eventing.Reader;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.XR.OpenXR.Features.Interactions;

//Script attached to rectangle to activate respctive task list
public class CurrentListActivator : MonoBehaviour
{
    public int index;
    public GameObject ListContainer;

    private Rectangle rect;

    private Line rectProgress;
    private float xStart = 0;
    private float xEnd = 0;
    private float completeX = 0;

    private void Awake()
    {
        rect = GetComponentInChildren<Rectangle>();

        Line[] tmp = GetComponentsInChildren<Line>();
        rectProgress = tmp[1];
        xStart = rectProgress.Start.x;
        xEnd = rectProgress.End.x;
        completeX = Mathf.Abs(xStart - xEnd);

        rectProgress.End = new Vector3(xStart, 0, 0);
    }

    // Update is called once per frame
    void Update()
    {
        //Once user looks at this object, set the task list visible
        if (EyeGazeManager.Instance != null)
        {
            if (EyeGazeManager.Instance.CurrentHitObj != null)
            {
                if (EyeGazeManager.Instance.CurrentHitObj.GetInstanceID() == this.gameObject.GetInstanceID())
                {
                    //fade in tasklist 
                    MultiTaskList.Instance.SetMenuActive(index);
                }
            }
        }
    }

    public void UpdateProgres(float ratio)
    {
        if (rect == null) return;
        rectProgress.End = new Vector3(xStart + completeX * ratio, 0, 0);
    }

    public void SetContainer(GameObject container)
    {
        ListContainer = container;
    }

    public void SetIndex(int index)
    {
        this.index = index;
    }
}
