using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.OpenXR.Features.Interactions;

//Script attached to rectangle to activate respctive task list
public class CurrentListActivator : MonoBehaviour
{
    public int index;
    public GameObject ListContainer;
    // Start is called before the first frame update
    void Start()
    {

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
                    FadeIn();
                    //Put orb into area
                }
            }
        }
    }
    private void FadeIn()
    {
        MultiTaskList.Instance.SetMenuActive(index);
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
