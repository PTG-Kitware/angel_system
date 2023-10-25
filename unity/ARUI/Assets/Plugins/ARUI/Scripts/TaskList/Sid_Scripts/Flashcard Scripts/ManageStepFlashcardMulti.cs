using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using Shapes;

public class ManageStepFlashcardMulti : MonoBehaviour
{
    public GameObject BorderRect;
    public GameObject ParentObj;
    public GameObject ParentRect;
    public TMP_Text ParentTaskText;
    public GameObject SubTaskPrefab;
    public GameObject VerticalLayoutGroupObj;

    //Function that takes in task and then uses it to initiailize the list 
    //Also need to add all required items
    public void InitializeFlashcad(Step currStep)
    {
        ResetSubTasks();
        ParentTaskText.SetText(currStep.StepDesc);
        for(int i = currStep.CurrSubStepIndex; i < currStep.SubSteps.Count; i++)
        {
            SubStep sub = currStep.SubSteps[i];
            GameObject currSubtask = Instantiate(SubTaskPrefab, VerticalLayoutGroupObj.transform);
            currSubtask.GetComponent<SubTaskStep>().SetSubStepText(sub.StepDesc);
            //Increase parent height 
            RectTransform parentRect = ParentObj.GetComponent<RectTransform>();
            parentRect.sizeDelta = new Vector2(parentRect.sizeDelta.x, parentRect.sizeDelta.y + 0.02f);
            //Modify parent collider
            BoxCollider parentCol = ParentObj.GetComponent<BoxCollider>();
            parentCol.center = new Vector3(parentCol.center.x, parentCol.center.y + 0.01f, parentCol.center.z);
            //Increase position of parent rectangle
            RectTransform ParentRectangle = ParentRect.GetComponent<RectTransform>();
            ParentRectangle.localPosition = new Vector2(ParentRectangle.localPosition.x, ParentRectangle.localPosition.y + 0.01f);
            //Increase rectangle border height and center
            BorderRect.GetComponent<Rectangle>().Height += 0.03f;
            RectTransform rect = BorderRect.GetComponent<RectTransform>();
            rect.localPosition = new Vector2(rect.localPosition.x, rect.localPosition.y - 0.005f);
            //Increase box collider border height and center
            BoxCollider collider = VerticalLayoutGroupObj.GetComponent<BoxCollider>();
            collider.center = new Vector3(collider.center.x, collider.center.y - 0.015f, collider.center.z);
            collider.size = new Vector3(collider.size.x, collider.size.y + 0.03f, collider.size.z);
        }
        int currIndex = currStep.CurrSubStepIndex;
    }


    //Function to reset the list to original values (delete all subtasks, rescale rects)
    public void ResetSubTasks()
    {
        foreach (Transform child in VerticalLayoutGroupObj.transform)
        {
            Destroy(child.gameObject);
            //Decrease parent height 
            RectTransform parentRect = ParentObj.GetComponent<RectTransform>();
            parentRect.sizeDelta = new Vector2(parentRect.sizeDelta.x, parentRect.sizeDelta.y - 0.02f);
            //Modify parent collider
            BoxCollider parentCol = ParentObj.GetComponent<BoxCollider>();
            parentCol.center = new Vector3(parentCol.center.x, parentCol.center.y - 0.01f, parentCol.center.z);
            //Decrease position of parent rectangle
            RectTransform ParentRectangle = ParentRect.GetComponent<RectTransform>();
            ParentRectangle.localPosition = new Vector2(ParentRectangle.localPosition.x, ParentRectangle.localPosition.y - 0.01f);
            //Decrease rectangle border height and center
            BorderRect.GetComponent<Rectangle>().Height -= 0.03f;
            RectTransform rect = BorderRect.GetComponent<RectTransform>();
            rect.localPosition = new Vector2(rect.localPosition.x, rect.localPosition.y + 0.005f);
            //Decrease box collider border height and center
            BoxCollider collider = VerticalLayoutGroupObj.GetComponent<BoxCollider>();
            collider.center = new Vector3(collider.center.x, collider.center.y + 0.015f, collider.center.z);
            collider.size = new Vector3(collider.size.x, collider.size.y - 0.03f, collider.size.z);
        }
    }
}
