using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TaskListElement : MonoBehaviour
{
    private string id;
    private bool isDone = false;

    private Color32 inactiveColor = new Color(0.7f, 0.7f, 0.7f, 1f);
    private Color32 activeColor = Color.white;
    private Color32 doneColor = Color.grey;

    private TMPro.TextMeshPro textCanvas;
    private Shapes.Rectangle checkBox;
    private GameObject checkmark;

    // Start is called before the first frame update
    public void InitElement(string text)
    {
        checkBox = GetComponentInChildren<Shapes.Rectangle>();
        if (checkBox == null) Debug.Log("Script could not be found: Shapes.Rectangle at " + gameObject.name);

        checkmark = checkBox.transform.GetChild(0).gameObject;

        textCanvas = GetComponentInChildren<TMPro.TextMeshPro>(true);
        if (textCanvas==null) Debug.Log("Script could not be found: TMPro.TextMeshPro at " + gameObject.name );

        SetText(text);
    }

    public void SetText(string text)
    {
        textCanvas.text = text;
        textCanvas.color = inactiveColor;
        checkBox.Color = inactiveColor;
    }

    public void SetIsDone(bool isDone)
    {
        this.isDone = isDone;
        checkmark.SetActive(isDone);
        
        if (isDone)
        {
            textCanvas.color = doneColor;
            checkBox.Color = doneColor;
        } else
        {
            textCanvas.color = inactiveColor;
            checkBox.Color = inactiveColor;
        }
    }

    public void SetAsCurrent()
    {
        checkmark.SetActive(false);

        textCanvas.color = activeColor;
        checkBox.Color = activeColor;
    }
}
