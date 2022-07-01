using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum ListPosition
{
    Top = 0,
    Bottom = 1,
    Middle =2,
}

public class TaskListElement : MonoBehaviour
{
    public int id;
    private bool isDone = false;

    private Color inactiveColor = Color.gray;
    private Color activeColor = Color.white;
    private Color doneColor = new Color(0.1f, 0.2f, 0.1f);

    private TMPro.TextMeshProUGUI textCanvas;
    private Shapes.Rectangle checkBox;
    private Shapes.Cone checkBoxCurrent;
    private Shapes.Line subTaskIndicator;

    private int taskLevel = 0;

    private float currentAlpha = 1f;
    // Left, Top, Right, Bottom
    private Vector4 prefabMargin;
    private Vector4 subTaskMargin = new Vector4(0.01f, 0, 0, 0);
    private float topBottomMargin = 0.03f;

    private string postMessage = "";
    private string taskMessage = "";

    private void InitIfNeeded()
    {
        if (checkBox == null)
        {
            checkBox = GetComponentInChildren<Shapes.Rectangle>();
            if (checkBox == null) Debug.Log("Script could not be found: Shapes.Rectangle at " + gameObject.name);
            
            checkBoxCurrent = GetComponentInChildren<Shapes.Cone>();
            if (checkBoxCurrent == null) Debug.Log("Script could not be found: Shapes.Cone at " + gameObject.name);

            subTaskIndicator = transform.GetComponentInChildren<Shapes.Line>();
            if (subTaskIndicator == null) Debug.Log("Script could not be found: Shapes.Line at " + gameObject.name);

            textCanvas = GetComponent<TMPro.TextMeshProUGUI>();
            if (textCanvas == null) Debug.Log("Script could not be found: TMPro.TextMeshProUGUI at " + gameObject.name);

            prefabMargin = new Vector4(textCanvas.margin.x, textCanvas.margin.y, textCanvas.margin.z, textCanvas.margin.w);
        }
    }

    public void SetText(int taskID, string text, int taskLevel)
    {
        InitIfNeeded();

        textCanvas.text = text;
        this.taskLevel = taskLevel;
        taskMessage = text;
        id = taskID;
        
        checkBox.gameObject.SetActive(false);
        checkBoxCurrent.gameObject.SetActive(false);

        UpdateColor(inactiveColor);

        if (taskLevel == 0)
        {
            textCanvas.margin = prefabMargin;
            subTaskIndicator.gameObject.SetActive(false);

            textCanvas.fontStyle = TMPro.FontStyles.UpperCase;
        }
        else
        {
            textCanvas.margin = prefabMargin + subTaskMargin;
            subTaskIndicator.gameObject.SetActive(false);
        }
    }
    
    public void UpdateListPosition(ListPosition pos)
    {
        if (pos.Equals(ListPosition.Top)) {
            if (taskLevel == 0)
                textCanvas.margin = prefabMargin + new Vector4(0, topBottomMargin, 0, 0f);
            else 
                textCanvas.margin = prefabMargin + subTaskMargin + new Vector4(0, topBottomMargin, 0, 0f);
        }
        else if (pos.Equals(ListPosition.Bottom))
        {
            if (taskLevel == 0)
                textCanvas.margin = prefabMargin + new Vector4(0, 0, 0, topBottomMargin);
            else
                textCanvas.margin = prefabMargin + subTaskMargin + new Vector4(0, 0, 0, topBottomMargin);
        }
        else if (pos.Equals(ListPosition.Middle))
        {
            if (taskLevel == 0)
                textCanvas.margin = prefabMargin;
            else
                textCanvas.margin = prefabMargin + subTaskMargin;
        }
    }

    public void SetIsDone(bool isDone)
    {
        InitIfNeeded();

        checkBox.gameObject.SetActive(true);
        checkBoxCurrent.gameObject.SetActive(false);

        this.isDone = isDone;

        //define color and alpha of element based on user attention and task state
        if (isDone)
            UpdateColor(doneColor);
        else
            UpdateColor(inactiveColor);

        this.postMessage = "";
        if (taskLevel == 0)
            textCanvas.text = taskMessage;
    }

    public void SetAsCurrent(string postMessage)
    {
        InitIfNeeded();

        checkBox.gameObject.SetActive(false);
        checkBoxCurrent.gameObject.SetActive(true);

        UpdateColor(activeColor);

        this.postMessage = postMessage;
        if (taskLevel==0 && postMessage.Length>0)
            textCanvas.text = taskMessage + " - " +postMessage;

    }

    private void UpdateColor(Color newColor)
    {
        textCanvas.color = new Color(newColor.r, newColor.g, newColor.b, currentAlpha);
        checkBoxCurrent.Color = new Color(newColor.r, newColor.g, newColor.b, currentAlpha);
        checkBox.Color = new Color(newColor.r, newColor.g, newColor.b, currentAlpha);
        subTaskIndicator.Color = new Color(newColor.r, newColor.g, newColor.b, currentAlpha);
    }

    public void SetAlpha(float alpha)
    {
        textCanvas.color = new Color(textCanvas.color.r, textCanvas.color.g, textCanvas.color.b, alpha);
        checkBoxCurrent.Color = new Color(checkBoxCurrent.Color.r, checkBoxCurrent.Color.g, checkBoxCurrent.Color.b, alpha);
        subTaskIndicator.Color = new Color(subTaskIndicator.Color.r, subTaskIndicator.Color.g, subTaskIndicator.Color.b, alpha);
        checkBox.Color = new Color(checkBox.Color.r, checkBox.Color.g, checkBox.Color.b, alpha);

        currentAlpha = alpha;
    }
}
