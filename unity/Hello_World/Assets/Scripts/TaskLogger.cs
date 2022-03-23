using System.Linq;
using DilmerGames.Core.Singletons;
using TMPro;
using UnityEngine;
using System;

/// <summary>
/// Contains methods for updating the task AR display.
/// </summary>
public class TaskLogger : Singleton<Logger>
{
    [SerializeField]
    private TextMeshProUGUI debugAreaText = null;

    [SerializeField]
    private bool enableDebug = false;

    [SerializeField]
    private int maxLines = 15;

    void Awake()
    {
        if (debugAreaText == null)
        {
            debugAreaText = GetComponent<TextMeshProUGUI>();
        }
    }

    void OnEnable()
    {
        debugAreaText.enabled = enableDebug;
    }

    /// <summary>
    /// Updates the task display with the given TaskUpdateMessage information.
    /// </summary>
    public void UpdateTaskDisplay(TaskUpdateMessage taskUpdateMessage)
    {
        ClearLines();

        // Display the current task
        debugAreaText.text += $"<color=\"green\"> {"Current task: "} {taskUpdateMessage._taskName}</color>\n";

        // Display the current activity being performed
        if (taskUpdateMessage._currActivity != taskUpdateMessage._nextActivity)
        {
            debugAreaText.text += $"<color=\"red\"> {"Current activity: "} {taskUpdateMessage._currActivity}</color>\n";
        }
        else
        {
            debugAreaText.text += $"<color=\"green\"> {"Current activity: "} {taskUpdateMessage._currActivity}</color>\n";
        }

        debugAreaText.text += $"<color=\"yellow\"> {"Next activity to perform: "} {taskUpdateMessage._nextActivity}</color>\n";
        debugAreaText.text += $"<color=\"white\"> {"Steps: "}</color>\n";

        // Display this task's steps
        int stepIndex = taskUpdateMessage._steps.FindIndex(a => a.Contains(taskUpdateMessage._currStep));
        for (int i = 0; i < taskUpdateMessage._numSteps; i++)
        {
            if (i < stepIndex)
            {
                // We've already completed this step so color the step green
                debugAreaText.text += $"<color=\"green\"> {"  "} {i + 1} {") "} {taskUpdateMessage._steps[i]}</color>\n";
            }
            else if (i == stepIndex)
            {
                // Current step, so color it yellow
                debugAreaText.text += $"<color=\"yellow\"> {"  "} {i + 1} {") "} {taskUpdateMessage._steps[i]}</color>\n";
            }
            else
            {
                // Future steps, so color them white
                debugAreaText.text += $"<color=\"white\"> {"  "} {i + 1} {") "} {taskUpdateMessage._steps[i]}</color>\n";
            }
        }
    }

    private void ClearLines()
    {
        debugAreaText.text = string.Empty;
    }
}