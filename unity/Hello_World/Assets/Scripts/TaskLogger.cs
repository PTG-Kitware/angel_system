using System.Linq;
using DilmerGames.Core.Singletons;
using TMPro;
using UnityEngine;
using System;
using RosMessageTypes.Angel;


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
    public void UpdateTaskDisplay(TaskUpdateMsg taskUpdateMessage)
    {
        ClearLines();

        // Display the current task
        debugAreaText.text += $"<color=\"green\"> {"Current task: "} {taskUpdateMessage.task_name}</color>\n";

        // Display the current activity being performed
        if (taskUpdateMessage.current_activity != taskUpdateMessage.next_activity)
        {
            debugAreaText.text += $"<color=\"red\"> {"Current activity: "} {taskUpdateMessage.current_activity}</color>\n";
        }
        else
        {
            debugAreaText.text += $"<color=\"green\"> {"Current activity: "} {taskUpdateMessage.current_activity}</color>\n";
        }

        debugAreaText.text += $"<color=\"yellow\"> {"Next activity to perform: "} {taskUpdateMessage.next_activity}</color>\n";
        debugAreaText.text += $"<color=\"white\"> {"Steps: "}</color>\n";

        // Display this task's steps
        int stepIndex = Array.FindIndex(taskUpdateMessage.steps, a => a.Contains(taskUpdateMessage.current_step));

        for (int i = 0; i < taskUpdateMessage.steps.Length; i++)
        {
            if (i < stepIndex)
            {
                // We've already completed this step so color the step green
                debugAreaText.text += $"<color=\"green\"> {"  "} {i + 1} {") "} {taskUpdateMessage.steps[i]}</color>\n";
            }
            else if (i == stepIndex)
            {
                // Current step, so color it yellow
                debugAreaText.text += $"<color=\"yellow\"> {"  "} {i + 1} {") "} {taskUpdateMessage.steps[i]}</color>\n";
            }
            else
            {
                // Future steps, so color them white
                debugAreaText.text += $"<color=\"white\"> {"  "} {i + 1} {") "} {taskUpdateMessage.steps[i]}</color>\n";
            }
        }
    }

    private void ClearLines()
    {
        debugAreaText.text = string.Empty;
    }
}