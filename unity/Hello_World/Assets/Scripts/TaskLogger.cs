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
        debugAreaText.text += $"<color=\"green\"> {"Current task: "} {taskUpdateMessage.task_name}</color>\n\n";

        // Display the task description
        debugAreaText.text += $"<color=\"white\"> {"Task description: "} {taskUpdateMessage.task_description}</color>\n\n";

        // Display the items required
        debugAreaText.text += $"<color=\"white\"> {"Ingredients: "}</color>\n";
        for (int i = 0; i < taskUpdateMessage.task_items.Length; i++)
        {
            debugAreaText.text += $"<color=\"white\"> {" - "} {taskUpdateMessage.task_items[i].quantity} {" "} {taskUpdateMessage.task_items[i].item_name}</color>\n";
        }

        // Display the current activity being performed
        if (taskUpdateMessage.current_activity != taskUpdateMessage.next_activity)
        {
            debugAreaText.text += $"<color=\"red\"> {"\nCurrent activity: "} {taskUpdateMessage.current_activity}</color>\n";
        }
        else
        {
            debugAreaText.text += $"<color=\"green\"> {"\nCurrent activity: "} {taskUpdateMessage.current_activity}</color>\n";
        }

        // Display this task's steps
        debugAreaText.text += $"<color=\"white\"> {"Steps: "}</color>\n";
        int stepIndex = Array.FindIndex(taskUpdateMessage.steps, a => a.Contains(taskUpdateMessage.current_step));
        for (int i = 0; i < taskUpdateMessage.steps.Length; i++)
        {
            if (i < stepIndex)
            {
                // We've already completed this step, so color the step green
                debugAreaText.text += $"<color=\"green\"> {"  "} {i + 1} {") "} {taskUpdateMessage.steps[i]}</color>\n";
            }
            else if (i == stepIndex)
            {
                // Current step, so color it yellow
                debugAreaText.text += $"<color=\"yellow\"> {"  "} {i + 1} {") "} {taskUpdateMessage.steps[i]}</color>";

                // Display the current time remaining on this step, if it is a time based step
                if (taskUpdateMessage.time_remaining_until_next_task >= 0)
                {
                    debugAreaText.text += $"<color=\"yellow\"> {", time remaining: "} {taskUpdateMessage.time_remaining_until_next_task} {"\n"} </color>";
                }
                else
                {
                    debugAreaText.text += $"\n";
                }
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