using DilmerGames.Core.Singletons;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.UI;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TestData : MonoBehaviour
{
    string[,] tasks =
    {
        {"0", "Measure 12 ounces of water in the liquid measuring cup"},
        {"1", "Pour the water from the liquid measuring cup into  the electric kettle"},
        {"1", "Turn on the electric kettle by pushing the button underneath the handle"},
        {"1", "Boil the water. The water is done boiling when the button underneath the handle pops up"},
        {"1", "While the water is boiling, assemble the filter cone. Place the dripper on top of a coffee mug"}, //4
        {"0", "Prepare the filter insert by folding the paper filter in half to create a semi-circle, and in half again to create a quarter-circle. Place the paper filter in the dripper and spread open to create a cone."},
        {"1", "Take the coffee filter and fold it in half to create a semi-circle"},
        {"1", "Folder the filter in half again to create a quarter-circle"},
        {"1", "Place the folded filter into the dripper such that the the point of the quarter-circle rests in the center of the dripper"},
        {"1", "Spread the filter open to create a cone inside the dripper"},
        {"0", "Place the dripper on top of the mug"},//10
        {"0","Weigh the coffee beans and grind until the coffee grounds are the consistency of coarse sand, about 20 seconds. Transfer the grounds to the filter cone."},
        {"1","Turn on the kitchen scale"},
         {"0"," Turn on the thermometer"},
         {"1"," Place the end of the thermometer into the water. The temperature should read 195-205 degrees Fahrenheit or between 91-96 degrees Celsius."},
         {"0","Pour the water over the coffee grounds"},
         {"0","Clean up the paper filter and coffee grounds"}, //16
    };

    private int currentTask = 0;

    private void Start()
    {
        AngelARUI.Instance.SetTasks(tasks);
    }

#if UNITY_EDITOR

    /// <summary>
    /// Listen to Keyevents for debugging (only in the editor)
    /// </summary>
    public void Update()
    {
        if (Input.GetKeyUp(KeyCode.RightArrow))
        {
            currentTask++;
            AngelARUI.Instance.SetCurrentTaskID(currentTask);
        }
        else if (Input.GetKeyUp(KeyCode.LeftArrow))
        {
            currentTask--;
            AngelARUI.Instance.SetCurrentTaskID(currentTask);
        }
        
        if (Input.GetKeyUp(KeyCode.Space))
        {
            AngelARUI.Instance.ToggleTasklist();
        }

        if (Input.GetKeyUp(KeyCode.D))
        {
            currentTask = tasks.GetLength(0) + 2;
            AngelARUI.Instance.SetAllTasksDone();
        }

        if (Input.GetKeyUp(KeyCode.R))
        {
            currentTask = UnityEngine.Random.Range(0, tasks.GetLength(0) + 2);
            AngelARUI.Instance.SetCurrentTaskID(currentTask);
        }
    }
#endif
}
