using UnityEngine;
using System.Collections.Generic;
using System.Collections;
using System;

public class ExampleScript : MonoBehaviour
{
    public bool Automate = true;
    private int _currentTask = 0;

    private void Start()
    {
        if (Automate)
            StartCoroutine(RunAutomatedTests());
    }

    private IEnumerator RunAutomatedTests()
    {
        yield return new WaitForSeconds(1f);

        //AngelARUI.Instance.PrintVMDebug = true;

        //test with dummy data
        var taskIDs = new List<string> { "Pinwheels", "Coffee", "Oatmeal", "Quesadilla", "Tea" };
        var allJsonTasks = new Dictionary<string, string>();
        foreach (string  taskID in taskIDs)
        {
            var jsonTextFile = Resources.Load<TextAsset>("Text/" + taskID);
            allJsonTasks.Add(taskID, jsonTextFile.text);
        }

        AngelARUI.Instance.InitManual(allJsonTasks);

        yield return new WaitForSeconds(2f);

        AngelARUI.Instance.PlayMessageAtOrb("This is a test of a very long text. I am just going to continue talking until somebody says stop or if I am getting interrupted by another incoming message. I enjoy helping people, so ask me any question you want about the tasks.");
    }

#if UNITY_EDITOR

    /// <summary>
    /// Listen to Keyevents for debugging(only in the Editor)
    /// </summary>
    public void Update()
    {
        CheckForRecipeChange();

        if (Input.GetKeyUp(KeyCode.O))
        {
            //test with dummy data
            var taskIDs = new List<string> { "Pinwheels", "Coffee", "Oatmeal", "Quesadilla", "Tea" };
            var allJsonTasks = new Dictionary<string, string>();
            foreach (string taskID in taskIDs)
            {
                var jsonTextFile = Resources.Load<TextAsset>("Text/" + taskID);
                allJsonTasks.Add(taskID, jsonTextFile.text);
            }

            AngelARUI.Instance.InitManual(allJsonTasks);
        }

        // Example how to step forward/backward in tasklist. 
        if (Input.GetKeyUp(KeyCode.RightArrow))
        {
            _currentTask++;
            AngelARUI.Instance.GoToStep("Pinwheels", _currentTask);
        }
        else if (Input.GetKeyUp(KeyCode.LeftArrow))
        {
            _currentTask--;
            AngelARUI.Instance.GoToStep("Pinwheels", _currentTask);
        }

        if (Input.GetKeyUp(KeyCode.V))
        {
            AngelARUI.Instance.SetViewManagement(!AngelARUI.Instance.IsVMActiv);
        }

        if (Input.GetKeyUp(KeyCode.A))
        {
            AngelARUI.Instance.DebugShowEyeGazeTarget(false);
        }
        if (Input.GetKeyUp(KeyCode.S))
        {
            AngelARUI.Instance.DebugShowEyeGazeTarget(true);
        }
        if (Input.GetKeyUp(KeyCode.D))
        {
            AngelARUI.Instance.PrintVMDebug = true;
        }
        if (Input.GetKeyUp(KeyCode.F))
        {
            AngelARUI.Instance.PlayMessageAtOrb("This is a test");
        }
    }

    private void CheckForRecipeChange()
    {
        // Example how to use the NLI confirmation dialogue
        if (Input.GetKeyUp(KeyCode.Alpha1))
        {
            AngelARUI.Instance.SetCurrentObservedTask("Pinwheels");
        }

        if (Input.GetKeyUp(KeyCode.Alpha2))
        {
            AngelARUI.Instance.SetCurrentObservedTask("Coffee");
        }

        if (Input.GetKeyUp(KeyCode.Alpha3))
        {
            AngelARUI.Instance.SetCurrentObservedTask("Oatmeal");
        }

        if (Input.GetKeyUp(KeyCode.Alpha4))
        {
            AngelARUI.Instance.SetCurrentObservedTask("Tea");
        }

        if (Input.GetKeyUp(KeyCode.Alpha5))
        {
            AngelARUI.Instance.SetCurrentObservedTask("Quesadilla");
        }
    }

#endif
}
