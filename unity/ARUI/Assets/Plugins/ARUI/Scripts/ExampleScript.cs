using UnityEngine;
using System.Collections.Generic;
using System.Collections;
using System;

public class ExampleScript : MonoBehaviour
{
    public bool Automate = true;

    private Dictionary<string, int> _currentStepMap;
    private string _currentTask = "";
    private void Start()
    {
        if (Automate)
            StartCoroutine(RunAutomatedTests());
    }

    private IEnumerator RunAutomatedTests()
    {
        yield return new WaitForSeconds(1f);

        AngelARUI.Instance.DebugShowEyeGazeTarget(true);
        AngelARUI.Instance.PrintVMDebug = false;

        //test with dummy data
        var taskIDs = new List<string> { "Pinwheels", "Coffee", "Oatmeal", "Quesadilla", "Tea" };
        _currentStepMap = new Dictionary<string, int> { 
            { "Pinwheels", 0 }, { "Coffee", 0 },
            { "Oatmeal", 0 }, { "Quesadilla", 0 }, { "Tea", 0 }};
        _currentTask = "Pinwheels";

        var allJsonTasks = new Dictionary<string, string>();
        foreach (string  taskID in taskIDs)
        {
            var jsonTextFile = Resources.Load<TextAsset>("Text/" + taskID);
            allJsonTasks.Add(taskID, jsonTextFile.text);
        }

        AngelARUI.Instance.InitManual(allJsonTasks);

        yield return new WaitForSeconds(2f);

        AngelARUI.Instance.PlayMessageAtOrb("This is a test of a very long text. I am just going to continue talking until somebody says stop or if I am getting interrupted by another incoming message. I enjoy helping people, so ask me any question you want about the tasks.");

        yield return new WaitForSeconds(5f);

        AngelARUI.Instance.SetCurrentObservedTask("Tea");
        _currentTask = "Tea";
    
        yield return new WaitForSeconds(1f);

        _currentStepMap[_currentTask]++;
        AngelARUI.Instance.GoToStep(_currentTask, _currentStepMap[_currentTask]);

        yield return new WaitForSeconds(2f);

        AngelARUI.Instance.SetNotification("You are skipping the this step.");

        _currentStepMap[_currentTask]++;
        AngelARUI.Instance.GoToStep(_currentTask, _currentStepMap[_currentTask]);

        yield return new WaitForSeconds(3f);

        _currentStepMap[_currentTask]++;
        AngelARUI.Instance.GoToStep(_currentTask, _currentStepMap[_currentTask]);

        yield return new WaitForSeconds(3f);

        _currentStepMap[_currentTask]++;
        AngelARUI.Instance.GoToStep(_currentTask, _currentStepMap[_currentTask]);

        yield return new WaitForSeconds(2f);

        _currentStepMap["Pinwheels"]++;
        AngelARUI.Instance.GoToStep("Pinwheels", _currentStepMap["Pinwheels"]);

        yield return new WaitForSeconds(3f);

        _currentStepMap[_currentTask]++;
        AngelARUI.Instance.GoToStep(_currentTask, _currentStepMap[_currentTask]);
        AngelARUI.Instance.RemoveNotification();

        yield return new WaitForSeconds(2f);

        AngelARUI.Instance.SetCurrentObservedTask("Pinwheels");
    }

    private IEnumerator RunAutomatedTests2()
    {
        yield return new WaitForSeconds(1f);

        AngelARUI.Instance.DebugShowEyeGazeTarget(true);
        AngelARUI.Instance.PrintVMDebug = false;

        //test with dummy data
        var taskIDs = new List<string> { "Pinwheels"};
        _currentStepMap = new Dictionary<string, int> {
            { "Pinwheels", 0 }};
        _currentTask = "Pinwheels";

        var allJsonTasks = new Dictionary<string, string>();
        foreach (string taskID in taskIDs)
        {
            var jsonTextFile = Resources.Load<TextAsset>("Text/" + taskID);
            allJsonTasks.Add(taskID, jsonTextFile.text);
        }

        AngelARUI.Instance.InitManual(allJsonTasks);

        yield return new WaitForSeconds(2f);

        AngelARUI.Instance.PlayMessageAtOrb("This is a test of a very long text. I am just going to continue talking until somebody says stop or if I am getting interrupted by another incoming message. I enjoy helping people, so ask me any question you want about the tasks.");

        yield return new WaitForSeconds(5f);

        AngelARUI.Instance.SetCurrentObservedTask("Pinwheels");
        _currentTask = "Pinwheels";

        yield return new WaitForSeconds(3f);

        _currentStepMap[_currentTask]++;
        AngelARUI.Instance.GoToStep(_currentTask, _currentStepMap[_currentTask]);

        yield return new WaitForSeconds(3f);

        AngelARUI.Instance.SetNotification("You are skipping the this step.");
        _currentStepMap[_currentTask]++;
        AngelARUI.Instance.GoToStep(_currentTask, _currentStepMap[_currentTask]);

        yield return new WaitForSeconds(3f);

        _currentStepMap[_currentTask]++;
        AngelARUI.Instance.GoToStep(_currentTask, _currentStepMap[_currentTask]);

        yield return new WaitForSeconds(3f);

        _currentStepMap[_currentTask]++;
        AngelARUI.Instance.GoToStep(_currentTask, _currentStepMap[_currentTask]);
        AngelARUI.Instance.RemoveNotification();

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
            _currentStepMap = new Dictionary<string, int> {
            { "Pinwheels", 0 }, { "Coffee", 0 },
            { "Oatmeal", 0 }, { "Quesadilla", 0 }, { "Tea", 0 }};
            _currentTask = "Pinwheels";

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
            _currentStepMap[_currentTask]++;
            AngelARUI.Instance.GoToStep(_currentTask, _currentStepMap[_currentTask]);
        }
        else if (Input.GetKeyUp(KeyCode.LeftArrow))
        {
            _currentStepMap[_currentTask]--;
            AngelARUI.Instance.GoToStep(_currentTask, _currentStepMap[_currentTask]);
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
            AngelARUI.Instance.PrintVMDebug = !AngelARUI.Instance.PrintVMDebug;
        }
        if (Input.GetKeyUp(KeyCode.F))
        {
            AngelARUI.Instance.PlayMessageAtOrb("This is a test");
        }

        if (Input.GetKeyUp(KeyCode.Alpha9))
        {
            AngelARUI.Instance.SetNotification("You skipped the last step.");
        }
        if (Input.GetKeyUp(KeyCode.Alpha0))
        {
            AngelARUI.Instance.RemoveNotification();
        }
    }

    private void CheckForRecipeChange()
    {
        // Example how to use the NLI confirmation dialogue
        if (Input.GetKeyUp(KeyCode.Alpha1))
        {
            AngelARUI.Instance.SetCurrentObservedTask("Pinwheels");
            _currentTask = "Pinwheels";
        }

        if (Input.GetKeyUp(KeyCode.Alpha2))
        {
            AngelARUI.Instance.SetCurrentObservedTask("Coffee");
            _currentTask = "Coffee";
        }

        if (Input.GetKeyUp(KeyCode.Alpha3))
        {
            AngelARUI.Instance.SetCurrentObservedTask("Oatmeal");
            _currentTask = "Oatmeal";
        }

        if (Input.GetKeyUp(KeyCode.Alpha4))
        {
            AngelARUI.Instance.SetCurrentObservedTask("Tea");
            _currentTask = "Tea";
        }

        if (Input.GetKeyUp(KeyCode.Alpha5))
        {
            AngelARUI.Instance.SetCurrentObservedTask("Quesadilla");
            _currentTask = "Quesadilla";
        }
    }

#endif
}
