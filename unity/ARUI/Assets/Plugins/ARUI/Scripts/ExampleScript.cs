using UnityEngine;
using System.Collections.Generic;
using System.Collections;
using System;
using UnityEditor;
using UnityEngine.Events;

public class ExampleScript : MonoBehaviour
{
    public bool Automate = true;

    private Dictionary<string, int> _currentStepMap;
    private string _currentTask = "";

    public bool multipleTasks = false;

    private void Start()
    {
        if (Automate)
        {
            if (multipleTasks)
            {
                StartCoroutine(RunAutomatedTestsRecipes());
            } else
            {
                StartCoroutine(RunAutomatedTestsMaintenance());
            }
        }
    }

    private IEnumerator RunAutomatedTestsRecipes()
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
            var jsonTextFile = Resources.Load<TextAsset>("Manuals/" + taskID);
            allJsonTasks.Add(taskID, jsonTextFile.text);
        }

        AngelARUI.Instance.InitManual(allJsonTasks);

        AngelARUI.Instance.SetAgentThinking(true);

        yield return new WaitForSeconds(4f);

        AngelARUI.Instance.PlayDialogueAtAgent
            ("What is this in front of me?", "A grinder.");

        yield return new WaitForSeconds(5f);

        AngelARUI.Instance.SetCurrentObservedTask("Tea");
        _currentTask = "Tea";
    
        yield return new WaitForSeconds(1f);

        _currentStepMap[_currentTask]++;
        AngelARUI.Instance.GoToStep(_currentTask, _currentStepMap[_currentTask]);

        yield return new WaitForSeconds(2f);

        AngelARUI.Instance.SetWarningMessage("You are skipping the this step.");

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
        AngelARUI.Instance.RemoveWarningMessage();

        yield return new WaitForSeconds(2f);

        AngelARUI.Instance.SetCurrentObservedTask("Pinwheels");
    }

    #region Maintenance Tests
    private IEnumerator RunAutomatedTestsMaintenance()
    {
        yield return new WaitForSeconds(1f);

        AngelARUI.Instance.DebugShowEyeGazeTarget(true);

        AngelARUI.Instance.PrintVMDebug = false;

        AngelARUI.Instance.RegisterKeyword("Start Procedure", () => { StartCoroutine(SpeechCommandRegistrationTest()); });
        AngelARUI.Instance.RegisterKeyword("Toggle Manual", () => { AngelARUI.Instance.ToggleTaskOverview(); });
        AngelARUI.Instance.RegisterKeyword("Next Step", () => { GoToNextStepConfirmation(); });
        AngelARUI.Instance.RegisterKeyword("Previous Step", () => { GoToPreviousStepConfirmation(); });
        AngelARUI.Instance.RegisterKeyword("Coach", () => { AngelARUI.Instance.CallAgentToUser(); });

        AngelARUI.Instance.RegisterKeyword("Right", () => { AngelARUI.Instance.SetAgentMessageAlignment(MessageAlignment.LockRight); });
        AngelARUI.Instance.RegisterKeyword("Left", () => { AngelARUI.Instance.SetAgentMessageAlignment(MessageAlignment.LockLeft); });
        AngelARUI.Instance.RegisterKeyword("Automatic", () => { ShowTestMultipleChoice(); });
        AngelARUI.Instance.RegisterKeyword("toggle debug", () => { AngelARUI.Instance.SetLoggerVisible(!Logger.Instance.IsVisible); });

        AngelARUI.Instance.RegisterKeyword("Hello", () => { AngelARUI.Instance.PlayMessageAtAgent("How can I help you?"); });

        AngelARUI.Instance.RegisterDetectedObject(transform.GetChild(0).gameObject, "test");

        yield return new WaitForSeconds(4f);

        AngelARUI.Instance.PlayMessageAtAgent
            ("This is a very long message the user asked. to test how a very very very very long message with verrrrrrrryyyylooooonngg words would look like");

        yield return new WaitForSeconds(5f);

        AngelARUI.Instance.SetWarningMessage("This is a very very very very very very very very long warning");

        yield return new WaitForSeconds(3f);

        AngelARUI.Instance.RemoveWarningMessage();
    }

    private void GoToNextStepConfirmation()
    {
        if (_currentStepMap == null)
        {
            AngelARUI.Instance.PlayMessageAtAgent("No manual is set yet.");
            return;
        }

        AngelARUI.Instance.TryGetUserConfirmation("Please confirm if you are 100% confident that you want to go to the next step in the current task. We really need your confirmation.", () => DialogueTestConfirmed(), () => DialogueTestFailed());
    }

    private void GoToPreviousStepConfirmation()
    {
        if (_currentStepMap == null)
        {
            AngelARUI.Instance.PlayMessageAtAgent("No manual is set yet.");
            return;
        }

        _currentStepMap[_currentTask]--;
        AngelARUI.Instance.GoToStep(_currentTask, _currentStepMap[_currentTask]);
    }

    private void DialogueTestConfirmed()
    {
        if (_currentStepMap == null)
        {
            AngelARUI.Instance.PlayMessageAtAgent("No manual is set yet.");
            return;
        }

        _currentStepMap[_currentTask]++;
        AngelARUI.Instance.GoToStep(_currentTask, _currentStepMap[_currentTask]);
    }

    private void DialogueTestFailed()
    {
        AngelARUI.Instance.PlayMessageAtAgent("okay, i wont");
    }

    private void ShowTestMultipleChoice()
    {
        AngelARUI.Instance.TryGetUserMultipleChoice("Please select your preferred instruction alignment:",
            new List<string> { "Right", "Left", "Automatic" },
            new List<UnityAction>()
            {
                    () => AngelARUI.Instance.SetAgentMessageAlignment(MessageAlignment.LockRight),
                    () => AngelARUI.Instance.SetAgentMessageAlignment(MessageAlignment.LockLeft),
                    () => AngelARUI.Instance.SetAgentMessageAlignment(MessageAlignment.Auto),
            }, null, 30);
    }

    private IEnumerator SpeechCommandRegistrationTest()
    {
        AngelARUI.Instance.DebugLogMessage("The keyword was triggered!", true);
        AngelARUI.Instance.SetAgentThinking(true);

        yield return new WaitForSeconds(2);

        AngelARUI.Instance.SetAgentThinking(false);

        //test with dummy data
        var taskIDs = new List<string> { "Filter Inspection" };
        _currentStepMap = new Dictionary<string, int> {
            { "Filter Inspection", 0 }};
        _currentTask = "Filter Inspection";

        var allJsonTasks = new Dictionary<string, string>();
        foreach (string taskID in taskIDs)
        {
            var jsonTextFile = Resources.Load<TextAsset>("Manuals/" + taskID);
            allJsonTasks.Add(taskID, jsonTextFile.text);
        }

        AngelARUI.Instance.InitManual(allJsonTasks);

    }
    #endregion

#if UNITY_EDITOR

    /// <summary>
    /// Listen to Keyevents for debugging(only in the Editor)
    /// </summary>
    public void Update()
    {
        CheckForRecipeChange();

        if (Input.GetKeyUp(KeyCode.O))
        {
            var taskIDs= new List<string>();
            if (multipleTasks)
            {
                //test with dummy data
                taskIDs = new List<string> { "Pinwheels", "Coffee", "Oatmeal", "Quesadilla", "Tea" };
                _currentStepMap = new Dictionary<string, int> {
            { "Pinwheels", 0 }, { "Coffee", 0 },
            { "Oatmeal", 0 }, { "Quesadilla", 0 }, { "Tea", 0 }};
                _currentTask = "Pinwheels";
            } else
            {
                taskIDs = new List<string> { "Filter Inspection" };
                _currentStepMap = new Dictionary<string, int> {
            { "Filter Inspection", 0 }};
                _currentTask = "Filter Inspection";
            }

            var allJsonTasks = new Dictionary<string, string>();
            foreach (string taskID in taskIDs)
            {
                var jsonTextFile = Resources.Load("Manuals/" + taskID) as TextAsset;
                allJsonTasks.Add(taskID, jsonTextFile.text);
            }

            AngelARUI.Instance.InitManual(allJsonTasks);
        } 

        // Example how to step forward/backward in tasklist. 
        if (Input.GetKeyUp(KeyCode.RightArrow))
        {
            GoToNextStepConfirmation();
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

        if (Input.GetKeyUp(KeyCode.M))
        {
            AngelARUI.Instance.PlayMessageAtAgent("Hello",10);
        }

        if (Input.GetKeyUp(KeyCode.N))
        {
            ShowTestMultipleChoice();
        }

        if (Input.GetKeyUp(KeyCode.B))
        {
            AngelARUI.Instance.TryGetUserYesNoChoice("Are you done with the previous step?",
                null, () => { GoToPreviousStepConfirmation(); }, null, 30) ;
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
            AngelARUI.Instance.SetLoggerVisible(!Logger.Instance.IsVisible);
        }
        if (Input.GetKeyUp(KeyCode.F))
        {
            AngelARUI.Instance.PlayMessageAtAgent("This is a test");
        }

        if (Input.GetKeyUp(KeyCode.Alpha9))
        {
            AngelARUI.Instance.SetWarningMessage("You skipped the last step.");
        }
        if (Input.GetKeyUp(KeyCode.Alpha0))
        {
            AngelARUI.Instance.RemoveWarningMessage();
        }
    }


    private void CheckForRecipeChange()
    {
        if (!multipleTasks) { return; }

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
