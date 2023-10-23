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
            StartCoroutine(RunTasksAtRuntime());
    }

    private IEnumerator RunTasksAtRuntime()
    {
        yield return new WaitForSeconds(1f);
        AngelARUI.Instance.PrintVMDebug = true;

        AngelARUI.Instance.InitManual(new List<string> { "Pinwheels", "Coffee", "Oatmeal", "Quesadilla", "Tea" });
        ManualManager.Instance.SetMenuActive(!ManualManager.Instance.MenuActive);

        AngelARUI.Instance.PlayMessageAtOrb("This is a test of a very long text. I am just going to continue talking until somebody says stop or if I am getting interrupted by another incoming message. I enjoy helping people, so ask me any question you want about the tasks.");

        //ngelARUI.Instance.SetNotification(NotificationType.warning, "Hello, this is a wanrning");

    }

#if UNITY_EDITOR

    /// <summary>
    /// Listen to Keyevents for debugging(only in the Editor)
    /// </summary>
    public void Update()
    {
        CheckForRecipeChange();


        if (Input.GetKeyUp(KeyCode.M))
        {
            ManualManager.Instance.SetMenuActive(!ManualManager.Instance.MenuActive);
        }

        if (Input.GetKeyUp(KeyCode.O))
        {
            AngelARUI.Instance.InitManual(new List<string> { "Pinwheels", "Coffee", "Oatmeal", "Quesadilla", "Tea" });
        }

        if (Input.GetKeyUp(KeyCode.U))
        {
            AngelARUI.Instance.IsGuidanceActive = !AngelARUI.Instance.IsGuidanceActive;
        }

        // Example how to step forward/backward in tasklist. 
        if (Input.GetKeyUp(KeyCode.RightArrow))
        {
            _currentTask++;
            AngelARUI.Instance.GoToStep("Pinwheels", _currentTask);
            Debug.Log("GOto:" + _currentTask);
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
            AngelARUI.Instance.ShowDebugEyeGazeTarget(false);
        }
        if (Input.GetKeyUp(KeyCode.S))
        {
            AngelARUI.Instance.ShowDebugEyeGazeTarget(true);
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
            AngelARUI.Instance.SetCurrentDetectedTask("Pinwheels");
        }

        if (Input.GetKeyUp(KeyCode.Alpha2))
        {
            AngelARUI.Instance.SetCurrentDetectedTask("Coffee");
        }

        if (Input.GetKeyUp(KeyCode.Alpha3))
        {
            AngelARUI.Instance.SetCurrentDetectedTask("Oatmeal");
        }

        if (Input.GetKeyUp(KeyCode.Alpha4))
        {
            AngelARUI.Instance.SetCurrentDetectedTask("Tea");
        }

        if (Input.GetKeyUp(KeyCode.Alpha5))
        {
            AngelARUI.Instance.SetCurrentDetectedTask("Quesadilla");
        }
    }

#endif
}
