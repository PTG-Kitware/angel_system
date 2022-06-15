using DilmerGames.Core.Singletons;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.UI;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TapTestData : MonoBehaviour, IMixedRealityInputActionHandler
{
    string[,] tasks =
    {
        {"0", "Place the kettle on the stove"},
        {"0","Turn the stove on"},
        {"1","Wait for the water to boil"},
        {"0","Turn the stove off"},
        {"0","Place the kettle on the trivet"},
    };

    string[,] tasks2 =
{
        {"0", "Boil the water"},
        {"0","Put the water in the cup"},
        {"0","Put the teabag in the cup"},
        {"0","Enjoy"},
    };

    private bool actionInProcess = false;

    private int currentTask = 0;

    public void Start()
    {
        //Set the task list
        AngelARUI.Instance.SetTasks(tasks,0);
    }


    /// <summary>
    /// Listen to Keyevents for debugging (only in the editor)
    /// </summary>
    public void Update()
    {
        if (!taskInit)
        {
            AngelARUI.Instance.SetTasks(tasks);
            AngelARUI.Instance.SetCurrentTaskID(0);
            taskInit = true;
        }
        else
        {
            if (Input.GetKeyUp(KeyCode.Space))
            {
                AngelARUI.Instance.ToggleTasklist();
            }
            else if (Input.GetKeyUp(KeyCode.RightArrow))
            {
                currentTask++;
                AngelARUI.Instance.SetCurrentTaskID(currentTask);
            }
            else if (Input.GetKeyUp(KeyCode.LeftArrow))
            {
                currentTask--;
                AngelARUI.Instance.SetCurrentTaskID(currentTask);
            }
        }

        if (Input.GetKeyUp(KeyCode.J))
        {
            Logger.Instance.LogInfo(EntityManager.Instance.PrintDict());
        }

        if (Input.GetKeyUp(KeyCode.D))
        {
            AngelARUI.Instance.SetTasks(tasks2, 0);
        }

        if (Input.GetKeyUp(KeyCode.S))
        {
            AngelARUI.Instance.SetTasks(tasks, 0);
        }
    }
    private IEnumerator AddIfHit(BaseInputEventData eventData)
    {
        var result = eventData.InputSource.Pointers[0].Result;
        if (result != null)
        {
            var hitPosition = result.Details.Point;

            if (result.CurrentPointerTarget?.layer == 31)
            {
                AngelARUI.Instance.UpdateDatabase("id", UpdateType.add, hitPosition, Utils.ConvertClassNumToStr((uint)UnityEngine.Random.Range(0, 99)));
            }
        }
        yield return new WaitForSeconds(1f);
        actionInProcess = false;
    }

    #region tap input registration 
    private void OnEnable()
    {
        CoreServices.InputSystem?.RegisterHandler<IMixedRealityInputActionHandler>(this);
        Logger.Instance.LogInfo("Generate Test Data using Tap gesture");
    }


    private void OnDisable()
    {
        CoreServices.InputSystem?.UnregisterHandler<IMixedRealityInputActionHandler>(this);
    }

    public void OnActionEnded(BaseInputEventData eventData)
    {
        if (eventData != null && eventData.InputSource.SourceType.Equals(InputSourceType.Hand) && !actionInProcess)
        {
            actionInProcess = true;
            StartCoroutine(AddIfHit(eventData));
        }
    }

    public void OnActionStarted(BaseInputEventData eventData) { }


    #endregion

}
