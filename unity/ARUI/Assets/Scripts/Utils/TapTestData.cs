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

    private bool actionInProcess = false;

    private bool taskInit = false;
    private int currentTask = 0;

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
        if (eventData!=null && eventData.InputSource.SourceType.Equals(InputSourceType.Hand) && !actionInProcess)
        {
            actionInProcess = true;
            StartCoroutine(AddIfHit(eventData));
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


    public void OnActionStarted(BaseInputEventData eventData) {}


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
       
        if (Input.GetKeyUp(KeyCode.H))
        {
            string id = "test_circular_arrow";
            // Method 1
            //GameObject arrObj = ArrowManager.Instance.SpawnCircularArrow(id, new Vector3(0.05f, -0.2f, 0.3f), new Vector3(0, 1, 0), 0.005f, 180, 0.05f);

            // Method 2
            ArrowEntity arrowEntity = EntityManager.Instance.CreateArrowEntity(id);
            arrowEntity.SetAsCircularArrow(new Vector3(0.05f, -0.2f, 0.3f), new Vector3(0, 1, 0), 180, 0.05f);
        }

        if (Input.GetKeyUp(KeyCode.J))
        {
            Logger.Instance.LogInfo(EntityManager.Instance.PrintDict());
        }

        if (Input.GetKeyUp(KeyCode.K))
        {
            string id = "test_circular_arrow";
            EntityManager.Instance.Remove(id);
        }

        // Pointer arrow test
        if (Input.GetKeyUp(KeyCode.B))
        {
            ArrowEntity arrowEntity = EntityManager.Instance.CreateArrowEntity("pointer_arrow");
            arrowEntity.SetAsPathArrow(new Vector3(0f, 0f, 0f), new Vector3(0f, 0.3f, 0f), 0.001f, true);
            arrowEntity = EntityManager.Instance.CreateArrowEntity("pointer1_arrow");
            arrowEntity.SetAsPathArrow(new Vector3(0.1f, 0f, 0.1f), new Vector3(0.1f, 0.3f, 0.1f), 0.001f, false);
            arrowEntity = EntityManager.Instance.CreateArrowEntity("pointer2_arrow");
            arrowEntity.SetAsPathArrow(new Vector3(0.2f, 0f, 0.2f), new Vector3(0.2f, 0.3f, 0.2f), 0.1f, false);
            arrowEntity = EntityManager.Instance.CreateArrowEntity("pointer3_arrow");
            arrowEntity.SetAsPointerArrow(new Vector3(0.3f, 0, 0.3f));
            //ArrowManager.Instance.SpawnPointerArrow("pointer_arrow", new Vector3(0f, 0f, 0f), 0.005f);
        }


        if (Input.GetKeyUp(KeyCode.N))
        {
            string id = "test_circular_arrow";
            ArrowEntity arrowEntity = EntityManager.Instance.Get(id) as ArrowEntity;
            arrowEntity.SetAnimation(1);
            arrowEntity.SetColor(Color.red);
        }

    }

}
