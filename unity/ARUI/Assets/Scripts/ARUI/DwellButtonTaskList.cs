using Microsoft.MixedReality.Toolkit.Input;
using System;
using System.Collections;
using UnityEngine;

public class DwellButtonTaskList : MonoBehaviour
{
    private Shapes.Disc loadingDisc;

    private float start;
    private EyeTrackingTarget eyeEvents;

    private float startingAngle;
    private bool isLooking = false;

    public void Awake()
    {
        Shapes.Disc[] discs = GetComponentsInChildren<Shapes.Disc>(true);
        loadingDisc = discs[1];
        startingAngle = loadingDisc.AngRadiansStart;

        eyeEvents = gameObject.GetComponentInChildren<EyeTrackingTarget>();
    }

    public void Start()
    { 
        eyeEvents.OnLookAtStart.AddListener(delegate { CurrentlyLooking(true); });
        eyeEvents.OnLookAway.AddListener(delegate { CurrentlyLooking(false); });
    }

    private void CurrentlyLooking(bool looking)
    {
        if (looking&&!isLooking)
        {
            isLooking = true;
            StartCoroutine(Dwelling());
        }  

        if (!looking&&isLooking)
        {
            StopCoroutine(Dwelling());
            ResetDwelling();
        }

        isLooking = looking;
        Orb.Instance.SetFollow(!looking);
    }

    // every 2 seconds perform the print()
    private IEnumerator Dwelling()
    {
        bool success = false;
        float duration = 6.24f/5f; //full circle in radians

        float elapsed = 0f;
        while (isLooking && elapsed < duration)
        {
            elapsed += Time.deltaTime;
            loadingDisc.AngRadiansEnd = elapsed*5f;
            loadingDisc.UpdateMesh(true);

            if (elapsed>duration)
                success = true;

            yield return null;
        }

        if (success)
        {
            Debug.Log("Dwelling success!");
            AngelARUI.Instance.ToggleTasklist();
        } else
            ResetDwelling();


    }

    private void ResetDwelling()
    {
        Debug.Log("Dwelling failed!");
        loadingDisc.AngRadiansEnd = startingAngle;
        loadingDisc.meshOutOfDate = true;

        isLooking = false;
    }
}