using Microsoft.MixedReality.Toolkit.Input;
using System;
using System.Collections;
using UnityEngine;

public class DwellButtonTaskList : MonoBehaviour
{
    private Shapes.Disc loadingDisc;

    private EyeTrackingTarget eyeEvents;

    private float startingAngle;
    private bool isLooking = false;

    private bool taskListActive = false;

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

        if (!looking)
        {
            isLooking = false;
            StopCoroutine(Dwelling());
        }

        isLooking = looking;
        Orb.Instance.SetFollowActive(!looking);
    }

    private IEnumerator Dwelling()
    {
        AudioManager.Instance.PlaySound(transform.position, SoundType.confirmation);

        bool success = false;
        float duration = 6.24f/4f; //full circle in radians

        float elapsed = 0f;
        while (isLooking && elapsed < duration)
        {
            elapsed += Time.deltaTime;
            loadingDisc.AngRadiansEnd = elapsed*4f;

            if (elapsed>duration && isLooking)
                success = true;

            yield return null;
        }

        if (success)
        {
            AngelARUI.Instance.ToggleTasklist();
        }

        loadingDisc.AngRadiansEnd = startingAngle;

    }

}