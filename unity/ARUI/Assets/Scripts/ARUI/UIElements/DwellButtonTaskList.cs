using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.UI;
using System;
using System.Collections;
using UnityEngine;

public class DwellButtonTaskList : MonoBehaviour
{
    private Shapes.Disc loadingDisc;
    private EyeTrackingTarget eyeEvents;

    private float startingAngle;
    private bool isLooking = false;

    public void Awake()
    {
        Shapes.Disc disc = GetComponentInChildren<Shapes.Disc>(true);
        loadingDisc = disc;

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

        if (isLooking)
            StartCoroutine(StartSmoothFollow());
        else
            Orb.Instance.SetFollowActive(false);
    }

    private IEnumerator StartSmoothFollow()
    {
        yield return new WaitForSeconds(1f);

        Orb.Instance.SetFollowActive(true);
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
            loadingDisc.Color = Color.white;

            if (elapsed>duration && isLooking)
                success = true;

            yield return null;
        }

        if (success)
            AngelARUI.Instance.ToggleTasklist();


        loadingDisc.AngRadiansEnd = startingAngle;

    }

    public void Update()
    {
        if (!isLooking && TaskListManager.Instance.IsTaskListActive())
        {
            loadingDisc.AngRadiansEnd = 6.24f;
            loadingDisc.Color = new Color(0.8f,0.8f,0.8f);
        }
        else if (!isLooking && !TaskListManager.Instance.IsTaskListActive())
        {
            loadingDisc.AngRadiansEnd =startingAngle;
            loadingDisc.Color = Color.white;
        }
    }

}