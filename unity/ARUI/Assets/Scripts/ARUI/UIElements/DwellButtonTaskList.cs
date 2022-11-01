using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.UI;
using Microsoft.MixedReality.Toolkit.Utilities.Solvers;
using System;
using System.Collections;
using UnityEngine;
using UnityEngine.UIElements;

public class DwellButtonTaskList : MonoBehaviour
{
    private Shapes.Disc loadingDisc;

    private float startingAngle;
    private bool isLookingAtBtn = false;

    private Material btnBGMat;
    private Color baseColor;
    
    public void Awake()
    {
        Shapes.Disc disc = GetComponentInChildren<Shapes.Disc>(true);
        loadingDisc = disc;

        startingAngle = loadingDisc.AngRadiansStart;

        btnBGMat = GetComponentInChildren<MeshRenderer>().material;
        baseColor = btnBGMat.color;
    }

    private void Update()
    {
        CurrentlyLooking(FollowEyeTarget.Instance.currentHit == EyeTarget.orbtasklistButton);

        if (!isLookingAtBtn && TaskListManager.Instance.GetIsTaskListActive())
        {
            loadingDisc.AngRadiansEnd = 6.24f;
            loadingDisc.Color = new Color(0.8f, 0.8f, 0.8f);
        }
        else if (!isLookingAtBtn && !TaskListManager.Instance.GetIsTaskListActive())
        {
            loadingDisc.AngRadiansEnd = startingAngle;
            loadingDisc.Color = Color.white;
        }
    }

    private void CurrentlyLooking(bool looking)
    {
        if (looking&&!isLookingAtBtn)
        {
            isLookingAtBtn = true;
            StartCoroutine(Dwelling());
        }  

        if (!looking)
        {
            isLookingAtBtn = false;
            StopCoroutine(Dwelling());
            btnBGMat.color = baseColor;
        }

        isLookingAtBtn = looking;
    }
    
    private IEnumerator Dwelling()
    {
        AudioManager.Instance.PlaySound(transform.position, SoundType.confirmation);

        btnBGMat.color = baseColor - new Color(0.2f, 0.2f, 0.2f, 1);
            
        bool success = false;
        float duration = 6.24f/4f; //full circle in radians

        float elapsed = 0f;
        while (isLookingAtBtn && elapsed < duration)
        {
            if (CoreServices.InputSystem.EyeGazeProvider.GazeTarget == null)
                break;

            elapsed += Time.deltaTime;
            loadingDisc.AngRadiansEnd = elapsed*4f;
            loadingDisc.Color = Color.white;

            if (elapsed>duration && isLookingAtBtn)
                success = true;

            yield return null;
        }

        if (success)
            AngelARUI.Instance.ToggleTasklist();

        loadingDisc.AngRadiansEnd = startingAngle;
        btnBGMat.color = baseColor;
    }

    #region Getter and Setter 

    public bool GetIsLookingAtBtn() => isLookingAtBtn;

    #endregion


}