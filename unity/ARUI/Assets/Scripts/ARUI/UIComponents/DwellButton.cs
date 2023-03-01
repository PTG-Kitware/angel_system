using Microsoft.MixedReality.Toolkit;
using System;
using System.Collections;
using System.Diagnostics.Eventing.Reader;
using UnityEngine;
using UnityEngine.Events;

public enum DwellButtonType
{
    Toggle=0,
    Select =1
}

public class DwellButton : MonoBehaviour
{
    private bool isLookingAtBtn = false;

    private float dwellSeconds = 4f;
    private DwellButtonType type = DwellButtonType.Toggle;
    private Shapes.Disc loadingDisc;

    private float startingAngle;

    /***Btn Design***/
    private Material btnBGMat;
    private Color baseColor;

    public EyeTarget thisTarget;
    public UnityEvent selectEvent;

    private BoxCollider btnCollider;
    public float Width { get { return btnCollider.size.y; } }

    public void Awake()
    {
        Shapes.Disc disc = GetComponentInChildren<Shapes.Disc>(true);
        loadingDisc = disc;

        startingAngle = loadingDisc.AngRadiansStart;

        btnBGMat = GetComponentInChildren<MeshRenderer>().material;
        baseColor = btnBGMat.color;

        selectEvent = new UnityEvent();

        btnCollider = GetComponentInChildren<BoxCollider>(true);
    }

    public void InitializeButton(EyeTarget target, UnityAction btnSelectEvent)
    {
        thisTarget = target;
        selectEvent.AddListener(btnSelectEvent);
    }

    private void Update()
    {
        CurrentlyLooking(FollowEyeTarget.Instance.currentHit == thisTarget);
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
        float duration = 6.24f/dwellSeconds; //full circle in radians

        float elapsed = 0f;
        while (isLookingAtBtn && elapsed < duration)
        {
            if (CoreServices.InputSystem.EyeGazeProvider.GazeTarget == null)
                break;

            elapsed += Time.deltaTime;
            loadingDisc.AngRadiansEnd = elapsed* dwellSeconds;
            loadingDisc.Color = Color.white;

            if (elapsed>duration && isLookingAtBtn)
                success = true;

            yield return null;
        }

        if (success)
            selectEvent.Invoke();

        if (type.Equals(DwellButtonType.Toggle))
            loadingDisc.AngRadiansEnd = startingAngle;
        else
            loadingDisc.AngRadiansEnd = 0;

        btnBGMat.color = baseColor;
    }

    #region Getter and Setter 

    public bool GetIsLookingAtBtn() => isLookingAtBtn;

    public void SetDwellButtonType(DwellButtonType type) => this.type = type;

    internal void SetSelected(bool selected)
    {
        if (selected)
        {
            loadingDisc.AngRadiansEnd = 6.24f;
            loadingDisc.Color = new Color(0.8f, 0.8f, 0.8f);
        } else
        {
            loadingDisc.AngRadiansEnd = startingAngle;
            loadingDisc.Color = Color.white;
        }
    }

    #endregion


}