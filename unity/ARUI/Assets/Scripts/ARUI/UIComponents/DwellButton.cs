using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using System;
using System.Collections;
//using System.Diagnostics.Eventing.Reader;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.EventSystems;

public enum DwellButtonType
{
    Toggle=0,
    Select =1
}

/// <summary>
/// Button that can be triggered using touch or eye-gaze dwelling
/// </summary>
public class DwellButton : MonoBehaviour, IMixedRealityTouchHandler
{
    public bool IsInteractingWithBtn = false;
    public float Width { get { return btnCollider.size.y; } }

    private bool isLookingAtBtn = false;
    public bool GetIsLookingAtBtn{ get { return isLookingAtBtn; } }

    private bool isTouchingBtn = false;
    private bool touchable = false;

    private EyeTarget target;
    private UnityEvent selectEvent;
    private UnityEvent quarterSelectEvent;
    private BoxCollider btnCollider;
    public BoxCollider Collider { get { return btnCollider; } }
    private GameObject btnmesh;

    private float dwellSeconds = 4f;
    private DwellButtonType type = DwellButtonType.Toggle;
    private bool toggled = false;
    public bool Toggled { 
        set { 
            toggled = value; 
            SetSelected(value);
        } 
    }

    //*** Btn Dwelling Feedback 
    private Shapes.Disc loadingDisc;
    private float startingAngle;

    //*** Btn Push Feedback
    private Shapes.Disc pushConfiromationDisc;
    private float thickness; 

    //*** Btn Design
    private Material btnBGMat;
    private Color baseColor = new Color(0.5377358f, 0.5377358f, 0.5377358f,0.24f);
    private Color activeColor = new Color(0.7f, 0.7f, 0.8f, 0.4f);

    private void Awake()
    {
        Shapes.Disc[] discs = GetComponentsInChildren<Shapes.Disc>(true);
        loadingDisc = discs[0];
        pushConfiromationDisc = discs[1];
        pushConfiromationDisc.enabled = false;
        thickness = pushConfiromationDisc.Thickness;

        startingAngle = loadingDisc.AngRadiansStart;

        MeshRenderer mr = GetComponentInChildren<MeshRenderer>();
        btnBGMat = new Material(mr.material);
        btnBGMat.color = baseColor;
        mr.material = btnBGMat;

        selectEvent = new UnityEvent();
        quarterSelectEvent = new UnityEvent();

        btnCollider = GetComponentInChildren<BoxCollider>(true);
        btnmesh = transform.GetChild(0).gameObject;
    }

    public void InitializeButton(EyeTarget target, UnityAction btnSelectEvent, UnityAction btnHalfSelect, bool touchable, DwellButtonType type)
    {
        this.target = target;
        selectEvent.AddListener(btnSelectEvent);

        if (btnHalfSelect != null)
            quarterSelectEvent.AddListener(btnHalfSelect);

        this.touchable = touchable;
        this.type = type;

        if (touchable)
            gameObject.AddComponent<NearInteractionTouchable>();
    }

    private void Update()
    {
        UpdateCurrentlyLooking();
        IsInteractingWithBtn = isTouchingBtn || isLookingAtBtn;
    }

    private void UpdateCurrentlyLooking()
    {
        bool currentLooking = FollowEyeTarget.Instance.currentHit == target;

        if (currentLooking && !isLookingAtBtn && !isTouchingBtn)
        {
            isLookingAtBtn = true;
            StartCoroutine(Dwelling());
        }  

        if (!currentLooking || isTouchingBtn)
        {
            isLookingAtBtn = false;
            StopCoroutine(Dwelling());
            btnBGMat.color = baseColor;
        }

        isLookingAtBtn = currentLooking;
    }
    
    private IEnumerator Dwelling()
    {
        AudioManager.Instance.PlaySound(transform.position, SoundType.confirmation);

        btnBGMat.color = activeColor;

        bool halfEventEvoked = false;
        bool success = false;
        float duration = 6.24f/dwellSeconds; //full circle in radians

        float elapsed = 0f;
        while (!isTouchingBtn && isLookingAtBtn && elapsed < duration)
        {
            if (CoreServices.InputSystem.EyeGazeProvider.GazeTarget == null)
                break;

            elapsed += Time.deltaTime;
            loadingDisc.AngRadiansEnd = elapsed* dwellSeconds;
            loadingDisc.Color = Color.white;

            if (!halfEventEvoked && isLookingAtBtn && quarterSelectEvent != null && elapsed > (duration / 4))
            {
                halfEventEvoked = true;
                quarterSelectEvent.Invoke();
            }
                
            if (elapsed>duration && isLookingAtBtn)
                success = true;

            yield return null;
        }

        if (success)
        {
            selectEvent.Invoke();
            if (type == DwellButtonType.Toggle)
            {
                toggled = !toggled;
                SetSelected(toggled);
            }
        } else
        {
            btnBGMat.color = baseColor;

            if (type != DwellButtonType.Toggle || (type == DwellButtonType.Toggle && !toggled))
                SetSelected(false);
            else if (type == DwellButtonType.Toggle && toggled)
                SetSelected(true);
        }
    }

    public void OnTouchStarted(HandTrackingInputEventData eventData)
    {
        if (!touchable) return;
        isTouchingBtn = true;
        btnBGMat.color = activeColor;
        pushConfiromationDisc.enabled = true;
    }

    public void OnTouchCompleted(HandTrackingInputEventData eventData)
    {
        if (!touchable) return;
        isTouchingBtn = false;

        btnBGMat.color = baseColor;
        btnmesh.transform.localPosition = Vector3.zero;
        pushConfiromationDisc.enabled = false;
    }

    public void OnTouchUpdated(HandTrackingInputEventData eventData) 
    {
        if (!touchable) return;
        btnmesh.transform.position = eventData.InputData;

        if (btnmesh.transform.localPosition.z > pushConfiromationDisc.transform.localPosition.z)
            pushConfiromationDisc.Color = Color.cyan;
        else pushConfiromationDisc.Color = Color.white;

        if (btnmesh.transform.localPosition.z > pushConfiromationDisc.transform.localPosition.z+0.01f)
            selectEvent.Invoke();
    }

    private void SetSelected(bool selected)
    {
        if (selected)
        {
            loadingDisc.AngRadiansEnd = 6.24f;
            loadingDisc.Color = new Color(0.8f, 0.8f, 0.8f);
            btnBGMat.color = activeColor;
        }
        else
        {
            loadingDisc.AngRadiansEnd = startingAngle;
            loadingDisc.Color = Color.white;
            btnBGMat.color = baseColor;
        }
    }
}