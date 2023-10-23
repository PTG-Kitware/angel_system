using System;
using System.Collections;
using UnityEngine;

public enum OrbStates
{
    Idle = 0,
    Loading = 1,
}

/// <summary>
/// Represents the visual representation of the orb (the disc)
/// </summary>
public class OrbFace : MonoBehaviour
{
    ///** Orb face parts and states
    private OrbStates _currentFaceState = OrbStates.Idle;
    private Shapes.Disc _face;
    private Shapes.Disc _eyes;
    private Shapes.Disc _mouth;
    private Shapes.Disc _orbHalo;

    private Shapes.Disc _shut;

    ///** Colors of orb states
    private Color _faceColorInnerStart = new Color(1, 1, 1, 1f);
    private Color _faceColorOuterStart = new Color(1, 1, 1, 0f);
    private Color _faceColorInnerEnd = new Color(1, 1, 1, 1f);
    private Color _faceColorOuterEnd = new Color(1, 1, 1, 0f);

    private GameObject _warningIcon;
    private GameObject _noteIcon;
    public bool MessageNotificationEnabled
    {
        set => SetNotificationPulse(value);
    }

    public void UpdateNotification(bool warning, bool note)
    {
        SetNotificationPulse(warning || note);

        if (warning && note)
        {
            _face.ColorInnerStart = Color.yellow;
            _face.ColorInnerEnd = Color.red;
        }

        if (warning && !note)
        {
            _face.ColorInnerStart = Color.red;
            _face.ColorInnerEnd = Color.red;
        }

        if (note && !warning)
        {
            _face.ColorInnerStart = Color.yellow;
            _face.ColorInnerEnd = Color.yellow;
        }

        if (!warning && !note)
        {
            _face.ColorInnerStart = _faceColorInnerStart;
            _face.ColorInnerEnd = _faceColorInnerEnd;
        }

        _noteIcon.SetActive(note);
        _warningIcon.SetActive(warning);
    }

    private float _initialMouthScale;
    public float MouthScale { 
        get => _mouth.Radius;
        set
        {
            if (value<=0)
            {
                _mouth.Radius = _initialMouthScale;
            } else
            {
                _mouth.Radius = Mathf.Clamp(_initialMouthScale - value, 0.5f, _initialMouthScale);
            }
            
        }
    }

    private bool _isPulsing = false;

    private bool _userIsLooking = false;
    public bool UserIsLooking
    {
        get => _userIsLooking;
        set { _userIsLooking = value; }
    }
    
    private bool _userIsGrabbing = false;
    public bool UserIsGrabbing
    {
        get => _userIsGrabbing;
        set
        {
            _userIsGrabbing = value;
        }
    }

    private void Start()
    {
        Shapes.Disc[] allDiscs = transform.GetChild(0).GetComponentsInChildren<Shapes.Disc>();
        _face = allDiscs[0];
        _mouth = allDiscs[1];
        _eyes = allDiscs[2];
        _eyes.gameObject.SetActive(false);

        _faceColorOuterStart = _face.ColorOuterStart;
        _faceColorInnerStart = _face.ColorInnerStart;
        _faceColorOuterEnd = _face.ColorOuterEnd;
        _faceColorInnerEnd = _face.ColorInnerEnd;

        _initialMouthScale = _mouth.Radius;

        _orbHalo = allDiscs[3];
        _orbHalo.gameObject.SetActive(false);

        _noteIcon = allDiscs[4].gameObject;
        _noteIcon.SetActive(false);
        _warningIcon = allDiscs[5].gameObject;
        _warningIcon.SetActive(false);

        _shut = transform.GetChild(1).GetComponentInChildren<Shapes.Disc>();
        _shut.gameObject.SetActive(false);
    }

    private void Update()
    {
        if (_userIsLooking || _userIsGrabbing && !_eyes.gameObject.activeSelf)
            _eyes.gameObject.SetActive(true);

        else if (!_userIsLooking && !_userIsGrabbing && _eyes.gameObject.activeSelf)
            _eyes.gameObject.SetActive(false);

        if (Orb.Instance.OrbBehavior.Equals(OrbMovementBehavior.Fixed))
            _mouth.Type = Shapes.DiscType.Disc;
        else
            _mouth.Type = Shapes.DiscType.Ring;
    }

    private void SetNotificationPulse(bool pulsing)
    {
       if (pulsing && !_isPulsing)
        {
            StartCoroutine("Pulse");

        } else if (!pulsing)
        {
            _isPulsing = false;
        }
    }

    private IEnumerator Pulse()
    {
        _isPulsing = true;

        float speed = 3f * Time.deltaTime;
        float pulse = 0;

        _orbHalo.gameObject.SetActive(true);

        while (_isPulsing)
        {
            pulse += speed;

            _face.ColorOuterStart = new Color(_face.ColorOuterStart.r, _face.ColorOuterStart.g, _face.ColorOuterStart.b, Mathf.Abs(Mathf.Sin(pulse)));
            _face.ColorOuterEnd = _face.ColorOuterStart;

            _orbHalo.Thickness = (pulse/10) * 2;

            yield return new WaitForEndOfFrame();   
        }

        _orbHalo.gameObject.SetActive(false);

        _face.ColorOuterStart = new Color(_face.ColorOuterStart.r, _face.ColorOuterStart.g, _face.ColorOuterStart.b, 0);
        _face.ColorOuterEnd = _face.ColorOuterStart;
    }

    public void SetOrbGuidance(bool isGuidanceActive)
    {
        transform.GetChild(0).gameObject.SetActive(isGuidanceActive);
        transform.GetChild(1).gameObject.SetActive(!isGuidanceActive);
    }

    public void SetOrbState(OrbStates newState)
    {
        if (newState.Equals(OrbStates.Loading) &&
            _currentFaceState!= OrbStates.Loading)
        {
            _face.Type = Shapes.DiscType.Arc;
            StartCoroutine(Rotating());

        } else if (newState.Equals(OrbStates.Idle) &&
            _currentFaceState != OrbStates.Idle)
        {
            _face.Type = Shapes.DiscType.Ring;
            StopCoroutine(Rotating());
        }

        _currentFaceState = newState;
    }

    /// <summary>
    /// For now, rotate the face while in loading state. 
    /// </summary>
    /// <returns></returns>
    private IEnumerator Rotating()
    {
        while (_currentFaceState == OrbStates.Loading)
        {
            _face.transform.Rotate(new Vector3(0,0,20f),Space.Self);
            yield return new WaitForEndOfFrame();
        }
    }
}