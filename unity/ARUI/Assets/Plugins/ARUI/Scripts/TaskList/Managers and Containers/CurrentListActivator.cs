using Microsoft.MixedReality.Toolkit.Utilities;
using Shapes;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.XR.OpenXR.Features.Interactions;

//Script attached to rectangle to activate respctive task list
public class CurrentListActivator : MonoBehaviour
{
    private int _index;
    public int Index {
        set
        {
            _index = value;
        }
     }

    private Rectangle _rect;

    private Line _rectProgress;
    private float _xStart = 0;
    private float _xEnd = 0;
    private float _currentRatio = 0;

    private TextMeshPro _textMeshProUGUI;
    public TextMeshPro Text
    {
        get => _textMeshProUGUI;
    }

    private Line progressPoints;

    public GameObject EyeGazeTarget;

    private void Awake()
    {
        _rect = GetComponentInChildren<Rectangle>();

        Line[] tmp = GetComponentsInChildren<Line>();
        _rectProgress = tmp[1];
        _xStart = _rectProgress.Start.x;
        _xEnd = _rectProgress.End.x;

        progressPoints = tmp[2];
        progressPoints.gameObject.SetActive(false) ;

        _rectProgress.End = new Vector3(_xStart, 0, 0);
        _textMeshProUGUI = GetComponentInChildren<TextMeshPro>();
        EyeGazeTarget = gameObject;
        EyeGazeManager.Instance.RegisterEyeTargetID(EyeGazeTarget);
    }

    // Update is called once per frame
    void Update()
    {
        //Once user looks at this object, set the task list visible
        if (EyeGazeManager.Instance != null && EyeGazeManager.Instance.CurrentHitID == EyeGazeTarget.GetInstanceID())
        {
            MultiTaskList.Instance.SetMenuActive(_index);
        } 

        if (progressPoints!=null && progressPoints.gameObject.activeSelf)
            progressPoints.DashOffset = Time.time % 250;
    }

    public void SetAsCurrent(bool isCurrent)
    {
        if (_rect == null) return;

        if (isCurrent)
        {
            _xStart = -0.025f;
            _textMeshProUGUI.fontSize = 0.20f;
            _textMeshProUGUI.fontStyle = FontStyles.Bold;
        } else
        {
            _xStart = -0.01f;
            _textMeshProUGUI.fontSize = 0.18f;
            _textMeshProUGUI.fontStyle = FontStyles.Normal;
        }

        progressPoints.gameObject.SetActive(isCurrent);

        _xEnd = _xStart * -1;
        _rect.Width = _xEnd * 2;

        _rectProgress.Start = new Vector3(_xStart, 0, 0);
        _rectProgress.End = new Vector3(_xStart + Mathf.Abs(_xEnd - _xStart) * _currentRatio, 0, 0);
    }

    public void UpdateProgres(float ratio)
    {
        if (_rect == null) return;
        _currentRatio = ratio;
        _rectProgress.End = new Vector3(_xStart + Mathf.Abs(_xEnd-_xStart) * _currentRatio, 0, 0);
    }
}
