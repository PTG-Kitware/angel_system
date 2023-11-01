using System;
using System.Collections;
using TMPro;
using UnityEngine;

public enum TaskType
{
    primary = 1, 
    secondary = 2, 
}

public class OrbTask : MonoBehaviour
{
    private TaskType _taskType = TaskType.secondary;
    public TaskType TaskType { get { return _taskType; } }

    private string _taskname;
    public string TaskName {
        get { return _taskname; }
        set { _taskname = value; }
    }

    private Shapes.Line _orbRect;
    private Shapes.Line _pieProgressRect;
   
    private float _thicknessActive;
    private float _thickness;
    private float _xStart = 0;
    private float _xEnd = 0;

    private FlexibleTextContainer _textContainer;
    public FlexibleTextContainer Text
    {
        get => _textContainer;
    }

    private float _initialmessageXOffset = 0;
    public float InitialXOffset
    {
        get => _initialmessageXOffset;
    }

    private GameObject _pieText;
    private TextMeshProUGUI _currentStepText;
    public string CurrentStepMessage
    {
        get => _currentStepText.text;
    }

    private Color _activeColorText = Color.white;

    private bool _textIsFadingOut = false;

    public void InitializeComponents(TaskType currenType)
    {
        _taskType = currenType;

        //init pie slice and components
        GameObject tmp = transform.GetChild(0).gameObject;
        _orbRect = tmp.GetComponent<Shapes.Line>();

        _pieProgressRect = _orbRect.transform.GetChild(0).GetComponent<Shapes.Line>();
        _pieProgressRect.End = _pieProgressRect.Start;
        _thicknessActive = _pieProgressRect.Thickness;
        _thickness = _thicknessActive / 2;

        _pieProgressRect.Thickness = _thickness;
        _orbRect.Thickness = _thickness;
        _xStart = _orbRect.Start.x;
        _xEnd = _orbRect.End.x;

        _textContainer = transform.GetChild(1).GetChild(0).gameObject.AddComponent<FlexibleTextContainer>();
        _initialmessageXOffset = _textContainer.transform.localPosition.x;

        UpdateAnchor();

        //init pie text
        _pieText = transform.GetChild(1).gameObject;
        var allChildren = _pieText.transform.GetAllDescendents();
        foreach (var child in allChildren)
        {
            if (child.GetComponent<TextMeshProUGUI>() != null)
            {
                _currentStepText = child.GetComponent<TextMeshProUGUI>();
                _currentStepText.text = "";
                break;
            }
        }

        _taskname = gameObject.name;

        SetPieActive(false);
    }

    public void ResetPie()
    {
        _taskname = "";
        _currentStepText.text = "";

        _pieProgressRect.Thickness = _thickness;
        _orbRect.Thickness = _thickness;
        _pieProgressRect.End = new Vector3(_xEnd,0,0);

        _textContainer.TextColor = new Color(_activeColorText.r, _activeColorText.g, _activeColorText.b, 1);
    }

    private void Update()
    {
        _textContainer.IsLookingAtText = EyeGazeManager.Instance.CurrentHitObj != null &&
            EyeGazeManager.Instance.CurrentHitObj.GetInstanceID() == _textContainer.gameObject.GetInstanceID();
    }

    public void SetPieActive(bool active)
    {
        if (active && _currentStepText.text.Length == 0) return;

        _orbRect.gameObject.SetActive(active);
        _pieText.SetActive(active);
    }


    /// <summary>
    /// Set the text message of this orb task container.
    /// </summary>
    /// <param name="stepIndex"></param>
    /// <param name="total"></param>
    /// <param name="message"></param>
    public void SetTaskMessage(int stepIndex, int total, string message)
    {
        int maxChar = 110;
        if (_taskType.Equals(TaskType.primary))
            maxChar = 80;

        string newPotentialMessage = Utils.SplitTextIntoLines(TaskName + " (" + (stepIndex + 1) + "/" + total + ") : " +
            message, maxChar);

        _currentStepText.text = newPotentialMessage;
    }

    #region Update Pie

    /// <summary>
    /// Update the green task progress bar 
    /// </summary>
    public void UpdateCurrentTaskStatus(float ratio)
    {
        //Update pie length
        if (_taskType.Equals(TaskType.primary))
        {
            _pieProgressRect.Thickness = _thicknessActive;
            _orbRect.Thickness = _thicknessActive;
            transform.SetSiblingIndex(0);
            _textContainer.TextSize = 0.008f;
        }
        else
        {
            _pieProgressRect.Thickness = _thickness;
            _orbRect.Thickness = _thickness;
            _textContainer.TextSize = 0.0065f;
        }

        //Update progress length
        if (ratio == 0)
            _pieProgressRect.End = new Vector3(_xStart, 0, 0);
        else
            _pieProgressRect.End = new Vector3(_xStart + (Mathf.Abs(_xEnd - _xStart) * ratio), 0, 0);
    }

    #endregion

    /// <summary>
    /// Update anchor of textbox
    /// </summary>
    /// <param name="anchor"></param>
    public void UpdateAnchor() => _textContainer.UpdateAnchorInstant();

    /// <summary>
    /// Update the color of the text based on visibility
    /// </summary>
    /// <param name="alpha"></param>
    public void SetTextAlpha(float alpha)
    {
        if (alpha == 0)
            _textContainer.TextColor = new Color(0, 0, 0, 0);
        else
            _textContainer.TextColor = new Color(_activeColorText.r, _activeColorText.g, _activeColorText.b, alpha);
    }
}