using System;
using TMPro;
using UnityEngine;

public class OrbPie : MonoBehaviour
{
    private string _taskname;
    public string TaskName {
        get { return _taskname; }
        set { _taskname = value; }
    }

    private GameObject _pieSlice;
    private BoxCollider _sliceCollider;
    private Shapes.Disc _pie;
    private Shapes.Disc _pieProgress;
    private TextMeshProUGUI _progressText;

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

    private float _minRadius = 0.0175f;
    private float _minThick = 0.005f;
    private float _maxRadius = 0.024f;
    private float _maxThick = 0.014f;
    private float _maxRadiusActive = 0.027f;
    private float _maxThickActive = 0.02f;

    private float _rDeg = 0;
    private float _lDeg = 0;

    private GameObject _pieText;
    private TextMeshProUGUI _currentStepText;
    public string CurrentStepMessage
    {
        get => _currentStepText.text;
    }

    private Color _activeColorText = Color.white;

    private bool _isLookingAtPies = false;

    public void InitializeComponents(float rDeg, float lDeg)
    {
        _rDeg = rDeg;
        _lDeg = lDeg;

        //init pie slice and components
        _pieSlice = transform.GetChild(0).gameObject;
        _pie = _pieSlice.GetComponent<Shapes.Disc>();
        _sliceCollider = _pie.GetComponentInChildren<BoxCollider>();
        _pieProgress = _pie.transform.GetChild(0).GetComponent<Shapes.Disc>();
        _pieProgress.Radius = 0;
        _pieProgress.Thickness = 0;
        _progressText = _pieProgress.GetComponentInChildren<TextMeshProUGUI>();
        _progressText.gameObject.SetActive(false);

        _textContainer = transform.GetChild(1).GetChild(0).gameObject.AddComponent<FlexibleTextContainer>();
        _initialmessageXOffset = _textContainer.transform.localPosition.x;

        UpdateAnchor(MessageAnchor.right);

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

        SetPieActive(false, "");
    }

    public void ResetPie()
    {
        _taskname = "";
        _currentStepText.text = "";

        _pieProgress.Radius = 0;
        _pieProgress.Thickness = 0;
        _pie.Radius = _maxRadius;
        _pie.Thickness = _maxThick;

        _textContainer.TextColor = new Color(_activeColorText.r, _activeColorText.g, _activeColorText.b, 1);

        SetPieActive(false, "");
    }

    public void UpdateMessageVisibility(string currentActiveID)
    {
        if (_currentStepText.text.Length == 0) return;

        if (_pieSlice.activeSelf && EyeGazeManager.Instance.CurrentHit.Equals(EyeTarget.pieCollider) || EyeGazeManager.Instance.CurrentHit.Equals(EyeTarget.orbMessage))
        {
            _isLookingAtPies = true;
            if (!_pieText.activeSelf)
                _pieText.SetActive(true);

        } else
        {
            _isLookingAtPies = false;
            if (!_taskname.Equals(currentActiveID))
                _pieText.SetActive(false);
        }

        if (_pieSlice.activeSelf && !_pieText.activeSelf && _taskname.Equals(currentActiveID))
            _pieText.SetActive(true);
    }

    public void SetPieActive(bool active, string currentActiveID)
    {
        if (active && _currentStepText.text.Length == 0) return;

        _pieSlice.SetActive(active);

        if (active && (_taskname.Equals(currentActiveID) && _currentStepText.text.Length > 0) || _isLookingAtPies)
        {
            _pieText.SetActive(true);
        } else
            _pieText.SetActive(false);
    }

    public void SetTaskMessage(int stepIndex, int total, string message, string currentTaskID)
    {
        AngelARUI.Instance.DebugLogMessage("Set step message: '" + message + "' for task: " + TaskName, true);

        string newPotentialMessage = Utils.SplitTextIntoLines(TaskName + " (" + (stepIndex + 1) + "/" + total + ") : " +
            message, 150);

        //Play sound if task is finised and play task messag in case it was not played before
        if (message.Contains("Done!") && !_currentStepText.text.Contains("Done!"))
        {
            AudioManager.Instance.PlaySound(transform.position, SoundType.taskDone);
        } else if (currentTaskID.Equals(TaskName) && !newPotentialMessage.ToLower().Equals(_currentStepText.text.ToLower()))
        {
            AudioManager.Instance.PlayText(message);
        }

        _currentStepText.text = newPotentialMessage;
        //_prevText.text = "";
        //        _nextText.text = "";
        //        if (previousMessage.Length > 0)
        //            _prevText.text = "<b>DONE:</b> " + Utils.SplitTextIntoLines(previousMessage, ARUISettings.OrbMessageMaxCharCountPerLine);

        //        if (nextMessage.Length > 0)
        //            _nextText.text = "<b>Upcoming:</b> " + Utils.SplitTextIntoLines(nextMessage, ARUISettings.OrbNoteMaxCharCountPerLine);

    }

    #region Update Pie

    /// <summary>
    /// 
    /// </summary>
    public void UpdateCurrentTaskStatus(float ratio, string currentActiveID)
    {
        bool isCurrent = _taskname.Equals(currentActiveID);

        //Update pie length
        if (isCurrent)
        {
            _pie.Radius = _maxRadiusActive;
            _pie.Thickness = _maxThickActive;
        }
        else
        {
            _pie.Radius = _maxRadius;
            _pie.Thickness = _maxThick;
        }

        //Update progress pie length
        if (ratio == 0)
        {
            _pieProgress.Radius = 0;
            _pieProgress.Thickness = 0;
        }
        else
        {
            if (isCurrent)
            {
                _pieProgress.Radius = _minRadius + (ratio * (_maxRadiusActive - _minRadius));
                _pieProgress.Thickness = _minThick + (ratio * (_maxThickActive - _minThick));
            }
            else
            {
                _pieProgress.Radius = _minRadius + (ratio * (_maxRadius - _minRadius));
                _pieProgress.Thickness = _minThick + (ratio * (_maxThick - _minThick));
            }
        }
    }

    #endregion

    /// <summary>
    /// 
    /// </summary>
    /// <param name="anchor"></param>
    public void UpdateAnchor(MessageAnchor anchor)
    {
        _textContainer.UpdateAnchorInstant();

        float deg = _rDeg;
        float YRot = 0;
        if (anchor.Equals(MessageAnchor.left))
        {
            deg = _lDeg;
            YRot = 180;
        }
        _sliceCollider.transform.localRotation = Quaternion.Euler(new Vector3(_sliceCollider.transform.localRotation.x, YRot, _sliceCollider.transform.localRotation.z));

        _pie.AngRadiansEnd = deg * Mathf.Deg2Rad;
        _pie.AngRadiansStart = (deg -21) * Mathf.Deg2Rad;

        _pieProgress.AngRadiansEnd = deg * Mathf.Deg2Rad;
        _pieProgress.AngRadiansStart = (deg -5) * Mathf.Deg2Rad;
    }

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