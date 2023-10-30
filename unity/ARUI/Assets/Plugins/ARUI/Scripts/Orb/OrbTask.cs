using System;
using System.Collections;
using TMPro;
using UnityEngine;

public class OrbTask : MonoBehaviour
{
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
    private bool _textIsFadingIn = false;

    public void InitializeComponents(float rDeg, float lDeg)
    {
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

        SetPieActive(false, "");
    }

    public void ResetPie()
    {
        _taskname = "";
        _currentStepText.text = "";

        _pieProgressRect.Thickness = _thickness;
        _orbRect.Thickness = _thickness;
        _pieProgressRect.End = new Vector3(_xEnd,0,0);

        _textContainer.TextColor = new Color(_activeColorText.r, _activeColorText.g, _activeColorText.b, 1);

        SetPieActive(false, "");
    }

    private void Update()
    {
        _textContainer.IsLookingAtText = EyeGazeManager.Instance.CurrentHitObj != null &&
            EyeGazeManager.Instance.CurrentHitObj.GetInstanceID() == _textContainer.gameObject.GetInstanceID();
    }

    public void UpdateMessageVisibility(OrbTask currentactivePie)
    {
        if (currentactivePie == null || _currentStepText.text.Length == 0) return;

        if (!_orbRect.gameObject.activeSelf)
        {
            SetPieTextActive(false);
            return;
        }

        //only show the pie if it is the current observed task
        if (_orbRect.gameObject.activeSelf && !_pieText.activeSelf && _taskname.Equals(currentactivePie.TaskName))
        {
            SetPieTextActive(true);
            return;
        }

        if (EyeGazeManager.Instance.CurrentHit.Equals(EyeTarget.pieCollider) && !_textIsFadingIn)
        {
            StartCoroutine(FadeInMessage());
            return;
        }

        if (!_taskname.Equals(currentactivePie.TaskName) && EyeGazeManager.Instance.CurrentHitObj != null &&
            EyeGazeManager.Instance.CurrentHitObj.GetInstanceID() == currentactivePie.Text.gameObject.GetInstanceID() && !_textIsFadingOut
            && gameObject.activeSelf)
        {
            StartCoroutine(FadeOutAllMessages());
            return;

        } else if (!_taskname.Equals(currentactivePie.TaskName) && EyeGazeManager.Instance.CurrentHit.Equals(EyeTarget.orbMessage) &&
            EyeGazeManager.Instance.CurrentHitObj != null &&
            EyeGazeManager.Instance.CurrentHitObj.GetInstanceID() != currentactivePie.Text.gameObject.GetInstanceID() && !_textIsFadingIn)
        {
            SetPieTextActive(true);
            return;
        }
    }

    /// <summary>
    /// Fade out message from the moment the user does not look at the message anymore
    /// </summary>
    /// <returns></returns>
    private IEnumerator FadeOutAllMessages()
    {
        float fadeOutStep = 0.001f;
        _textIsFadingOut = true;

        yield return new WaitForSeconds(2.0f);

        float shade = ARUISettings.OrbMessageBGColor.r;
        float alpha = 1f;

        while (_textIsFadingOut && shade > 0)
        {
            alpha -= (fadeOutStep * 20);
            shade -= fadeOutStep;

            if (alpha < 0)
                alpha = 0;
            if (shade < 0)
                shade = 0;

            Text.BackgroundColor = new Color(shade, shade, shade, shade);
            SetTextAlpha(alpha);

            yield return new WaitForEndOfFrame();
        }

        _textIsFadingOut = false;
        _pieText.SetActive(!(shade <= 0));
    }

    /// <summary>
    /// 
    /// </summary>
    /// <returns></returns>
    private IEnumerator FadeInMessage()
    {
        _textIsFadingIn = true;
        yield return new WaitForSeconds(0.8f);

        if (EyeGazeManager.Instance.CurrentHit.Equals(EyeTarget.pieCollider));
            SetPieTextActive(true);

        _textIsFadingIn = false;
    }


    private void SetPieTextActive(bool isActive)
    {
        _pieText.SetActive(isActive);
        
        if (isActive)
        {
            StopCoroutine(FadeOutAllMessages());
            _textIsFadingOut = false;
            Text.BackgroundColor = ARUISettings.OrbMessageBGColor;
            SetTextAlpha(1);
        }
    }

    public void SetPieActive(bool active, string currentActiveID)
    {
        if (active && _currentStepText.text.Length == 0) return;

        _orbRect.gameObject.SetActive(active);

        if (!active)
        {
            _pieText.SetActive(false);
        }
    }

    public void SetTaskMessage(int stepIndex, int total, string message, string currentTaskID)
    {
        AngelARUI.Instance.DebugLogMessage("Set step message: '" + message + "' for task: " + TaskName, true);

        string newPotentialMessage = Utils.SplitTextIntoLines(TaskName + " (" + (stepIndex + 1) + "/" + total + ") : " +
            message, 110);

        _currentStepText.text = newPotentialMessage;
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
    /// 
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