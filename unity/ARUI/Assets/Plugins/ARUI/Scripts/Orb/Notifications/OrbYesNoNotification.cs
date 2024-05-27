using Microsoft.MixedReality.Toolkit.Input;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

/// <summary>
/// TODO
/// </summary>
public class OrbYesNoNotification : OrbNotificationTemplate 
{
    private DwellButton _yesBtn;
    private DwellButton _noBtn;                     

    private UnityEvent _yesSelectEvent;          /// <Event that will be invoked if the user selects 'yes'
    private UnityEvent _noSelectEvent;           /// <Event that will be invoked if the user selects 'no'

    private void Awake()
    {
        _textContainer = transform.GetChild(1).GetChild(0).gameObject.AddComponent<FlexibleTextContainer>();

        type = NotificationType.YesNo;

        //init yes
        var allButtons = transform.GetChild(0);

        _yesSelectEvent = new UnityEvent();
        DwellButton dwell = allButtons.GetChild(0).gameObject.AddComponent<DwellButton>();
        dwell.InitializeButton(dwell.gameObject, () => Confirmed(true, _yesSelectEvent), null, true, DwellButtonType.Select, true);
        _yesBtn = dwell;

        //init no
        _noSelectEvent = new UnityEvent();
        dwell = allButtons.GetChild(1).gameObject.AddComponent<DwellButton>();
        dwell.InitializeButton(dwell.gameObject, () => Confirmed(true, _noSelectEvent), null, true, DwellButtonType.Select, true);
        _noBtn = dwell;


        _time = _textContainer.transform.GetComponentInChildren<Shapes.Line>();
        _time.enabled = false;

        _timeOutEvent = new UnityEvent();
        _selfDestruct = new UnityEvent();

        transform.SetLayerAllChildren(StringResources.LayerToInt(StringResources.UI_layer));
    }


    /// <summary>
    /// Initialize the dialgoue components - text and confirmation event
    /// </summary>
    /// <param name="selectionMsg"></param>
    /// <param name="actionOnYes"></param>
    /// <param name="actionOnNo"></param>
    /// <param name="actionOnTimeOut"></param>
    /// <param name="selfDestruct"></param>
    /// <param name="timeout"></param>
    public void InitNotification(string selectionMsg, UnityAction actionOnYes, UnityAction actionOnNo, UnityAction actionOnTimeOut, UnityAction selfDestruct, float timeout)
    {
        _timeOutInSeconds = timeout;
        string dialogText = selectionMsg + "\n<b>";
        if (actionOnYes != null)
            _yesSelectEvent.AddListener(actionOnYes);

        if (actionOnNo != null)
            _noSelectEvent.AddListener(actionOnNo);

        _textContainer.Text = dialogText;
        _textContainer.AddShortLineToText("</b><i><size=0.006><color=#d3d3d3>Confirm by saying 'Select Yes' or 'Select No'</color></size></i>");

        _selfDestruct.AddListener(selfDestruct);
        if (actionOnTimeOut != null)
            _timeOutEvent.AddListener(actionOnTimeOut);

        _init = true;
    }

    /// <summary>
    /// Called if the user either actively confirmed, or passively did not confirm the dialogue.
    /// if isConfirmed is true, the event assigned to the dialogue during initialization is triggered
    /// </summary>
    /// <param name="isConfirmed">true if confirmed by user, else false</param>
    private void Confirmed(bool isConfirmed, UnityEvent confirmationEvent)
    {
        if (isConfirmed)
        {
            AngelARUI.Instance.DebugLogMessage("The user selected.", true);

            if (confirmationEvent != null)
            {
                confirmationEvent.Invoke();
            }

            AudioManager.Instance.PlaySound(transform.position, SoundType.actionConfirmation);
        }
        else
        {
            AngelARUI.Instance.DebugLogMessage("The user did not confirm the mulitple choice", true);

            if (_timeOutEvent != null)
                _timeOutEvent.Invoke();
        }

        StopCoroutine(DecreaseTime());
        _selfDestruct.Invoke();
    }

    /// <summary>
    /// /TODO
    /// </summary>
    /// <param name="input"></param>
    public void ConfirmedViaSpeech(AcceptedSpeechInput input)
    {
        if (input.Equals(AcceptedSpeechInput.SelectYes))
        {
            Confirmed(true, _yesSelectEvent);

        } else if (input.Equals(AcceptedSpeechInput.SelectNo))
        {
            Confirmed(true, _noSelectEvent);
        }
    }

    #region Timeout

    /// <summary>
    /// Start the timer if the dialogue is initialized and the timer is not running yet.
    /// </summary>
    private void Update()
    {
        if (_init & !_timerStarted && _textContainer.TextRect.width > 0.001f)
            StartCoroutine(DecreaseTime());
    }

    private IEnumerator DecreaseTime()
    {
        AudioManager.Instance.PlaySound(transform.position, SoundType.select);

        _timerStarted = true;
        _time.enabled = true;

        _time.Start = new Vector3(0, 0, 0);
        _time.End = new Vector3(_textContainer.TextRect.width, 0, 0);
        Vector3 xEnd = _time.End;

        yield return new WaitForFixedUpdate();
        float timeElapsed= 0.00001f;
        float lerpDuration = _timeOutInSeconds;

        while (timeElapsed < lerpDuration)
        {
            yield return new WaitForEndOfFrame();

            if (!(_yesBtn.IsInteractingWithBtn || _noBtn.IsInteractingWithBtn || _textContainer.IsLookingAtText))
            {
                _time.End = Vector3.Lerp(_time.Start, xEnd, 1 - (timeElapsed / lerpDuration));
                timeElapsed += Time.deltaTime;
            }
        }

        Confirmed(false, null);
    }

    #endregion
}
