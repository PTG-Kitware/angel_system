using Microsoft.MixedReality.Toolkit.Input;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

/// <summary>
/// Dialogue that asks for user confirmation of a given action. Used for the Natural Language Interface.
/// The user has timeInSeconds seconds to decide if the given action should be executed. Confirmation can be done by
/// looking at the button or touching it.
/// </summary>
public class OrbConfirmationNotification : OrbNotificationTemplate
{
    private DwellButton _okBtn;                      /// <Dialogue button

    private List<UnityEvent> _selectEvents;          /// <Events that will be invoked if the user confirms the dialogue

    private void Awake()
    {
        _textContainer = transform.GetChild(1).GetChild(0).gameObject.AddComponent<FlexibleTextContainer>();

        type = NotificationType.OkayChoice;

        GameObject btn = transform.GetChild(0).gameObject;
        _okBtn = btn.AddComponent<DwellButton>();
        _okBtn.InitializeButton(btn, () => Confirmed(true), null, true, DwellButtonType.Select, true);
        _okBtn.gameObject.SetActive(false);

        _time = transform.GetComponentInChildren<Shapes.Line>();
        _time.enabled = false;

        _selectEvents = new List<UnityEvent>();
        _timeOutEvent = new UnityEvent();
        _selfDestruct = new UnityEvent();

        transform.SetLayerAllChildren(StringResources.LayerToInt(StringResources.UI_layer));
    }

    /// <summary>
    /// Initialize the dialgoue components - text and confirmation event
    /// </summary>
    /// <param name="intentMsg">Contains message that is shown to the user.</param>
    /// <param name="confirmedEvent">confirmation event, invoked when the user is triggering the okay button</param>
    public void InitNotification(string intentMsg, List<UnityAction> confirmedEvent, UnityAction actionOnTimeOut, UnityAction selfDestruct, float timeout)
    {
        if (intentMsg == null || intentMsg.Length == 0) return;

        _timeOutInSeconds = timeout;
        _textContainer.Text = intentMsg;
        _textContainer.AddShortLineToText("<i><size=0.006><color=#d3d3d3>Confirm by saying 'Select Okay'</color></size></i>");

        int i = 0;
        foreach (UnityAction action in confirmedEvent)
        {
            _selectEvents.Add(new UnityEvent());
            _selectEvents[i].AddListener(action);
            i++;
        }

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
    public void Confirmed(bool isConfirmed)
    {
        if (isConfirmed)
        {
            foreach (UnityEvent action in _selectEvents)
            {
                action.Invoke();
            }
            AngelARUI.Instance.DebugLogMessage("The user did confirm the dialogue - "+gameObject.GetInstanceID(), true);
            AudioManager.Instance.PlaySound(transform.position, SoundType.actionConfirmation);
        }
        else
        {
            if (_timeOutEvent != null)
                _timeOutEvent.Invoke();

            AngelARUI.Instance.DebugLogMessage("The user did not confirm the dialogue - "+gameObject.GetInstanceID(), true);
        }

        StopCoroutine(DecreaseTime());
        _selfDestruct.Invoke();
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

        _okBtn.gameObject.SetActive(true);
        _time.enabled = true;

        //_okBtn.transform.localPosition = _textContainer.transform.localPosition + new Vector3(_textContainer.TextRect.width + _okBtn.Width/2, 0, 0);

        _time.Start = new Vector3(0, _textContainer.TextRect.height/2, 0);
        _time.End = new Vector3(_textContainer.TextRect.width, _textContainer.TextRect.height / 2, 0);
        Vector3 xEnd = _time.End;

        yield return new WaitForFixedUpdate();
        float timeElapsed= 0.00001f;
        float lerpDuration = _timeOutInSeconds;

        while (timeElapsed < lerpDuration)
        {
            yield return new WaitForEndOfFrame();

            if (!_okBtn.IsInteractingWithBtn && !_textContainer.IsLookingAtText)
            {
                _time.End = Vector3.Lerp(_time.Start, xEnd, 1 - (timeElapsed / lerpDuration));
                timeElapsed += Time.deltaTime;
            }
        }

        Confirmed(false);
    }

    #endregion
}
