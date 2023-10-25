using System.Collections;
using UnityEngine;
using UnityEngine.Events;
using Microsoft.MixedReality.Toolkit.Utilities.Solvers;
using System;
using TMPro;

/// <summary>
/// Dialogue that asks for user confirmation of a given action. Used for the Natural Language Interface.
/// The user has timeInSeconds seconds to decide if the given action should be executed. Confirmation can be done by
/// looking at the button or touching it.
/// </summary>
public class ConfirmationDialogue : MonoBehaviour
{
    private bool _init = false;                      /// <true if dialogue was initialized (e.g. message, event)
    private bool _timerStarted = false;              /// <true if timer started already

    private FlexibleTextContainer _textContainer;
    private DwellButton _okBtn;                      /// <Dialogue button
    private Orbital _movingBehavior;
    private bool _delayedMoving = false;

    private UnityEvent _selectEvent;                 /// <Event that will be invoked if the user confirms the dialogue
    private string _userIntent;

    private Shapes.Line _time;                       /// <Line that shows the user how much time is left to make a decision
    
    private void Awake()
    {
        _textContainer = transform.GetChild(1).GetChild(0).gameObject.AddComponent<FlexibleTextContainer>();
        //_textContainer.AddVMNC();

        GameObject btn = transform.GetChild(0).gameObject;
        _okBtn = btn.AddComponent<DwellButton>();
        _okBtn.InitializeButton(EyeTarget.okButton, () => Confirmed(true), null, true, DwellButtonType.Select);
        _okBtn.gameObject.SetActive(false);

        _time = transform.GetComponentInChildren<Shapes.Line>();
        _time.enabled = false;

        transform.position = AngelARUI.Instance.ARCamera.ViewportToWorldPoint(new Vector3(0.5f, 0.7f, 1f), Camera.MonoOrStereoscopicEye.Left);
        _movingBehavior = gameObject.GetComponent<Orbital>();
        _movingBehavior.enabled = true;

        _selectEvent = new UnityEvent();

        transform.SetLayerAllChildren(StringResources.LayerToInt(StringResources.UI_layer));
    }


    /// <summary>
    /// Start the timer if the dialogue is initialized and the timer is not running yet.
    /// </summary>
    private void Update()
    {
        if (_init & !_timerStarted && _textContainer.TextRect.width > 0.001f)
            StartCoroutine(DecreaseTime());

        if (_okBtn.IsInteractingWithBtn && _movingBehavior.enabled)
            _movingBehavior.enabled = false;
        else if (!_okBtn.IsInteractingWithBtn && !_movingBehavior.enabled && !_delayedMoving)
            StartCoroutine(DelayedStartMoving());

    }

    private IEnumerator DelayedStartMoving()
    {
        _delayedMoving = true;

        yield return new WaitForSeconds(0.5f);

        if (!_okBtn.IsInteractingWithBtn)
            _movingBehavior.enabled = true;

        _delayedMoving = false;
    }

    /// <summary>
    /// Initialize the dialgoue components - text and confirmation event
    /// </summary>
    /// <param name="intentMsg">Contains message that is shown to the user.</param>
    /// <param name="confirmedEvent">confirmation event, invoked when the user is triggering the okay button</param>
    public void InitializeConfirmationNotification(string intentMsg, UnityAction confirmedEvent)
    {
        if (intentMsg == null || intentMsg.Length == 0) return;

        _userIntent = intentMsg;
        _textContainer.Text = intentMsg;
        _selectEvent.AddListener(confirmedEvent);
        _init = true;
    }

    /// <summary>
    /// Called if the user either actively confirmed, or passively did not confirm the dialogue.
    /// if isConfirmed is true, the event assigned to the dialogue during initialization is triggered
    /// </summary>
    /// <param name="isConfirmed">true if confirmed by user, else false</param>
    private void Confirmed(bool isConfirmed)
    {
        if (isConfirmed)
            _selectEvent.Invoke();
        else
            AngelARUI.Instance.DebugLogMessage("The user did not confirm the dialogue", true);

        StopCoroutine(DecreaseTime());
        Destroy(this.gameObject);
    }

    private IEnumerator DecreaseTime()
    {
        AudioManager.Instance.PlaySound(transform.position, SoundType.select);

        _timerStarted = true;

        _okBtn.gameObject.SetActive(true);
        _time.enabled = true;

        _okBtn.transform.localPosition = _textContainer.transform.localPosition + new Vector3(_textContainer.TextRect.width + _okBtn.Width/2, 0, 0);

        _time.Start = new Vector3(0, _textContainer.TextRect.height/2, 0);
        _time.End = new Vector3(_textContainer.TextRect.width, _textContainer.TextRect.height / 2, 0);
        Vector3 xEnd = _time.End;

        yield return new WaitForFixedUpdate();
        float timeElapsed= 0.00001f;
        float lerpDuration = ARUISettings.DialogueTimeInSeconds;
        while (timeElapsed < lerpDuration)
        {
            yield return new WaitForEndOfFrame();

            if (!_okBtn.IsInteractingWithBtn)
            {
                _time.End = Vector3.Lerp(_time.Start, xEnd, 1 - (timeElapsed / lerpDuration));
                timeElapsed += Time.deltaTime;
            }


        }

        Confirmed(false);
    }
}
