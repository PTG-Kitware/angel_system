using System.Collections;
using UnityEngine;
using UnityEngine.Events;
using RosMessageTypes.Angel;
using Microsoft.MixedReality.Toolkit.Utilities.Solvers;
using System;
using TMPro;

/// <summary>
/// Custom one argument version of a UnityEvent to allow passing of
/// InterpretedAudioUserIntentMsg arguments.
/// See: https://docs.unity3d.com/ScriptReference/Events.UnityEvent_1.html
/// </summary>
[System.Serializable]
public class InterpretedAudioUserIntentEvent : UnityEvent<InterpretedAudioUserIntentMsg> {}

/// <summary>
/// Dialogue that asks for user confirmation of a given action. Used for the Natural Language Interface.
/// The user has timeInSeconds seconds to decide if the given action should be executed. Confirmation can be done by 
/// looking at the button or touching it.
/// </summary>
public class ConfirmationDialogue : MonoBehaviour
{
    private bool init = false;                              /// <true if dialogue was initialized (e.g. message, event)
    private bool timerStarted = false;                      /// <true if timer started already

    private FlexibleTextContainer textContainer;    
    private DwellButton okBtn;                              /// <Dialogue button
    private Orbital movingBehavior;
    private bool delayedMoving = false;

    private InterpretedAudioUserIntentEvent selectEvent;     /// <Event that will be invoked if the user confirms the dialogue
    private InterpretedAudioUserIntentMsg userIntent;

    private Shapes.Line time;                               /// <Line that shows the user how much time is left to make a decision
    private float timeInSeconds = 6f;                       /// <How much time the user has to decide (excluding the time the use is loking at the ok button


    private void Awake()
    {
        textContainer = transform.GetChild(1).GetChild(0).gameObject.AddComponent<FlexibleTextContainer>();
        textContainer.gameObject.AddComponent<VMNonControllable>();

        GameObject btn = transform.GetChild(0).gameObject;
        okBtn = btn.AddComponent<DwellButton>();
        okBtn.InitializeButton(EyeTarget.okButton, () => Confirmed(true),true, DwellButtonType.Select);
        okBtn.gameObject.SetActive(false);

        time = transform.GetComponentInChildren<Shapes.Line>();
        time.enabled = false;

        transform.position = AngelARUI.Instance.ARCamera.ViewportToWorldPoint(new Vector3(0.5f, 0.7f, 1f), Camera.MonoOrStereoscopicEye.Left);
        movingBehavior = gameObject.GetComponent<Orbital>();
        movingBehavior.enabled = true;

        selectEvent = new InterpretedAudioUserIntentEvent();
    }

    /// <summary>
    /// Start the timer if the dialogue is initialized and the timer is not running yet.
    /// </summary>
    private void Update()
    {
        if (init & !timerStarted && textContainer.TextRect.width > 0.001f)
            StartCoroutine(DecreaseTime());

        if (okBtn.IsInteractingWithBtn && movingBehavior.enabled)
            movingBehavior.enabled = false;
        else if (!okBtn.IsInteractingWithBtn && !movingBehavior.enabled && !delayedMoving)
            StartCoroutine(DelayedStartMoving());
            
    }

    private IEnumerator DelayedStartMoving()
    {
        delayedMoving = true;

        yield return new WaitForSeconds(1f);

        if (!okBtn.IsInteractingWithBtn)
            movingBehavior.enabled = true;

        delayedMoving = false;
    }

    /// <summary>
    /// Initialize the dialgoue components - text and confirmation event
    /// </summary>
    /// <param name="intentMsg">Contains message that is shown to the user.</param>
    /// <param name="confirmedEvent">confirmation event, invoked when the user is triggering the okay button</param>
    public void InitializeConfirmationNotification(InterpretedAudioUserIntentMsg intentMsg, UnityAction<InterpretedAudioUserIntentMsg> confirmedEvent)
    {
        if (intentMsg == null || intentMsg.user_intent.Length == 0) return;

        userIntent = intentMsg;
        textContainer.SetText(intentMsg.user_intent);
        selectEvent.AddListener(confirmedEvent);
        init = true;
    }

    /// <summary>
    /// Called if the user either actively confirmed, or passively did not confirm the dialogue.
    /// if isConfirmed is true, the event assigned to the dialogue during initialization is triggered
    /// </summary>
    /// <param name="isConfirmed">true if confirmed by user, else false</param>
    private void Confirmed(bool isConfirmed)
    {
        if (isConfirmed)
            selectEvent.Invoke(userIntent);
        else
            AngelARUI.Instance.LogDebugMessage("The user did not confirm the dialogue", true);

        StopCoroutine(DecreaseTime());
        Destroy(this.gameObject);
    }

    private IEnumerator DecreaseTime()
    {
        timerStarted = true;

        okBtn.gameObject.SetActive(true);
        time.enabled = true;

        okBtn.transform.position = textContainer.transform.position + new Vector3(textContainer.TextRect.width - okBtn.Width/2, 0, 0);

        time.Start = new Vector3(0, textContainer.TextRect.height/2, 0);
        time.End = new Vector3(textContainer.TextRect.width, textContainer.TextRect.height / 2, 0);
        Vector3 xEnd = time.End;

        yield return new WaitForFixedUpdate();
        float timeElapsed= 0.00001f;
        float lerpDuration = timeInSeconds;
        while (timeElapsed < lerpDuration)
        {
            yield return new WaitForEndOfFrame();

            if (!okBtn.IsInteractingWithBtn)
            {
                time.End = Vector3.Lerp(time.Start, xEnd, 1 - (timeElapsed / lerpDuration));
                timeElapsed += Time.deltaTime;
            }


        }

        Confirmed(false);
    }
}
