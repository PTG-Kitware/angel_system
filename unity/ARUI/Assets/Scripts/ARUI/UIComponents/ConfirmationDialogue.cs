using System.Collections;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UIElements;

/// <summary>
/// Dialogue that asks for user confirmation of a given action. Used for the Natural Language Interface.
/// The user has timeInSeconds seconds to decide if the given action should be executed. Confirmation can be done by 
/// looking at the button or touching it.
/// </summary>
public class ConfirmationDialogue : MonoBehaviour
{
    private bool init = false;
    private bool timerStarted = false;

    private FlexibleTextContainer textContainer;    
    private DwellButton okBtn;                      /// <Dialogue button
    public UnityEvent selectEvent;                  /// <Event that will be invoked if the user confirms the dialogue

    private Shapes.Line time;                       /// <Line that shows the user how much time is left to make a decision
    private float timeInSeconds = 6f;               /// <How much time the user has to decide (excluding the time the use is loking at the ok button

    private void Awake()
    {
        textContainer = transform.GetChild(1).GetChild(0).gameObject.AddComponent<FlexibleTextContainer>();

        GameObject btn = transform.GetChild(0).gameObject;
        okBtn = btn.AddComponent<DwellButton>();
        okBtn.InitializeButton(EyeTarget.okButton, () => Confirmed(true),true, DwellButtonType.Select);
        okBtn.gameObject.SetActive(false);

        selectEvent = new UnityEvent();

        time = transform.GetComponentInChildren<Shapes.Line>();
        time.enabled = false;
    }

    /// <summary>
    /// Start the timer if the dialogue is initialized and the timer is not running yet.
    /// </summary>
    private void Update()
    {
        if (init & !timerStarted && textContainer.TextRect.width > 0.001f)
            StartCoroutine(DecreaseTime());
    }

    /// <summary>
    /// Initialize the dialgoue components - text and confirmation event
    /// </summary>
    /// <param name="msg">message that is shown to the user.</param>
    /// <param name="confirmedEvent">confirmation event, invoked when the user is triggering the okay button</param>
    public void InitializeConfirmationNotification(string msg, UnityAction confirmedEvent)
    {
        if (msg == null || msg.Length == 0) return;

        textContainer.SetText(msg);
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
            selectEvent.Invoke();
        else
            AngelARUI.Instance.LogDebugMessage("The user did not confirm the dialogue", true);

        StopCoroutine(DecreaseTime());
        Destroy(this.gameObject);
    }

    /// <summary>
    /// Start and decrease the dialogue timer
    /// </summary>
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
