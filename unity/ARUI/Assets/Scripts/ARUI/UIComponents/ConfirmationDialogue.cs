using Cogobyte.Demo.ProceduralIndicators;
using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.Events;

public class ConfirmationDialogue : MonoBehaviour
{
    private FlexibleTextContainer textContainer;

    private DwellButton okBtn;

    public UnityEvent selectEvent;

    private Shapes.Line timerLine;
    private float timeInSeconds = 6f;

    private void Awake()
    {
        textContainer = transform.GetChild(1).GetChild(0).gameObject.AddComponent<FlexibleTextContainer>();

        GameObject btn = transform.GetChild(0).gameObject;
        okBtn = btn.AddComponent<DwellButton>();
        okBtn.InitializeButton(EyeTarget.okButton, () => Confirmed(true));
        okBtn.SetDwellButtonType(DwellButtonType.Select);
        okBtn.gameObject.SetActive(false);

        selectEvent = new UnityEvent();

        timerLine = transform.GetComponentInChildren<Shapes.Line>();
        timerLine.enabled = false;
    }

    private void Update()
    {
        if (okBtn.gameObject.activeSelf==false && textContainer.TextRect.width > 0.001f)
            StartCoroutine(DecreaseTime());
    }

    private void Confirmed(bool isConfirmed)
    {
        if (isConfirmed)
            selectEvent.Invoke();
        else
            AngelARUI.Instance.LogDebugMessage("The user did not confirm the dialogue", true);

        StopCoroutine(DecreaseTime());
        Destroy(this.gameObject);
    }


    public void InitializeConfirmationNotification(string msg, UnityAction confirmedEvent)
    {
        if (msg == null || msg.Length == 0) return;

        textContainer.SetText(msg);
        selectEvent.AddListener(confirmedEvent);
    }

    private IEnumerator DecreaseTime()
    {
        okBtn.gameObject.SetActive(true);
        timerLine.enabled = true;

        okBtn.transform.position = textContainer.transform.position + new Vector3(textContainer.TextRect.width - okBtn.Width/2, 0, 0);

        timerLine.Start = new Vector3(0, textContainer.TextRect.height/2, 0);
        timerLine.End = new Vector3(textContainer.TextRect.width, textContainer.TextRect.height / 2, 0);
        Vector3 xEnd = timerLine.End;

        yield return new WaitForFixedUpdate();
        float distance = timerLine.End.x - timerLine.Start.x;
        float increments = distance / timeInSeconds;

        float timeElapsed= 0.00001f;
        float lerpDuration = timeInSeconds;

        while (timeElapsed < lerpDuration)
        {
            yield return new WaitForEndOfFrame();

            // Set our position as a fraction of the distance between the markers.
            timerLine.End = Vector3.Lerp(timerLine.Start, xEnd, 1- (timeElapsed / lerpDuration));
            timeElapsed += Time.deltaTime;
            /// timerLine.End = new Vector3(timerLine.End.x - increments, timerLine.End.y, timerLine.End.z);

        }

        Confirmed(false);
    }
}
