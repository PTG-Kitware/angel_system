using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class Timer : MonoBehaviour
{
    public bool isRunning = false;
    public float timeLeft = 0f;

    private TextMeshPro labelTimer;
    private float targetTime = 5f;

    private AudioSource timerAudio;

    // Start is called before the first frame update
    void Start()
    {
        GameObject labelTextObj = new GameObject();
        labelTimer = labelTextObj.AddComponent<TextMeshPro>();
        labelTimer.rectTransform.sizeDelta = new Vector2(1f, 0.2f);
        labelTimer.fontSize = 0.3f;
        labelTimer.alignment = TextAlignmentOptions.Center;
        labelTimer.fontMaterial.SetFloat(ShaderUtilities.ID_FaceDilate, 0.20f);
        labelTimer.outlineColor = new Color(0.55f, 0.55f, 0.55f, 1.0f);
        labelTimer.outlineWidth = 0.20f;
        labelTimer.text = "test";
        labelTimer.gameObject.SetActive(false);

        timerAudio = labelTextObj.AddComponent<AudioSource>();
        timerAudio.clip = Resources.Load(StringResources.bellsound_path) as AudioClip;

        labelTextObj.transform.parent = transform;

        timeLeft = targetTime;
    }

    public void UpdateTime()
    {
        labelTimer.text = timeLeft.ToString("F") + " s";
        labelTimer.transform.rotation = Quaternion.LookRotation(labelTimer.transform.position - AngelARUI.Instance.mainCamera.transform.position, Vector3.up);
        Orb.Instance.SetTime(labelTimer.text);
    }

    public void StartTimer()
    {
        timeLeft = targetTime;
        isRunning = true;
        UpdateTime();

        labelTimer.gameObject.SetActive(true);
        Orb.Instance.SetTimerActive(true);
    }

    public void StopAndResetTimer(bool playSound)
    {
        isRunning = false;
        targetTime = 5f;
        timeLeft = targetTime;

        labelTimer.gameObject.SetActive(false);

        if (playSound) 
            timerAudio.Play();

        Orb.Instance.SetTimerActive(false);
    }
}
