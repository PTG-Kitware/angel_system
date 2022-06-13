using DilmerGames.Core.Singletons;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities.Solvers;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Orb : Singleton<Orb>
{
    private GameObject face;
    private SpriteRenderer faceSprite;
    
    private GameObject messageContainer;
    private TMPro.TextMeshPro message;

    private GameObject timerContainer;
    private TMPro.TextMeshPro timer;

    private GameObject recordingIcon;

    private EyeTrackingTarget eyeEvents;

    private bool isLooking = false;

    private Dictionary<string, Sprite> allFaces;

    private RadialView followSolver;
    private SpeechInputHandler speechInput;
    private bool isProcessingSpeechInput = false;

    private GameObject taskListbutton;

    // Start is called before the first frame update
    void Start()
    {
        gameObject.name = "Orb";

        faceSprite = GetComponentInChildren<SpriteRenderer>(true);
        face = faceSprite.transform.parent.parent.gameObject;

        allFaces = new Dictionary<string, Sprite>();

        Texture2D texture = Resources.Load(StringResources.idle_orb_path) as Texture2D;
        Sprite sprite = Sprite.Create(texture, new Rect(0, 0, texture.width, texture.height), Vector2.zero);
        allFaces.Add("idle", sprite);

        texture = Resources.Load(StringResources.listening_orb_path) as Texture2D;
        sprite = Sprite.Create(texture, new Rect(0, 0, texture.width, texture.height), Vector2.zero);
        allFaces.Add("mic", sprite);
        //innerface.sprite = allFaces["idle"];

        messageContainer = transform.GetChild(0).GetChild(1).gameObject;
        message = messageContainer.GetComponentInChildren<TMPro.TextMeshPro>();
        message.text = "";

        messageContainer.SetActive(false);

        timerContainer = transform.GetChild(0).GetChild(2).gameObject;
        timer = timerContainer.GetComponentInChildren<TMPro.TextMeshPro>();
        timer.text = "";

        timerContainer.SetActive(false);

        recordingIcon = face.transform.GetChild(1).gameObject;
        recordingIcon.SetActive(false);

        eyeEvents = face.transform.GetComponent<EyeTrackingTarget>();
        eyeEvents.WhileLookingAtTarget.AddListener(delegate { CurrentlyLooking(true); });
        eyeEvents.OnLookAway.AddListener(delegate { CurrentlyLooking(false); });

        eyeEvents.OnSelected.AddListener(delegate { MenuSelected(); });

        taskListbutton = transform.GetChild(0).GetChild(3).gameObject;
        taskListbutton.AddComponent<DwellButtonTaskList>();

        followSolver = gameObject.GetComponentInChildren<RadialView>();

        speechInput = gameObject.GetComponentInChildren<SpeechInputHandler>();
        speechInput.AddResponse("Help", delegate { HelpSelected(); });

        taskListbutton.SetActive(false);
    }

    public void ActivateTaskListButton(bool isActive)
    {
        taskListbutton.SetActive(isActive);
    }


    private void MenuSelected()
    {
        if (isProcessingSpeechInput) return;

        Debug.Log("Menu Voice Select was actived");
        isProcessingSpeechInput = true;
        StartCoroutine(TurnOnOnffCurrentTask());
    }

    private IEnumerator TurnOnOnffCurrentTask()
    {
        //DemoHandler.Instance.TurnOnOffCurrentTask();

        yield return new WaitForSeconds(1);

        isProcessingSpeechInput = false;
    }

    private void HelpSelected()
    {
        Debug.Log("Help Voice Select was actived");
        //DemoHandler.Instance.TurnOnOffHints();
    }

    private void CurrentlyLooking(bool v)
    {
        isLooking = v;
        SetFollow(!v);

        if (v)
            recordingIcon.SetActive(true);
        else
            recordingIcon.SetActive(false);
    }

    public void SetMessage(string message, bool show)
    {
        this.message.text = message;

        if (show)
            ToggleMessage(true);
    }

    public void ToggleMessage(bool isOn)
    {
        if (!isOn)
            message.text = "";

        messageContainer.SetActive(isOn);
    }

    public void SetTime(string time) => timer.text = time;

    public void ShowTimer(bool isOn) => timerContainer.SetActive(isOn);

    public void SetFollow(bool isOn) => followSolver.enabled = isOn;
}
