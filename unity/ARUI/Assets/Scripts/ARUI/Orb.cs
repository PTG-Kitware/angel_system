using DilmerGames.Core.Singletons;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities.Solvers;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Orb : Singleton<Orb>
{
    //The white sphere represents the 'face' of the assistant
    private GameObject face;
    private SpriteRenderer faceSprite;
    private GameObject recordingIcon;
    private Dictionary<string, Sprite> allFaces;
    
    //Text feedback 
    private GameObject messageContainer;
    private TMPro.TextMeshPro message;
    private bool isMessageActive = false;

    private GameObject timerContainer;
    private TMPro.TextMeshPro timer;

    //Input events 
    private EyeTrackingTarget eyeEvents;
    private SpeechInputHandler speechInput;
    private bool isProcessingSpeechInput = false;

    //Placement behaviors
    private Orbital followSolver;
    
    //temporary button that takes users eye gaze as input
    private DwellButtonTaskList taskListbutton;

    /// <summary>
    /// Instantiate and Initialise all objects related to the orb.
    /// </summary>
    void Awake()
    {
        gameObject.name = "Orb";

        //Load and Initialize faces of orb 
        faceSprite = GetComponentInChildren<SpriteRenderer>(true);
        face = faceSprite.transform.parent.parent.gameObject;

        allFaces = new Dictionary<string, Sprite>();

        Texture2D texture = Resources.Load(StringResources.idle_orb_path) as Texture2D;
        Sprite sprite = Sprite.Create(texture, new Rect(0, 0, texture.width, texture.height), Vector2.zero);
        allFaces.Add("idle", sprite);

        texture = Resources.Load(StringResources.listening_orb_path) as Texture2D;
        sprite = Sprite.Create(texture, new Rect(0, 0, texture.width, texture.height), Vector2.zero);
        allFaces.Add("mic", sprite);

        //Get gameobject references of orb prefab
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

        //Init input events
        eyeEvents = face.transform.GetComponent<EyeTrackingTarget>();
        eyeEvents.WhileLookingAtTarget.AddListener(delegate { CurrentlyLooking(true); });
        eyeEvents.OnLookAway.AddListener(delegate { CurrentlyLooking(false); });
        eyeEvents.OnSelected.AddListener(delegate { HelpSelected(); });

        followSolver = gameObject.GetComponentInChildren<Orbital>();
        speechInput = gameObject.GetComponentInChildren<SpeechInputHandler>();

        //Init tasklist button
        GameObject taskListbtn = transform.GetChild(0).GetChild(3).gameObject;
        taskListbutton = taskListbtn.AddComponent<DwellButtonTaskList>();
        taskListbutton.gameObject.SetActive(true);
    }

    public void SetMessage(string message)
    {
        if (isMessageActive && !messageContainer.activeSelf)
            messageContainer.SetActive(true);

        if (message.Length < 1 && isMessageActive && messageContainer.activeSelf)
            messageContainer.SetActive(false);

        this.message.text = message;
    }

    public void SetMessageActive(bool isActive)
    {
        isMessageActive = isActive;

        if (isMessageActive && message.text.Length>0)
        {
            messageContainer.SetActive(true);
        } else if (!isMessageActive)
            messageContainer.SetActive(false);
    }

    public void SetTime(string time) => timer.text = time;

    public void SetTimerActive(bool isActive) => timerContainer.SetActive(isActive);

    public void SetFollowActive(bool isActive) => followSolver.enabled = isActive;

    public void SetTaskListButtonActive(bool isActive) => taskListbutton.gameObject.SetActive(isActive);

    #region callbacks input events

    private void HelpSelected()
    {
        if (isProcessingSpeechInput) return;

        Debug.Log("Help Voice Select was actived");
        isProcessingSpeechInput = true;

        StartCoroutine(ToggleCurrentTaskActive());
    }

    private IEnumerator ToggleCurrentTaskActive()
    {
        SetMessageActive(!isMessageActive);

        yield return new WaitForSeconds(1);

        isProcessingSpeechInput = false;
    }

    private void CurrentlyLooking(bool v)
    {
        SetFollowActive(!v);

        if (v)
        {
            recordingIcon.SetActive(true);

        }
        else
            recordingIcon.SetActive(false);
    }

    #endregion
}
