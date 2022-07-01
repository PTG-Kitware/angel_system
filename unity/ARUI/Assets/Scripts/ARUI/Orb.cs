using DilmerGames.Core.Singletons;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities.Solvers;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

public class Orb : Singleton<Orb>
{
    //The white sphere represents the 'face' of the assistant
    private GameObject face;
    private SpriteRenderer faceSprite;
    private GameObject recordingIcon;
    private Dictionary<string, Sprite> allFaces;
    
    //Text feedback 
    private GameObject messageContainer;
    private TMPro.TextMeshProUGUI message;
    private bool isMessageActive = false;
    private Material messageContainerMaterial;
    private RectTransform messageContainerBackground;
    private int maxLineCount = 70;
    private EyeTrackingTarget messageEyeEvents;

    // Eye-gaze based updates
    private bool isLookingAtMessage = false;
    private bool isMessageVisible = false;
    private bool isMessageFading = false;
    private float currentAlpha = 1f;

    private Color activeColor = new Color(0.06f, 0.06f, 0.06f, 0.5f);
    private float step = 0.005f;

    private GameObject timerContainer;
    private TMPro.TextMeshProUGUI timer;
    private RectTransform timerContainerBackground;    

    //Input events 
    private EyeTrackingTarget eyeEvents;

    //Placement behaviors
    private Orbital followSolver;
    private bool isProcessingSmoothFollow = false;
    
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
        message = messageContainer.GetComponentInChildren<TMPro.TextMeshProUGUI>();
        message.text = "";

        messageContainerBackground = message.GetComponent<RectTransform>();
        messageContainerMaterial = messageContainer.GetComponentInChildren<Image>().material;

        messageEyeEvents = messageContainer.transform.GetComponent<EyeTrackingTarget>();
        messageEyeEvents.OnLookAtStart.AddListener(delegate { IsLookingAtMessage(true); });
        messageEyeEvents.OnLookAway.AddListener(delegate { IsLookingAtMessage(false); });

        messageContainer.SetActive(false);

        timerContainer = transform.GetChild(0).GetChild(2).gameObject;
        timer = timerContainer.GetComponentInChildren<TMPro.TextMeshProUGUI>();
        timer.text = "";

        timerContainer.SetActive(false);

        timerContainerBackground = timer.GetComponent<RectTransform>();

        recordingIcon = face.transform.GetChild(1).gameObject;
        recordingIcon.SetActive(false);

        //Init input events
        eyeEvents = face.transform.GetComponent<EyeTrackingTarget>();
        eyeEvents.OnLookAtStart.AddListener(delegate { IsLookingAtFace(true); });
        eyeEvents.OnLookAway.AddListener(delegate { IsLookingAtFace(false); });
        
        followSolver = gameObject.GetComponentInChildren<Orbital>();

        //Init tasklist button
        GameObject taskListbtn = transform.GetChild(0).GetChild(3).gameObject;
        taskListbutton = taskListbtn.AddComponent<DwellButtonTaskList>();
        taskListbutton.gameObject.SetActive(false);
    }

    public void Update()
    {
        UpdateMessageEyeEvents();
        
    }

    public void SetMessage(string message)
    {
        if (isMessageActive && !messageContainer.activeSelf)
            messageContainer.SetActive(true);

        if (message.Length < 1 && isMessageActive && messageContainer.activeSelf)
            messageContainer.SetActive(false);

        var charCount = 0;
        var lines = message.Split(new string[] { " " }, StringSplitOptions.RemoveEmptyEntries)
                        .GroupBy(w => (charCount += w.Length + 1) / maxLineCount)
                        .Select(g => string.Join(" ", g));

        this.message.text = String.Join("\n", lines.ToArray());
    }

    private void LateUpdate()
    {
        face.transform.localPosition = new Vector3(messageContainerBackground.rect.x - 0.040f, 0, -0.001f);
        taskListbutton.transform.localPosition = new Vector3((messageContainerBackground.rect.x - 0.005f), 0, -0.001f);
    }

    public void SetMessageActive(bool isActive)
    {
        isMessageActive = isActive;

        if ( (isMessageActive && message.text.Length>0) || !isMessageActive)
            messageContainer.SetActive(isMessageActive);
    }

    public void SetTime(string time) => timer.text = time;

    public void SetTimerActive(bool isActive) => timerContainer.SetActive(isActive);

    public void SetFollowActive(bool isActive)
    {
        if (isActive && !isProcessingSmoothFollow)
            StartCoroutine(SmoothFollow());

        else if (isActive == false)
        {
            StopCoroutine(SmoothFollow());
            followSolver.enabled = false;
            isProcessingSmoothFollow = false;
        }
    }

    private IEnumerator SmoothFollow()
    {
        isProcessingSmoothFollow = true;

        while (Utils.InFOV(AngelARUI.Instance.mainCamera, faceSprite.transform.position))
            yield return new WaitForSeconds(1f);


        yield return new WaitForSeconds(2f);
        
        followSolver.enabled = true;
        isProcessingSmoothFollow = false;
    }

    public void SetTaskListButtonActive(bool isActive) => taskListbutton.gameObject.SetActive(isActive);

    #region eye-gaze message events
    private void UpdateMessageEyeEvents()
    {
        if (isLookingAtMessage && !isMessageVisible && !isMessageFading)
        {
            messageContainerMaterial.color = activeColor;
            SetTextAlpha(1f);
            isMessageVisible = true;
        }
        else if (isLookingAtMessage && isMessageVisible && isMessageFading)
        {
            StopCoroutine(FadeOutMessage());

            isMessageFading = false;
            messageContainerMaterial.color = activeColor;
            SetTextAlpha(1f);
        }
        else if (!isLookingAtMessage && isMessageVisible && !isMessageFading)
        {
            StartCoroutine(FadeOutMessage());
        }
    }

    private IEnumerator FadeOutMessage()
    {
        isMessageFading = true;

        yield return new WaitForSeconds(1.0f);

        float shade = activeColor.r;
        float alpha = 1f;

        while (isMessageFading && shade > 0)
        {
            alpha -= (step * 20);
            shade -= step;

            messageContainerMaterial.color = new Color(shade, shade, shade);

            if (alpha >= 0)
                SetTextAlpha(Mathf.Max(0, alpha));

            yield return new WaitForEndOfFrame();
        }

        isMessageFading = false;
        isMessageVisible = false;
    }

    public void SetTextAlpha(float alpha)
    {
        message.color = new Color(message.color.r, message.color.g, message.color.b, alpha);
        currentAlpha = alpha;
    }

    #endregion

    #region callbacks input events

    private void IsLookingAtMessage(bool isLooking) => isLookingAtMessage = isLooking;

    private void IsLookingAtFace(bool isLooking)
    {
        SetFollowActive(!isLooking);

        recordingIcon.SetActive(isLooking);
    }

    #endregion
}
