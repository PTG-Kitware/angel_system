using DilmerGames.Core.Singletons;
using Microsoft.MixedReality.Toolkit;
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
    private Shapes.Disc faceBG;
    private Shapes.Disc draggableHandle;
    private Color faceColorInner = new Color(1, 1, 1, 1f);

    private bool IsLookingAtOrb = false;
    private bool allowRepositioning = true;

    //Text feedback 
    private GameObject messageContainer;
    private TMPro.TextMeshProUGUI message;
    private bool isMessageActive = false;
    private Material messageContainerMaterial;
    private RectTransform messageContainerBackground;
    private int maxLineCount = 70;

    // Eye-gaze based updates
    private bool isMessageVisible = false;
    private bool isMessageFading = false;
    private float currentAlpha = 1f;

    private Color activeColorBG = new Color(0.06f, 0.06f, 0.06f, 0.5f);
    private Color activeColorText = Color.white;
    private float step = 0.002f;

    private GameObject timerContainer;
    private TMPro.TextMeshProUGUI timer;
    private RectTransform timerContainerBackground;    

    //Input events 
    private EyeTrackingTarget eyeEvents;

    //Placement behaviors
    private Orbital followSolver;
    private bool lazyFollowStarted = false;

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

        Shapes.Disc[] allDiscs = face.GetComponentsInChildren<Shapes.Disc>();
        faceBG = allDiscs[0];
        draggableHandle = allDiscs[1];
        draggableHandle.gameObject.SetActive(false);

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
        UpdateOrbVisibility();
        UpdateOrbPosition();
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
        //face.transform.localPosition = new Vector3(messageContainerBackground.rect.x - 0.040f, 0, -0.001f);
        //taskListbutton.transform.localPosition = new Vector3((messageContainerBackground.rect.x - 0.005f), 0, -0.001f);
    }

    public void SetMessageActive(bool isActive)
    {
        isMessageActive = isActive;

        if ( (isMessageActive && message.text.Length>0) || !isMessageActive)
            messageContainer.SetActive(isMessageActive);
    }

    public void SetTime(string time) => timer.text = time;
    public void SetTimerActive(bool isActive) => timerContainer.SetActive(isActive);
    public void SetTaskListButtonActive(bool isActive) => taskListbutton.gameObject.SetActive(isActive);
    private void IsLookingAtFace(bool isLooking) => IsLookingAtOrb = isLooking;

    public void SetNearHover(bool isHovering)
    {
        if (isHovering)
        {
            draggableHandle.gameObject.SetActive(true);
            allowRepositioning = false;
        }
        else {

            draggableHandle.gameObject.SetActive(false);
            allowRepositioning = true;
        }
    }

    public void SetIsDragging(bool isDragging)
    {
        if (isDragging)
            faceBG.ColorInner = Color.black;
        else
            faceBG.ColorInner = faceColorInner;
    }

    #region eye-gaze message events

    /// <summary>
    /// Update the visibility of the orb message based on eye gaze collisions with the orb collider 
    /// </summary>
    private void UpdateOrbVisibility()
    {
        if ((IsLookingAtOrb || taskListbutton.isLooking) && !isMessageVisible && !isMessageFading)
        { //Set the message visible!
            messageContainerMaterial.color = activeColorBG;
            SetTextAlpha(1f);
            isMessageVisible = true;
        }
        else if (IsLookingAtOrb && isMessageVisible && isMessageFading)
        { //Stop Fading, set the message visible
            StopCoroutine(FadeOutMessage());

            isMessageFading = false;
            messageContainerMaterial.color = activeColorBG;
            SetTextAlpha(1f);
        }
        else if (!IsLookingAtOrb && !taskListbutton.isLooking && isMessageVisible && !isMessageFading)
        { //Start Fading
            StartCoroutine(FadeOutMessage());
        }
    }

    private IEnumerator FadeOutMessage()
    {
        isMessageFading = true;

        yield return new WaitForSeconds(1.0f);

        float shade = activeColorBG.r;
        float alpha = 1f;

        while (isMessageFading && shade > 0)
        {
            alpha -= (step * 20);
            shade -= step;

            if (alpha < 0)
                alpha = 0;
            if (shade < 0)
                shade = 0;

            messageContainerMaterial.color = new Color(shade, shade, shade, shade);
            SetTextAlpha(alpha);


            yield return new WaitForEndOfFrame();
        }

        isMessageFading = false;
        isMessageVisible = false;
    }

    private void SetTextAlpha(float alpha)
    {
        if (alpha == 0)
            message.color = new Color(0, 0, 0, 0);
        else
            message.color = new Color(activeColorText.r, activeColorText.g, activeColorText.b, alpha);

        currentAlpha = alpha;
    }

    private void UpdateOrbPosition()
    {
        if (CoreServices.InputSystem.EyeGazeProvider.GazeTarget == null)
            IsLookingAtOrb = false;

        if (IsLookingAtOrb && !CoreServices.InputSystem.EyeGazeProvider.GazeTarget.name.Contains("Face"))
            IsLookingAtOrb = false;

        //Debug.Log(followSolver.isActiveAndEnabled + " " + lazyFollowStarted + " " + Utils.InFOV(AngelARUI.Instance.mainCamera, faceSprite.transform.position));

        if (!allowRepositioning && followSolver.isActiveAndEnabled)
        {
            if (lazyFollowStarted)
                StopCoroutine(EnableLazyFollow());

            followSolver.enabled = false;
            return;
        } else if (allowRepositioning)
        {
            if ((IsLookingAtOrb || taskListbutton.isLooking) && !followSolver.isActiveAndEnabled && lazyFollowStarted)
            { // Stop Lazy Follow
                StopCoroutine(EnableLazyFollow());
            }
            else if ((IsLookingAtOrb || taskListbutton.isLooking) && followSolver.isActiveAndEnabled && !lazyFollowStarted)
            { //Stop follow
                followSolver.enabled = false;
            }
            else if (!followSolver.enabled && !lazyFollowStarted && !Utils.InFOV(AngelARUI.Instance.mainCamera, faceSprite.transform.position))
            { //Start Lazy Follow
                StartCoroutine(EnableLazyFollow());
            }
        }
        
    }


    private IEnumerator EnableLazyFollow()
    {
        lazyFollowStarted = true;

        yield return new WaitForSeconds(2f);

        if (!IsLookingAtOrb && !taskListbutton.isLooking)
        {
            followSolver.enabled = true;
        }

        lazyFollowStarted = false;
    }

    #endregion

}
