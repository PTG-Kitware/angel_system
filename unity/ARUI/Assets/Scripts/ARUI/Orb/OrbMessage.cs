using Image = UnityEngine.UI.Image;
using System.Collections;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using System;

/// <summary>
/// Represents the message container next to the orb
/// </summary>
public class OrbMessage : MonoBehaviour
{
    public enum messageAnchor{
        left = 1, //message is left from the orb
        right = 2, //message is right from the orb
    }

    //Flexible Textbox for taskmessage
    private RectTransform HGroupTaskMessage;
    private TMPro.TextMeshProUGUI textTask;
    private RectTransform textTaskRect;
    private Material taskBackgroundMat;
    private Color32 glowColor = Color.white;
    private float maxglowAlpha = 0.3f;
    private BoxCollider taskMessageCollider;

    private Color activeColorBG = new Color(0.06f, 0.06f, 0.06f, 0.5f);
    private Color activeColorText = Color.white;

    public bool isMessageActive = false;

    //Flexible Textbox for Notification Message
    private RectTransform HGroupNotificationMessage;
    private TMPro.TextMeshProUGUI textNotification;
    public bool isNotificationActive = false;

    private int maxCharCountPerLine = 70;

    // flags
    public bool userHasNotSeenNewTask = false;
    public bool isLookingAtMessage = false;
    public bool isMessageVisible = false;
    public bool isMessageFading = false;
    public bool messageIsLerping = false;
 
    public messageAnchor currentAnchor = messageAnchor.right;

    private GameObject textContainer;
    private GameObject indicator;
    private Vector3 initialIndicatorPos;
    private float initialmessageYOffset;

    private TMPro.TextMeshProUGUI progressText;

    private void Start()
    {
        HorizontalLayoutGroup[] temp = gameObject.GetComponentsInChildren<HorizontalLayoutGroup>();

        //init task message group
        HGroupTaskMessage = temp[1].gameObject.GetComponent<RectTransform>();
        TMPro.TextMeshProUGUI[] allText = HGroupTaskMessage.gameObject.GetComponentsInChildren<TMPro.TextMeshProUGUI>();
        textTask = allText[0];
        textTask.text = "";
        textTaskRect = textTask.gameObject.GetComponent<RectTransform>();
        progressText = textTask.transform.GetChild(0).gameObject.GetComponent<TMPro.TextMeshProUGUI>();
        progressText.text = "";

        Image bkgr = HGroupTaskMessage.GetComponentInChildren<Image>();
        taskBackgroundMat = new Material(bkgr.material);
        bkgr.material = taskBackgroundMat;
        glowColor = bkgr.material.GetColor("_InnerGlowColor");
        bkgr.material.SetColor("_InnerGlowColor", new Color(glowColor.r, glowColor.g, glowColor.b, 0));

        textContainer = transform.GetChild(1).gameObject;
        initialmessageYOffset = textContainer.transform.position.x;
        taskMessageCollider = textContainer.GetComponent<BoxCollider>();

        //init notification message group
        HGroupNotificationMessage = temp[0].gameObject.GetComponent<RectTransform>();
        textNotification = HGroupNotificationMessage.gameObject.GetComponentInChildren<TMPro.TextMeshProUGUI>();
        textNotification.text = "";
        HGroupNotificationMessage.gameObject.SetActive(false);

        //message direction indicator
        indicator = gameObject.GetComponentInChildren<Shapes.Polyline>().gameObject;
        initialIndicatorPos = indicator.transform.position;

        SetActive(false);
    }

    private void Update()
    {
        // Update eye tracking flag
        if (isLookingAtMessage && FollowEyeTarget.Instance.currentHit != EyeTarget.orbMessage)
            isLookingAtMessage = false;
        else if (!isLookingAtMessage && FollowEyeTarget.Instance.currentHit == EyeTarget.orbMessage)
            isLookingAtMessage = true;

        // Update collider of messagebox
        taskMessageCollider.size = new Vector3(HGroupTaskMessage.rect.width, taskMessageCollider.size.y, taskMessageCollider.size.z);
        taskMessageCollider.center = new Vector3(HGroupTaskMessage.rect.width / 2, 0, 0);

        if (!(isMessageVisible && isMessageActive) || messageIsLerping ) return;

        // Update messagebox anchor
        if (MessageBoxBelongsRight(100))
        {
            UpdateAnchorLerp(messageAnchor.right);
        }
        else if (MessageBoxBelongsLeft(100))
        {
            UpdateAnchorLerp(messageAnchor.left);
        }

    }


    #region Message and Notification Updates
    /// <summary>
    /// Turn on or off message fading
    /// </summary>
    /// <param name="active"></param>
    public void SetFadeOutMessage(bool active)
    {
        if (active)
        {
            StartCoroutine(FadeOutMessage());
        } else
        {
            StopCoroutine(FadeOutMessage());
            isMessageFading = false;
            taskBackgroundMat.color = activeColorBG;
            SetTextAlpha(1f);
        }
    }

    /// <summary>
    /// Fade out message from the moment the user does not look at the message anymore
    /// </summary>
    /// <returns></returns>
    private IEnumerator FadeOutMessage()
    {
        float fadeOutStep = 0.001f;
        isMessageFading = true;

        yield return new WaitForSeconds(1.0f);

        float shade = activeColorBG.r;
        float alpha = 1f;

        while (isMessageFading && shade > 0)
        {
            alpha -= (fadeOutStep * 20);
            shade -= fadeOutStep;

            if (alpha < 0)
                alpha = 0;
            if (shade < 0)
                shade = 0;

            taskBackgroundMat.color = new Color(shade, shade, shade, shade);
            SetTextAlpha(alpha);


            yield return new WaitForEndOfFrame();
        }

        isMessageFading = false;

        if (shade <= 0)
        {
            SetActive(false);
            isMessageVisible = false;
        }
    }

    public void HandleNewTask() => StartCoroutine(FadeNewTaskGlow());

    private IEnumerator FadeNewTaskGlow()
    {
        SetFadeOutMessage(false);

        userHasNotSeenNewTask = true;

        taskBackgroundMat.SetColor("_InnerGlowColor", new Color(glowColor.r, glowColor.g, glowColor.b, maxglowAlpha));

        while (!isLookingAtMessage)
        {
            yield return new WaitForEndOfFrame();
        }

        float step = (maxglowAlpha / 10);
        float current = maxglowAlpha;
        while (current > 0)
        {
            current -= step;
            taskBackgroundMat.SetColor("_InnerGlowColor", new Color(glowColor.r, glowColor.g, glowColor.b, current));
            yield return new WaitForSeconds(0.1f);
        }

        taskBackgroundMat.SetColor("_InnerGlowColor", new Color(glowColor.r, glowColor.g, glowColor.b, 0f));
        userHasNotSeenNewTask = false;
    }

    #endregion

    #region Position Updates

    /// <summary>
    /// Updates the anchor of the messagebox smoothly
    /// </summary>
    /// <param name="messageAnchor">The new anchor</param>
    public void UpdateAnchorLerp(messageAnchor newMessageAnchor)
    {
        if (messageIsLerping) return; 

        if (newMessageAnchor != currentAnchor)
        {
            messageIsLerping = true;
            currentAnchor = newMessageAnchor;
            UpdateBoxIndicatorPos();

            StartCoroutine(MoveMessageBox(initialmessageYOffset, newMessageAnchor != messageAnchor.right, false));

        }
    }

    /// <summary>
    /// Updates the anchor of the messagebox instantly
    /// </summary>
    private void UpdateAnchorInstant()
    {
        taskMessageCollider.center = new Vector3(HGroupTaskMessage.rect.width / 2, 0, 0);
        taskMessageCollider.size = new Vector3(HGroupTaskMessage.rect.width, taskMessageCollider.size.y, taskMessageCollider.size.z);

        bool isLeft = false;
        if (MessageBoxBelongsLeft(0))
        {
            currentAnchor = messageAnchor.left;
            isLeft = true;
        }
        else
            currentAnchor = messageAnchor.right;

        UpdateBoxIndicatorPos();
        StartCoroutine(MoveMessageBox(initialmessageYOffset, isLeft, true));
    }


    private void UpdateBoxIndicatorPos()
    {
        if (currentAnchor == messageAnchor.right)
        {
            indicator.transform.localPosition = new Vector3(initialIndicatorPos.x, 0, 0);
            indicator.transform.localRotation = Quaternion.identity;
        } else
        {
            indicator.transform.localPosition = new Vector3(-initialIndicatorPos.x, 0, 0);
            indicator.transform.localRotation = Quaternion.Euler(0, 180, 0);
        }
    }

    private bool MessageBoxBelongsRight(float offsetPaddingInPixel)
    {
        return (AngelARUI.Instance.mainCamera.WorldToScreenPoint(transform.position).x < ((AngelARUI.Instance.mainCamera.pixelWidth * 0.5f) - offsetPaddingInPixel));
    }
    private bool MessageBoxBelongsLeft(float offsetPaddingInPixel)
    {
        return (AngelARUI.Instance.mainCamera.WorldToScreenPoint(transform.position).x > ((AngelARUI.Instance.mainCamera.pixelWidth * 0.5f) + offsetPaddingInPixel));
    }

    IEnumerator MoveMessageBox(float YOffset, bool addWidth, bool instant)
    {
        float initialYOffset = YOffset;
        float step = 0.1f;

        if (instant)
            step = 0.5f;

        while (step < 1)
        {
            if (addWidth)
                YOffset = -initialYOffset-taskMessageCollider.size.x;

            textContainer.transform.localPosition = Vector2.Lerp(textContainer.transform.localPosition, new Vector3(YOffset,0,0), step += Time.deltaTime);

            step += Time.deltaTime;
            yield return new WaitForEndOfFrame();
        }

        messageIsLerping = false;
    }

    #endregion

    #region Getter and Setter


    /// <summary>
    /// Sets the orb message to the given message and adds line break based on the max word count per line
    /// </summary>
    /// <param name="message"></param>
    public void SetTaskMessage(string message)
    {
        var charCount = 0;
        var lines = message.Split(new string[] { " " }, StringSplitOptions.RemoveEmptyEntries)
                        .GroupBy(w => (charCount += w.Length + 1) / maxCharCountPerLine)
                        .Select(g => string.Join(" ", g));

        this.textTask.text = String.Join("\n", lines.ToArray());

        progressText.text = TaskListManager.Instance.GetCurrentTaskID() + "/" + TaskListManager.Instance.GetTaskCount();

        if (message.Contains("Done") ){
            progressText.gameObject.SetActive(false);
        } else
        {
            progressText.gameObject.SetActive(true);
        }
    }

    public void SetTextAlpha(float alpha)
    {
        if (alpha == 0)
            textTask.color = new Color(0, 0, 0, 0);
        else
            textTask.color = new Color(activeColorText.r, activeColorText.g, activeColorText.b, alpha);
    }

    /// <summary>
    /// Actives or disactivates the messagebox of the orb.
    /// </summary>
    /// <param name="active"></param>
    public void SetActive(bool active)
    {
        textContainer.SetActive(active);
        indicator.SetActive(active);

        if (active)
        {
            UpdateAnchorInstant();
            taskBackgroundMat.color = activeColorBG;
            SetTextAlpha(1f);
        }
        else
            isMessageFading = false;

        isMessageVisible = active;
    }
   
    public bool isActive() => textContainer.activeSelf;

    public BoxCollider GetCollider() => taskMessageCollider;

    public void SetNotificationText(string message)
    {
        var charCount = 0;
        var lines = message.Split(new string[] { " " }, StringSplitOptions.RemoveEmptyEntries)
                        .GroupBy(w => (charCount += w.Length + 1) / maxCharCountPerLine)
                        .Select(g => string.Join(" ", g));

        textNotification.text = String.Join("\n", lines.ToArray());
    }

    public void SetNotificationTextActive(bool isActive)
    {
        HGroupNotificationMessage.gameObject.SetActive(isActive);
        isNotificationActive = isActive;

        if (!isActive)
            textNotification.text = "";

        if (isActive)
        {
            HGroupNotificationMessage.transform.SetLocalYPos(textTaskRect.rect.height / 2);
        }
    }



    #endregion
}