using Image = UnityEngine.UI.Image;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

/// <summary>
/// Represents the message container next to the orb
/// </summary>
public class OrbMessage : MonoBehaviour
{
    public enum MessageAnchor
    {
        left = 1, //message is left from the orb
        right = 2, //message is right from the orb
    }

    private MessageAnchor currentAnchor = MessageAnchor.right;

    //***Flexible Textbox for Notification Message
    private RectTransform notificationMessageRect;
    private TMPro.TextMeshProUGUI textNotification;

    private Color activeColorBG = new Color(0.06f, 0.06f, 0.06f, 0.5f);
    private Color32 glowColor = Color.white;
    private float maxglowAlpha = 0.3f;
    private Color activeColorText = Color.white;

    //Flags
    private bool isNotificationActive = false;
    public bool IsNotificationActive
    {
        get { return isNotificationActive; }
        set { SetNotificationTextActive(value); }
    }

    private bool userHasNotSeenNewTask = false;
    public bool UserHasNotSeenNewTask
    {
        get { return userHasNotSeenNewTask; }
    }

    private bool isLookingAtMessage = false;
    public bool IsLookingAtMessage
    {
        get { return isLookingAtMessage; }
    }

    private bool isMessageVisible = false;
    public bool IsMessageVisible
    {
        get { return isMessageVisible; }
    }

    private bool isMessageFading = false;
    public bool IsMessageFading
    {
        get { return isMessageFading; }
    }

    private bool messageIsLerping = false;

    private FlexibleTextContainer textContainer;
    private GameObject indicator;
    private Vector3 initialIndicatorPos;
    private float initialmessageYOffset;

    private TMPro.TextMeshProUGUI progressText;

    private void Start()
    {
        textContainer = transform.GetChild(1).gameObject.AddComponent<FlexibleTextContainer>();
        textContainer.gameObject.name += "_orb";

        TMPro.TextMeshProUGUI[] allText = textContainer.GetAllTextMeshComponents();

        progressText = allText[1].gameObject.GetComponent<TMPro.TextMeshProUGUI>();
        progressText.text = "";

        initialmessageYOffset = textContainer.transform.position.x;

        //init notification message group
        notificationMessageRect = allText[2].gameObject.GetComponent<RectTransform>();
        textNotification = notificationMessageRect.gameObject.GetComponent<TMPro.TextMeshProUGUI>();
        textNotification.text = "";
        notificationMessageRect.gameObject.SetActive(false);

        //message direction indicator
        indicator = gameObject.GetComponentInChildren<Shapes.Polyline>().gameObject;
        initialIndicatorPos = indicator.transform.position;

        glowColor = textContainer.GetGlowColor();

        SetIsActive(false, false);
    }

    private void Update()
    {
        // Update eye tracking flag
        if (isLookingAtMessage && FollowEyeTarget.Instance.currentHit != EyeTarget.orbMessage)
            isLookingAtMessage = false;
        else if (!isLookingAtMessage && FollowEyeTarget.Instance.currentHit == EyeTarget.orbMessage)
            isLookingAtMessage = true;

        notificationMessageRect.sizeDelta = new Vector2(textContainer.TextRect.width / 2, notificationMessageRect.rect.height);

        if (!(isMessageVisible && GetIsActive()) || messageIsLerping) return;

        // Update messagebox anchor
        if (ChangeMessageBoxToRight(100))
            UpdateAnchorLerp(MessageAnchor.right);

        else if (ChangeMessageBoxToLeft(100))
            UpdateAnchorLerp(MessageAnchor.left);
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
            textContainer.SetBackgroundColor(activeColorBG);

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

            textContainer.SetBackgroundColor(new Color(shade, shade, shade, shade));
            SetTextAlpha(alpha);

            yield return new WaitForEndOfFrame();
        }

        isMessageFading = false;

        if (shade <= 0)
        {
            SetIsActive(false, false);
            isMessageVisible = false;
        }
    }

    private IEnumerator FadeNewTaskGlow()
    {
        SetFadeOutMessage(false);

        userHasNotSeenNewTask = true;

        textContainer.SetGlowColor(new Color(glowColor.r, glowColor.g, glowColor.b, maxglowAlpha));

        while (!isLookingAtMessage)
        {
            yield return new WaitForEndOfFrame();
        }

        float step = (maxglowAlpha / 10);
        float current = maxglowAlpha;
        while (current > 0)
        {
            current -= step;
            textContainer.SetGlowColor(new Color(glowColor.r, glowColor.g, glowColor.b, current));
            yield return new WaitForSeconds(0.1f);
        }

        textContainer.SetGlowColor(new Color(glowColor.r, glowColor.g, glowColor.b, 0f));
        userHasNotSeenNewTask = false;
    }

    #endregion

    #region Position Updates

    /// <summary>
    /// Updates the anchor of the messagebox smoothly
    /// </summary>
    /// <param name="MessageAnchor">The new anchor</param>
    public void UpdateAnchorLerp(MessageAnchor newMessageAnchor)
    {
        if (messageIsLerping) return;

        if (newMessageAnchor != currentAnchor)
        {
            messageIsLerping = true;
            currentAnchor = newMessageAnchor;
            UpdateBoxIndicatorPos();

            StartCoroutine(MoveMessageBox(initialmessageYOffset, newMessageAnchor != MessageAnchor.right, false));
        }
    }

    /// <summary>
    /// Updates the anchor of the messagebox instantly (still need to run coroutine to allow the Hgroup rect to update properly
    /// </summary>
    private void UpdateAnchorInstant()
    {
        textContainer.UpdateAnchorInstant();

        bool isLeft = false;
        if (ChangeMessageBoxToLeft(0))
        {
            currentAnchor = MessageAnchor.left;
            isLeft = true;
        }
        else
            currentAnchor = MessageAnchor.right;

        UpdateBoxIndicatorPos();
        StartCoroutine(MoveMessageBox(initialmessageYOffset, isLeft, true));
    }

    /// <summary>
    /// Updates the position and orientation of the messagebox indicator
    /// </summary>
    private void UpdateBoxIndicatorPos()
    {
        if (currentAnchor == MessageAnchor.right)
        {
            indicator.transform.localPosition = new Vector3(initialIndicatorPos.x, 0, 0);
            indicator.transform.localRotation = Quaternion.identity;
        }
        else
        {
            indicator.transform.localPosition = new Vector3(-initialIndicatorPos.x, 0, 0);
            indicator.transform.localRotation = Quaternion.Euler(0, 180, 0);
        }
    }

    /// <summary>
    /// Check if message box should be anchored right
    /// </summary>
    /// <param name="offsetPaddingInPixel"></param>
    /// <returns></returns>
    private bool ChangeMessageBoxToRight(float offsetPaddingInPixel)
    {
        return (AngelARUI.Instance.ARCamera.WorldToScreenPoint(transform.position).x < ((AngelARUI.Instance.ARCamera.pixelWidth * 0.5f) - offsetPaddingInPixel));
    }

    /// <summary>
    /// Check if message box should be anchored left
    /// </summary>
    private bool ChangeMessageBoxToLeft(float offsetPaddingInPixel)
    {
        return (AngelARUI.Instance.ARCamera.WorldToScreenPoint(transform.position).x > ((AngelARUI.Instance.ARCamera.pixelWidth * 0.5f) + offsetPaddingInPixel));
    }

    /// <summary>
    /// Lerps the message box to the other side
    /// </summary>
    /// <param name="YOffset">y offset of the message box to the orb prefab</param>
    /// <param name="addWidth"> if messagebox on the left, change the signs</param>
    /// <param name="instant">if lerp should be almost instant (need to do this in a coroutine anyway, because we are waiting for the Hgroup to update properly</param>
    /// <returns></returns>
    IEnumerator MoveMessageBox(float YOffset, bool isLeft, bool instant)
    {
        float initialYOffset = YOffset;
        float step = 0.1f;

        if (instant)
            step = 0.5f;

        while (step < 1)
        {
            if (isLeft)
                YOffset = -initialYOffset - textContainer.MessageCollider.size.x;

            textContainer.transform.localPosition = Vector2.Lerp(textContainer.transform.localPosition, new Vector3(YOffset, 0, 0), step += Time.deltaTime);
            step += Time.deltaTime;
            yield return new WaitForEndOfFrame();
        }

        messageIsLerping = false;
    }

    #endregion

    #region Getter and Setter
    /// <summary>
    /// Returns true if the message box container gameObject is currently active, else false
    /// </summary>
    /// <returns></returns>
    public bool GetIsActive() => textContainer.gameObject.activeSelf;

    public BoxCollider GetMessageCollider() => textContainer.MessageCollider;

    /// <summary>
    /// Actives or disactivates the messagebox of the orb in the hierarchy
    /// </summary>
    /// <param name="active"></param>
    public void SetIsActive(bool active, bool newTask)
    {
        textContainer.gameObject.SetActive(active);
        indicator.SetActive(active);

        if (active)
        {
            UpdateAnchorInstant();
            textContainer.SetBackgroundColor(activeColorBG);
            SetTextAlpha(1f);
        }
        else
            isMessageFading = false;

        isMessageVisible = active;

        if (newTask)
        {
            StartCoroutine(FadeNewTaskGlow());
            if (isNotificationActive)
                SetNotificationMessage("");
        }
    }

    /// <summary>
    /// Sets the orb task message to the given message and adds line break based on maxCharCountPerLine
    /// </summary>
    /// <param name="message"></param>
    public void SetTaskMessage(string message)
    {
        textContainer.SetText(message);
        progressText.text = TaskListManager.Instance.GetCurrentTaskID() + "/" + TaskListManager.Instance.GetTaskCount();

        if (message.Contains("Done"))
        {
            progressText.gameObject.SetActive(false);
        }
        else
        {
            progressText.gameObject.SetActive(true);
        }
    }

    /// <summary>
    /// Sets the orb notification message to the given message and adds line break based on maxCharCountPerLine
    /// </summary>
    /// <param name="message"></param>
    public void SetNotificationMessage(string message)
    {
        textNotification.text = Utils.SplitTextIntoLines(message, textContainer.MaxCharCountPerLine);
    }

    /// <summary>
    /// Update the visibility of the notification message
    /// </summary>
    /// <param name="isActive"></param>
    private void SetNotificationTextActive(bool isActive)
    {
        notificationMessageRect.gameObject.SetActive(isActive);
        isNotificationActive = isActive;

        if (!isActive)
            textNotification.text = "";

        if (isActive)
            notificationMessageRect.transform.SetLocalYPos(textContainer.TextRect.height / 2);
    }

    /// <summary>
    /// Update the color of the text based on visibility
    /// </summary>
    /// <param name="alpha"></param>
    private void SetTextAlpha(float alpha)
    {
        if (alpha == 0)
            textContainer.SetTextColor(new Color(0, 0, 0, 0));
        else
            textContainer.SetTextColor(new Color(activeColorText.r, activeColorText.g, activeColorText.b, alpha));
    }

    #endregion
}