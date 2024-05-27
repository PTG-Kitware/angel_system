using UnityEngine;

public class OrbWarning : MonoBehaviour
{
    private bool _init = false;

    private RectTransform _notificationMessageRect;

    private TMPro.TextMeshProUGUI _textContent;
    public TMPro.TextMeshProUGUI Text {  get { return _textContent; } }

    public bool IsSet => _textContent.text.Length > 0;

    private float _xOffset = 0;
    public float XOffset
    {
        get { return _xOffset; }
    }

    /// <summary>
    /// Ini
    /// </summary>
    /// <param name="message"></param>
    public void Init(string message, float containerHeight)
    {
        if (!_init)
        {
            //init notification message group
            _textContent = gameObject.GetComponentInChildren<TMPro.TextMeshProUGUI>();

            //init notification message group
            _notificationMessageRect = gameObject.GetComponent<RectTransform>();

            _textContent.text = Utils.SplitTextIntoLines(message, ARUISettings.OrbMessageMaxCharCountPerLine);
            _xOffset = transform.localPosition.x;

            _notificationMessageRect.rotation = Quaternion.identity;
            _notificationMessageRect.localRotation = Quaternion.identity;
            UpdateYPos(containerHeight, false);
            _notificationMessageRect.SetLocalZPos(0);

            _init = true;
        }
    }

    public void UpdateSize(float xSize)
    {
        if (!_init)
            _notificationMessageRect.sizeDelta = new Vector2(xSize, _notificationMessageRect.rect.height);
    }

    /// <summary>
    /// Set the notification of the notification
    /// </summary>
    /// <param name="message"></param>
    /// <param name="charPerLine"></param>
    public void SetMessage(string message, int charPerLine) => _textContent.text = Utils.SplitTextIntoLines(message, charPerLine);

    public void UpdateYPos(float containerHeight, bool prevMessageActive)
    {
        if (!prevMessageActive)
            _notificationMessageRect.SetLocalYPos(containerHeight);
        else 
            _notificationMessageRect.SetLocalYPos(containerHeight);
    }
}
