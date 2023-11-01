using UnityEngine;

public class Notification : MonoBehaviour
{
    private bool _init = false;

    private TMPro.TextMeshProUGUI _textNotification;
    public TMPro.TextMeshProUGUI Text {  get { return _textNotification; } }

    public bool IsSet => _textNotification.text.Length > 0;

    private float _xOffset = 0;
    public float XOffset
    {
        get { return _xOffset; }
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="message"></param>
    public void Init(string message)
    {
        if (!_init)
        {
            //init notification message group
            _textNotification = gameObject.GetComponentInChildren<TMPro.TextMeshProUGUI>();

            _textNotification.text = Utils.SplitTextIntoLines(message, ARUISettings.OrbMessageMaxCharCountPerLine);
            _xOffset = transform.localPosition.x;
            _init = true;
        }
    }

    /// <summary>
    /// Set the notification of the notification
    /// </summary>
    /// <param name="message"></param>
    /// <param name="charPerLine"></param>
    public void SetMessage(string message, int charPerLine) => _textNotification.text = Utils.SplitTextIntoLines(message, charPerLine);

}
