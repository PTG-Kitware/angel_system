using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum NotificationType
{
    warning,
    note
}

public class Notification : MonoBehaviour
{
    public bool userHasSeen = false;

    private bool _init = false;

    private NotificationType _notificationType;

    private RectTransform _notificationMessageRect;
    private TMPro.TextMeshProUGUI _textNotification;
    public string GetMessage => _textNotification.text;

    public bool IsSet => _textNotification.text.Length > 0;

    public void init(NotificationType type, string message, float containerHeight)
    {
        if (!_init)
        {
            _notificationType = type;

            //init notification message group
            _notificationMessageRect = gameObject.GetComponent<RectTransform>();
            _textNotification = _notificationMessageRect.gameObject.GetComponent<TMPro.TextMeshProUGUI>();

            _textNotification.text = Utils.SplitTextIntoLines(message, ARUISettings.OrbMessageMaxCharCountPerLine);

            _notificationMessageRect.rotation = Quaternion.identity;
            _notificationMessageRect.localRotation = Quaternion.identity;   
            _notificationMessageRect.SetLocalXPos(0);
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

    public void SetMessage(string message, int charPerLine)
    {
        _textNotification.text = Utils.SplitTextIntoLines(message, charPerLine);
    }

    public void UpdateYPos(float containerHeight, bool taskListIsActive)
    {
        if (_notificationType == NotificationType.note && !taskListIsActive)
            _notificationMessageRect.SetLocalYPos(-containerHeight);
        else if (_notificationType == NotificationType.note && taskListIsActive)
            _notificationMessageRect.SetLocalYPos(-containerHeight);
        else if (_notificationType == NotificationType.warning && !taskListIsActive)
            _notificationMessageRect.SetLocalYPos(containerHeight);
        else if (_notificationType == NotificationType.warning && taskListIsActive) 
            _notificationMessageRect.SetLocalYPos(containerHeight);
    }
}
