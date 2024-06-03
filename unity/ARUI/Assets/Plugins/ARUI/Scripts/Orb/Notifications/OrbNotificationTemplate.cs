using Microsoft.MixedReality.Toolkit.Input;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

public enum NotificationType
{
    MulitpleChoice = 0,
    OkayChoice = 1,
    YesNo = 2
}
public class OrbNotificationTemplate : MonoBehaviour
{
    public NotificationType type = NotificationType.OkayChoice;
    protected bool _init = false;                      /// <true if dialogue was initialized (e.g. message, event)
    protected bool _timerStarted = false;              /// <true if timer started already

    protected FlexibleTextContainer _textContainer;

    protected UnityEvent _timeOutEvent;                /// <Event that will be invoked if the notification timesout
    protected UnityEvent _selfDestruct;

    protected Shapes.Line _time;                       /// <Line that shows the user how much time is left to make a decision

    protected float _timeOutInSeconds = 10;

}
