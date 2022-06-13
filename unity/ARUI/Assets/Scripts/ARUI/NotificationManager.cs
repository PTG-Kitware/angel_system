using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DilmerGames.Core.Singletons;
using TMPro;
using System;

public class NotificationManager : Singleton<NotificationManager>
{

    public void saveToEntityDictionary(GameObject arrObj)
    {
        Entity e = arrObj.AddComponent<Entity>();
        e.entityType = Type.Notification;
        e.id = arrObj.name;
        EntityManager.Instance.AddEntity(e);
    }
    public GameObject SpawnShortMessage(string name, string textContentShort, Vector3 position)
    {
        if (EntityManager.Instance.contains(name))
        {
            throw new InvalidOperationException(string.Format("Arrow with name{0} already exists", name));
        }
        GameObject msgBar = Resources.Load("Prefabs/MessageBar", typeof(GameObject)) as GameObject;
        msgBar.name = name;
        GameObject messageObj = msgBar.transform.GetChild(1).gameObject;
        TextMeshPro textArea = messageObj.GetComponent<TextMeshPro>();
        textArea.text = textContentShort;

        Instantiate(msgBar, position, Quaternion.identity);

        saveToEntityDictionary(msgBar);
        return msgBar;
    }

    public GameObject SpawnFullMessage(string name, string textContentShort, string textContentLong, Vector3 position)
    {
        if (EntityManager.Instance.contains(name))
        {
            throw new InvalidOperationException(string.Format("Arrow with name{0} already exists", name));
        }
        GameObject msgBoard = Resources.Load("Prefabs/MessageBoard", typeof(GameObject)) as GameObject;
        msgBoard.name = name;
        GameObject messageObj = msgBoard.transform.GetChild(1).gameObject;
        TextMeshPro textArea = messageObj.GetComponent<TextMeshPro>();
        textArea.text = textContentShort;

        messageObj = msgBoard.transform.GetChild(3).gameObject;
        textArea = messageObj.GetComponent<TextMeshPro>();
        textArea.text = textContentLong;

        Instantiate(msgBoard, position, Quaternion.identity);

        saveToEntityDictionary(msgBoard);
        return msgBoard;
    }

}
