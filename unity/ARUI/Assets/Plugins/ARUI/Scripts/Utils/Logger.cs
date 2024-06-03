using System.Linq;
using TMPro;
using UnityEngine;
using System;
using System.Collections.Generic;
using System.Diagnostics.Eventing.Reader;

public class Logger : Singleton<Logger>
{
    [SerializeField]
    private TextMeshProUGUI _debugAreaText = null;

    private int _maxMessages = 15;

    private List<string> _allMessages = new List<string>();

    public bool showUnityLog = true;

    public bool IsVisible {
        get => transform.GetChild(0).gameObject.activeSelf;
        set
        {
            transform.GetChild(0).gameObject.SetActive(value);
            transform.position = AngelARUI.Instance.ARCamera.transform.position + Vector3.Scale(AngelARUI.Instance.ARCamera.transform.forward, new Vector3(1.5f, 1.5f, 1.5f));
            transform.rotation = Quaternion.LookRotation(Logger.Instance.transform.position - AngelARUI.Instance.ARCamera.transform.position, Vector3.up);
        }
    }

    private void OnEnable()
    {
        if (_debugAreaText == null) return;

        Application.logMessageReceived += HandleUnityLog;
    }

    private void OnDisable()
    {
        if (_debugAreaText == null) return;

        Application.logMessageReceived -= HandleUnityLog;
    }

    private void HandleUnityLog(string logString, string stackTrace, LogType type)
    {
        if (showUnityLog == false) return;

        if (logString.Contains("colliders found in PokePointer overlap query")) return;

        string newMessage = $"<color=\"white\"><b>{DateTime.Now.ToString("HH:mm:ss.fff")} {logString} </b></color> <size=12><color=#d3d3d3>{"\n" + stackTrace}</color></size>\n";
        _allMessages.Add(newMessage);
        UpdateString();
    }

    public void LogInfo(string message)
    {
        if (_debugAreaText==null || showUnityLog) return;

        string newMessage = $"<color=\"green\"><b>{DateTime.Now.ToString("HH:mm:ss.fff")} {message}</b></color>\n";
        _allMessages.Add(newMessage);
        UpdateString();
    }

    private void UpdateString()
    {
        string[] outputArray = _allMessages.ToArray();
        Array.Reverse(outputArray);

        _debugAreaText.text = String.Join(String.Empty, outputArray);
    }

}