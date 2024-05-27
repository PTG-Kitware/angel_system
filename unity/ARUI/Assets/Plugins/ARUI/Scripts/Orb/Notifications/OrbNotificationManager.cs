using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

public enum AcceptedSpeechInput{

    SelectA = 0, SelectB = 1, SelectC = 2, SelectD = 3, SelectE = 4,
    SelectYes = 5, SelectNo = 6,
    SelectOkay = 7,
}

public class OrbNotificationManager : MonoBehaviour
{
    private Dictionary<int, OrbNotificationTemplate> _allNotificationDialog = null;     /// <Reference to confirmation dialogue

    ///****** All notification prefabs
    private GameObject _confirmationNotificationPrefab = null;
    private GameObject _selectNotificationPrefab = null;
    private GameObject _yesNoNotificationPrefab = null;

    // Start is called before the first frame update
    private void Start()
    {
        _allNotificationDialog = new Dictionary<int, OrbNotificationTemplate>();

        //Load resources for UI elements
        _confirmationNotificationPrefab = Resources.Load(StringResources.ConfNotificationOrb_path) as GameObject;
        _confirmationNotificationPrefab.gameObject.name = "***ARUI-ConfirmationNotification";
        AudioManager.Instance.RegisterKeyword("Select Okay", () => { HandleNotificationSpeechInput(AcceptedSpeechInput.SelectOkay); });

        _selectNotificationPrefab = Resources.Load(StringResources.MultiSelectNotificationOrb_path) as GameObject;
        _selectNotificationPrefab.gameObject.name = "***ARUI-SelectNotification";
        AudioManager.Instance.RegisterKeyword("Select A", () => { HandleNotificationSpeechInput(AcceptedSpeechInput.SelectA); });
        AudioManager.Instance.RegisterKeyword("Select B", () => { HandleNotificationSpeechInput(AcceptedSpeechInput.SelectB); });
        AudioManager.Instance.RegisterKeyword("Select C", () => { HandleNotificationSpeechInput(AcceptedSpeechInput.SelectC); });
        AudioManager.Instance.RegisterKeyword("Select D", () => { HandleNotificationSpeechInput(AcceptedSpeechInput.SelectD); });
        AudioManager.Instance.RegisterKeyword("Select E", () => { HandleNotificationSpeechInput(AcceptedSpeechInput.SelectE); });

        _yesNoNotificationPrefab = Resources.Load(StringResources.YesNoNotificationOrb_path) as GameObject;
        _yesNoNotificationPrefab.gameObject.name = "***ARUI-YesNoNotification";
        AudioManager.Instance.RegisterKeyword("Select Yes", () => { HandleNotificationSpeechInput(AcceptedSpeechInput.SelectYes); });
        AudioManager.Instance.RegisterKeyword("Select No", () => { HandleNotificationSpeechInput(AcceptedSpeechInput.SelectNo); });
    }

    public void HandleNotificationSpeechInput(AcceptedSpeechInput input)
    { 
        List<OrbNotificationTemplate> instancesOfType= new List<OrbNotificationTemplate> ();
        switch (input)
        {
            case AcceptedSpeechInput.SelectOkay:
                foreach (var obj in _allNotificationDialog.Values)
                {
                    if (obj.type.Equals(NotificationType.OkayChoice))
                        instancesOfType.Add(obj);
                }
                if (instancesOfType.Count == 1)
                {
                    ((OrbConfirmationNotification)instancesOfType[0]).Confirmed(true);
                }
                else if (instancesOfType.Count > 1)
                {
                    //TODO
                }

                break;
            case AcceptedSpeechInput.SelectA:
            case AcceptedSpeechInput.SelectB:
            case AcceptedSpeechInput.SelectC:
            case AcceptedSpeechInput.SelectD:
            case AcceptedSpeechInput.SelectE:
                foreach (var obj in _allNotificationDialog.Values)
                {
                    if (obj.type.Equals(NotificationType.MulitpleChoice))
                        instancesOfType.Add(obj);
                }
                if (instancesOfType.Count == 1)
                {
                    ((OrbMultipleChoiceNotification)instancesOfType[0]).ConfirmedViaSpeech(input);
                }
                else if (instancesOfType.Count > 1)
                {
                    //TODO
                }
                break;

            case AcceptedSpeechInput.SelectNo:
            case AcceptedSpeechInput.SelectYes:
                foreach (var obj in _allNotificationDialog.Values)
                {
                    if (obj.type.Equals(NotificationType.YesNo))
                        instancesOfType.Add(obj);
                }
                if (instancesOfType.Count == 1)
                {
                    ((OrbYesNoNotification)instancesOfType[0]).ConfirmedViaSpeech(input);
                }
                else if (instancesOfType.Count > 1)
                {
                    //TODO
                }
                break;


        }

        
    }

    /// <summary>
    /// If confirmation action is set - SetUserIntentCallback(...) - and no confirmation window is active at the moment, the user is shown a 
    /// timed confirmation window. Recommended text: "Did you mean ...". If the user confirms the dialogue, the onUserIntentConfirmedAction action is invoked. 
    /// </summary>
    /// <param name="msg">Message that is shown in the Confirmation Dialogue</param>
    /// <param name="actionOnConfirmation">Action triggerd if the user confirms the dialogue</param>
    /// <param name="actionOnTimeOut">OPTIONAL - Action triggered if notification times out</param>
    public void TryGetUserConfirmation(string msg, List<UnityAction> actionOnConfirmation, UnityAction actionOnTimeOut, float timeOut)
    {
        if (_confirmationNotificationPrefab == null || !Utils.StringValid(msg)) return;

        GameObject window = Instantiate(_confirmationNotificationPrefab, transform);
        window.gameObject.name = "***ARUI-Confirmation-" + msg;
        window.transform.SetParent(transform, true);
        OrbConfirmationNotification dialogue = window.AddComponent<OrbConfirmationNotification>();
        _allNotificationDialog.Add(window.gameObject.GetInstanceID(), dialogue);
        dialogue.InitNotification(msg, actionOnConfirmation, actionOnTimeOut, () => { DestroyWindow(window.gameObject.GetInstanceID()); }, timeOut) ;
    }

    public void TryGetUserChoice(string selectionMsg, List<string> choices, List<UnityAction> actionOnSelection, UnityAction actionOnTimeOut, float timeout)
    {
        if (choices.Count!= actionOnSelection.Count) return;

        GameObject window = Instantiate(_selectNotificationPrefab, transform);
        window.gameObject.name = "***ARUI-Multiselect-" + selectionMsg;
        window.transform.parent = transform;
        OrbMultipleChoiceNotification dialogue = window.AddComponent<OrbMultipleChoiceNotification>();
        _allNotificationDialog.Add(window.gameObject.GetInstanceID(), dialogue);
        dialogue.InitNotification(selectionMsg, choices, actionOnSelection, actionOnTimeOut, () => { DestroyWindow(window.gameObject.GetInstanceID()); }, timeout);
    }


    public void TryGetUserYesNoChoice(string selectionMsg, UnityAction actionOnYes, UnityAction actionOnNo, UnityAction actionOnTimeOut, float timeout)
    {
        if (!Utils.StringValid(selectionMsg)) return;

        GameObject window = Instantiate(_yesNoNotificationPrefab, transform);
        window.gameObject.name = "***ARUI-YesNo-" + selectionMsg;
        window.transform.parent = transform;
        OrbYesNoNotification dialogue = window.AddComponent<OrbYesNoNotification>();
        _allNotificationDialog.Add(window.gameObject.GetInstanceID(), dialogue);
        dialogue.InitNotification(selectionMsg, actionOnYes, actionOnNo, actionOnTimeOut, () => { DestroyWindow(window.gameObject.GetInstanceID()); }, timeout);
    }

        
    /// <summary>
    /// Destroy the given current notification window, happens either after timeout if user selected an option at the notification
    /// </summary>
    /// <param name="ID"></param>
    private void DestroyWindow(int ID)
    {
        Destroy(_allNotificationDialog[ID]);
        Destroy(_allNotificationDialog[ID].gameObject);
        _allNotificationDialog.Remove(ID);
    }
}
