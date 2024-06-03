using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

public class OrbSpeechBubble : MonoBehaviour
{
    private TMPro.TextMeshProUGUI _textComponent;

    private bool _isFading = false;

    private float _currentTimeOut = 0;

    private GameObject _eyeGazeTarget;

    // Start is called before the first frame update
    public void Init()
    {
        HorizontalLayoutGroup temp = gameObject.GetComponentInChildren<HorizontalLayoutGroup>();
        //init task message group
        RectTransform _HGroupTaskMessage = temp.gameObject.GetComponent<RectTransform>();
        TMPro.TextMeshProUGUI[] allText = _HGroupTaskMessage.gameObject.GetComponentsInChildren<TMPro.TextMeshProUGUI>();

        _textComponent = allText[0];
        _textComponent.text = "";

        _eyeGazeTarget = gameObject.GetComponentInChildren<BoxCollider>().gameObject;
        EyeGazeManager.Instance.RegisterEyeTargetID(_eyeGazeTarget);
    }

    public void Update()
    {
        var lookPos = transform.position - AngelARUI.Instance.ARCamera.transform.position;
        lookPos.y = 0;
        transform.rotation = Quaternion.LookRotation(lookPos, Vector3.up);
    }

    public void OnDisable()
    {
        _isFading = false;
        StopCoroutine(FadeGPTDialogue());
        Orb.Instance.SetDialogueActive(false);
    }

    public void SetText(string utterance, string response, float timeout)
    {
        string res_short = Utils.SplitTextIntoLines(response, ARUISettings.OrbMessageMaxCharCountPerLine);

        if (utterance.Length==0)
        {
            _textComponent.text = res_short;
        } else
        {
            string utt_short = Utils.SplitTextIntoLines(utterance, ARUISettings.OrbMessageMaxCharCountPerLine);

            _textComponent.text = "<b>You:</b> " + utt_short + "\n\n" + "<b>Angel:</b> " + res_short;
        }

        _currentTimeOut = Mathf.Max(timeout, _currentTimeOut);

        if (!_isFading)
        {
            StartCoroutine(FadeGPTDialogue());
        }
    }

    private IEnumerator FadeGPTDialogue()
    {
        _isFading = true;
        while (_isFading && _currentTimeOut > 0)
        {
            if (EyeGazeManager.Instance.CurrentHitID!= _eyeGazeTarget.GetInstanceID())
            {
                _currentTimeOut -= Time.deltaTime;
            }

            yield return new WaitForEndOfFrame();
        }

        if (_currentTimeOut <= 0)
            Orb.Instance.SetDialogueActive(false);

        _isFading = false;
    }
}
