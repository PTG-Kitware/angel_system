using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class SubTaskStep : MonoBehaviour
{
    [SerializeField]
    TMP_Text TextObj;

    public void SetSubStepText(string text)
    {
        TextObj.SetText(text);
    }
}
