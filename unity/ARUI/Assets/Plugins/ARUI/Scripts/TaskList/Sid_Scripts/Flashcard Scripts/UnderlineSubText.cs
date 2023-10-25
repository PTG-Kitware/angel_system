using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System.Globalization;

public class UnderlineSubText : MonoBehaviour
{
    public TMP_Text text;
    
    public void FindSubtext(string substring){
        string currTextLower = text.text.ToLower(new CultureInfo("en-US", false));
        string currText = text.text;
        int index = currTextLower.IndexOf(substring);
        if(index >= 0){
            currText = currText.Insert(index, "<u><b>");
            currText = currText.Insert(index + substring.Length + 6, "</u></b>");
        }
        text.SetText(currText);
    }
}
