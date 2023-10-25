using System.Globalization;
using TMPro;
using UnityEngine;

public class ManageStepFlashcardSolo : MonoBehaviour
{
    public TMP_Text TaskText;
    public TasklistPositionManager centerScript;
    public Step currStep;

    public void InitializeFlashcard(Step newStep)
    {
        currStep = newStep;
        TaskText.SetText(currStep.StepDesc);
        foreach(string item in currStep.RequiredItems) {
            this.GetComponent<UnderlineSubText>().FindSubtext(item);
        }
    }

    void Update()
    {
        if (centerScript != null)
        {
            int increment = 1;
            foreach (string item in currStep.RequiredItems)
            {
                int offset = 6 * increment;
                string currTextLower = TaskText.text.ToLower(new CultureInfo("en-US", false));
                string currText = TaskText.text;
                int index = currTextLower.IndexOf(item);
                index -= offset;
                //UnityEngine.Debug.Log("Index of " + item + " is " + index);
                TMP_TextInfo textInfo = TaskText.textInfo;
                TMP_CharacterInfo charInfo = textInfo.characterInfo[index];
                Vector3 bottomLeft = charInfo.bottomRight;
                //UnityEngine.Debug.Log("Position of " + item +" is " + TaskText.transform.TransformPoint(bottomLeft).ToString());
                centerScript.SetLineStart(item, TaskText.transform.TransformPoint(bottomLeft));
                increment += 2;
            }
        }
    }
}
