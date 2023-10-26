using System.Globalization;
using TMPro;
using UnityEngine;

public class ManageStepFlashcardSolo : MonoBehaviour
{
    public TMP_Text TaskText;
    public Step currStep;

    public void InitializeFlashcard(Step newStep, int current, int allSteps)
    {
        currStep = newStep;
        TaskText.SetText("("+ current + "/"+ allSteps+") : "+currStep.StepDesc);

        foreach(string item in currStep.RequiredItems) {
            FindSubtext(item);
        }
    }

    public void InitializeFlashcard(Step newStep)
    {
        currStep = newStep;
        TaskText.SetText(currStep.StepDesc);

        foreach (string item in currStep.RequiredItems)
        {
            FindSubtext(item);
        }
    }

    public void FindSubtext(string substring)
    {
        string currTextLower = TaskText.text.ToLower(new CultureInfo("en-US", false));
        string currText = TaskText.text;
        int index = currTextLower.IndexOf(substring);
        if (index >= 0)
        {
            currText = currText.Insert(index, "<u><b>");
            currText = currText.Insert(index + substring.Length + 6, "</u></b>");
        }
        TaskText.SetText(currText);
    }
}
