using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class Step 
{
    public string StepDesc;
    public List<string> RequiredItems;
    public List<SubStep> SubSteps;
    public int CurrSubStepIndex;
}
