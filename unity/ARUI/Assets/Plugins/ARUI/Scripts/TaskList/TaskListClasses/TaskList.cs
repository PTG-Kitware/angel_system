using System.Collections.Generic;

[System.Serializable]
public class TaskList
{
    public string Name;
    public List<Step> Steps;
    public int CurrStepIndex;
    public int PrevStepIndex;
    public int NextStepIndex;
}
