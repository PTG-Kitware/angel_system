using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

//Contains all important parts of each task list
public class TaskOverviewContainerRepo : MonoBehaviour
{
    //This is for the rectnable to make the task list visible 
    //when user gazes at it
    public CurrentListActivator multiListInstance;
    //This handles setting up all the flashcards
    public SetupCurrTaskOverview setupInstance;
    //TMP object containing the name of the tasklist
    public TMP_Text taskNameText;
    //Task overview UI object 
    public GameObject taskUI;
}
