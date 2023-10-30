using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

//Handles setting up the text + required items for the flashcards 

public class SetupCurrTaskOverview : MonoBehaviour
{ 
    public ManageStepFlashcardSolo currFlashcardSolo;
    public ManageStepFlashcardSolo prevFlashcard;
    public ManageStepFlashcardSolo[] nextFlashcards;

    public void SetupCurrTask(List<Step> allSteps, int currentStep)
    {
        if (allSteps == null || currentStep <= -1)
        {
            currFlashcardSolo.SetAsDone("Task Done");
        } else {
            currFlashcardSolo.SetFlashcard(allSteps[currentStep], currentStep+1, allSteps.Count);
        }
    }

    public void SetupPrevTask(List<Step> allSteps, int prevStep)
    {
        if (allSteps == null || prevStep <= -1)
        {
            prevFlashcard.gameObject.SetActive(false);
        }
        else
        {
            prevFlashcard.gameObject.SetActive(true);
            prevFlashcard.SetFlashcard(allSteps[prevStep]);
        }
    }

    public void SetupNextTasks(List<Step> allSteps, int nextStep)
    {
        if (allSteps == null || nextStep <= -1)
        {
            foreach (var card in nextFlashcards)
                card.gameObject.SetActive(false);
        }
        else
        {
            int i = nextStep;
            foreach (var card in nextFlashcards)
            {
                if (i < allSteps.Count)
                {
                    card.gameObject.SetActive(true);
                    card.SetFlashcard(allSteps[i]);
                }
                else
                    card.gameObject.SetActive(false);

                i++;
            }
        }
    }

    public void DeactivateNextTasks()
    {
        foreach (var card in nextFlashcards)
            card.gameObject.SetActive(false);
    }
}
