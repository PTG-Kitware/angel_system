using DilmerGames.Core.Singletons;
using System;
using System.Collections.Generic;
using UnityEngine;

public class UIElementManager : Singleton<UIElementManager>
{
    private GameObject detectedObjectsParent;

    private void Start()
    {
        //Instantiate database
        detectedObjectsParent = new GameObject("DetectedObjectsParent");
        detectedObjectsParent.transform.parent = this.transform;
    }


}
