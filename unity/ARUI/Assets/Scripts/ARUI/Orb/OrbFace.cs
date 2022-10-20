using Microsoft.MixedReality.Toolkit.Input;
using System;
using System.Collections.Generic;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;
using UnityEngine;

/// <summary>
/// Represents the visual representation of the orb (the disc)
/// </summary>
public class OrbFace : MonoBehaviour
{
    private GameObject notificatonIcon;
    private Dictionary<string, Sprite> allFaces;
    private Shapes.Disc faceBG;
    private Shapes.Disc draggableHandle;
    private Color faceColorInner = new Color(1, 1, 1, 1f);
    private Color faceColorOuter;
    private Color faceColorOuterNotification = new Color(0.68f, 0.51f, 0f, 0.3f);
    
    private void Start()
    {
        Shapes.Disc[] allDiscs = GetComponentsInChildren<Shapes.Disc>();
        faceBG = allDiscs[0];
        draggableHandle = allDiscs[1];
        draggableHandle.gameObject.SetActive(false);

        allFaces = new Dictionary<string, Sprite>();

        Texture2D texture = Resources.Load(StringResources.idle_orb_path) as Texture2D;
        Sprite sprite = Sprite.Create(texture, new Rect(0, 0, texture.width, texture.height), Vector2.zero);
        allFaces.Add("idle", sprite);

        texture = Resources.Load(StringResources.listening_orb_path) as Texture2D;
        sprite = Sprite.Create(texture, new Rect(0, 0, texture.width, texture.height), Vector2.zero);
        allFaces.Add("mic", sprite);

        //Get notification object in orb prefab
        notificatonIcon = transform.GetChild(1).gameObject;
        notificatonIcon.SetActive(false);

        faceColorOuter = faceBG.ColorOuter;
        faceColorInner = faceBG.ColorInner;
    }

    public void SetNotificationIconActive(bool active) => notificatonIcon.SetActive(active);

    public void SetDraggableHandle(bool active) => draggableHandle.gameObject.SetActive(active);

    #region Color Changes
    public void ChangeColorToNotificationActive(bool active)
    {
        if (active)
            faceBG.ColorOuter = faceColorOuterNotification;
        else
            faceBG.ColorOuter = faceColorOuter;
    }

    public void ChangeDragginColorActive(bool active)
    {
        if (active)
            faceBG.ColorInner = Color.black;
        else
            faceBG.ColorInner = faceColorInner;
    }

    public void ChangeColorToDone(bool isDone)
    {
        if (isDone)
            faceBG.ColorOuter = new Color(0,0.5f,0,0.7f);
        else
            faceBG.ColorOuter = faceColorOuter;
    }

    #endregion
}