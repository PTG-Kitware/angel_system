using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

/// <summary>
/// Represents a scene label in 3D space
/// </summary>
public class DetectedEntity : Entity
{
    private TextMeshPro labelMesh;
    private ObjectIndicator halo;

    private Vector3 haloOffset = new Vector3(0, 0.01f ,0);

    void Awake()
    {
        id = gameObject.name;

        GameObject labelTextObj = new GameObject();
        labelTextObj.transform.parent = transform;
        labelTextObj.name = "Label";

        labelMesh = labelTextObj.AddComponent<TextMeshPro>();
        labelMesh.rectTransform.sizeDelta = new Vector2(1f, 0.2f);
        labelMesh.fontSize = 0.5f;
        labelMesh.alignment = TextAlignmentOptions.Center;
        labelMesh.fontMaterial.SetFloat(ShaderUtilities.ID_FaceDilate, 0.20f);
        labelMesh.outlineColor = new Color(0.55f, 0.55f, 0.55f, 1.0f);
        labelMesh.outlineWidth = 0.20f;

        GameObject haloObj = Instantiate(Resources.Load(StringResources.POIHalo_path)) as GameObject;
        haloObj.transform.parent = transform;
        haloObj.transform.position = transform.position;
        halo = haloObj.GetComponent<ObjectIndicator>();
        halo.gameObject.SetActive(false);

        halo.transform.position = halo.transform.position + haloOffset;
    }

    public void InitEntity(string id, Vector3 position, string text, bool showDetectedObj)
    {
        entityType = Type.Actual;
        id = text; //TODO: set actual id

        SetText(text);
        SetTextLabelOn(showDetectedObj);

        transform.position = position;
    }

    public void SetText(string text)  {
        this.label = text;

        gameObject.name = text;
        labelMesh.text = text; 
        UpdateFontSize(); 
    }

    void Update()
    {
        labelMesh.transform.rotation = Quaternion.LookRotation(transform.position- Camera.main.transform.position, Vector3.up);
        UpdateFontSize();
    }

    private void UpdateFontSize()
    {
        float currentDist = Mathf.Abs((transform.position - Camera.main.transform.position).magnitude);
        labelMesh.fontSize = Mathf.Max(Mathf.Min(Mathf.Log(currentDist,2) / 3f, UISettings.minMaxFontsize[1]), UISettings.minMaxFontsize[0]);
    }

    public void SetTextLabelOn(bool on)
    {
       if (on)
            labelMesh.text = label;
       else
            labelMesh.text = "";
    }

    public void SetHaloOn(bool on, bool isFlat)
    {
        halo.SetFlat(isFlat);
        halo.gameObject.SetActive(on);

    }

}
