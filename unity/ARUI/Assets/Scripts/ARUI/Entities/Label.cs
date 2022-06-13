using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

/// <summary>
/// Represents a scene label in 3D space
/// </summary>
public class Label : MonoBehaviour
{
    private TextMeshPro labelText;

    // Start is called before the first frame update
    void Awake()
    {
        labelText = gameObject.AddComponent<TextMeshPro>();
        labelText.rectTransform.sizeDelta = new Vector2(1f, 0.2f);
        labelText.fontSize = 0.5f;
        labelText.alignment = TextAlignmentOptions.Center;
        labelText.fontMaterial.SetFloat(ShaderUtilities.ID_FaceDilate, 0.20f);
        labelText.outlineColor = new Color(0.55f, 0.55f, 0.55f, 1.0f);
        labelText.outlineWidth = 0.20f;
    }

    public void SetText(string text)  {
        gameObject.name = text;
        labelText.text = text; 
        UpdateFontSize(); 
    }

    void Update()
    {
        transform.rotation = Quaternion.LookRotation(transform.position- Camera.main.transform.position, Vector3.up);
        UpdateFontSize();
    }

    private void UpdateFontSize()
    {
        float currentDist = Mathf.Abs((transform.position - Camera.main.transform.position).magnitude);
        labelText.fontSize = Mathf.Max(Mathf.Min(Mathf.Log(currentDist,2) / 3f, UISettings.minMaxFontsize[1]), UISettings.minMaxFontsize[0]);
    }
}
