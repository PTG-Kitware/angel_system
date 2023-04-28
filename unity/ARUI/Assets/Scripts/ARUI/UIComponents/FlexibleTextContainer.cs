using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class FlexibleTextContainer : MonoBehaviour
{
    public TMPro.TextMeshProUGUI TextComponent;

    private int maxCharCountPerLine = 70;
    public int MaxCharCountPerLine { get { return maxCharCountPerLine; } }

    //***Flexible Textbox for taskmessage
    private RectTransform HGroupTaskMessage;
    public Rect TextRect { get { return HGroupTaskMessage.rect; } }

    private RectTransform textRect;

    private Material taskBackgroundMat;

    //***Flexible Textbox for taskmessage
    private BoxCollider taskMessageCollider;
    public BoxCollider MessageCollider
    {
        get { return taskMessageCollider; }
    }

    // Start is called before the first frame update
    void Awake()
    {
        HorizontalLayoutGroup temp = gameObject.GetComponentInChildren<HorizontalLayoutGroup>();
        //init task message group
        HGroupTaskMessage = temp.gameObject.GetComponent<RectTransform>();
        TMPro.TextMeshProUGUI[] allText = HGroupTaskMessage.gameObject.GetComponentsInChildren<TMPro.TextMeshProUGUI>();

        TextComponent = allText[0];
        TextComponent.text = "";
        textRect = TextComponent.gameObject.GetComponent<RectTransform>();

        //Init background image
        Image bkgr = HGroupTaskMessage.GetComponentInChildren<Image>();
        taskBackgroundMat = new Material(bkgr.material);
        Color firstColor = GetGlowColor();
        taskBackgroundMat.SetColor("_InnerGlowColor", firstColor);
        bkgr.material = taskBackgroundMat;

        taskMessageCollider = transform.GetComponent<BoxCollider>();
    }

    public TMPro.TextMeshProUGUI[] GetAllTextMeshComponents()
    {
        return HGroupTaskMessage.gameObject.GetComponentsInChildren<TMPro.TextMeshProUGUI>();
    }

    // Update is called once per frame
    void Update()
    {
        // Update collider of messagebox
        taskMessageCollider.size = new Vector3(HGroupTaskMessage.rect.width, taskMessageCollider.size.y, taskMessageCollider.size.z);
        taskMessageCollider.center = new Vector3(HGroupTaskMessage.rect.width / 2, 0, 0);
    }

    public void SetBackgroundColor(Color activeColorBG)
    {
        taskBackgroundMat.color = activeColorBG;
    }

    public void SetTextColor(Color textColor)
    {
        TextComponent.color = textColor;
    }

    public void SetGlowColor(Color glowColor)
    {
        taskBackgroundMat.SetColor("_InnerGlowColor", glowColor);
    }

    public Color GetGlowColor() => taskBackgroundMat.GetColor("_InnerGlowColor");

    public void UpdateAnchorInstant()
    {
        taskMessageCollider.center = new Vector3(HGroupTaskMessage.rect.width / 2, 0, 0);
        taskMessageCollider.size = new Vector3(HGroupTaskMessage.rect.width, taskMessageCollider.size.y, taskMessageCollider.size.z);
    }

    internal void SetText(string message)
    {
        this.TextComponent.text = Utils.SplitTextIntoLines(message, maxCharCountPerLine);
    }
}
