using System;
using UnityEngine;
using UnityEngine.UI;

public class FlexibleTextContainer : MonoBehaviour
{
    private TMPro.TextMeshProUGUI _textComponent;

    public TMPro.TextMeshProUGUI[] AllTextMeshComponents => _HGroupTaskMessage.gameObject.GetComponentsInChildren<TMPro.TextMeshProUGUI>();

    //*** Flexible Textbox for taskmessage
    private RectTransform _HGroupTaskMessage;
    public Rect TextRect => _HGroupTaskMessage.rect;
    public Transform VGroup => _HGroupTaskMessage.parent;

    //*** Collider of taskmessage
    private BoxCollider _taskMessageCollider;
    public BoxCollider MessageCollider => _taskMessageCollider;

    private Material _taskBackgroundMat;

    private Color32 _innerGlowColorStartup;
    public Color32 GlowColor
    {
        get => _taskBackgroundMat.GetColor("_InnerGlowColor");
        set { _taskBackgroundMat.SetColor("_InnerGlowColor", value); }
    }

    public Color TextColor
    {
        get => _textComponent.color;
        set { _textComponent.color = value; }
    }

    public Color BackgroundColor
    {
        get => _taskBackgroundMat.color;
        set { _taskBackgroundMat.color = value; }
    }

    public string Text
    {
        get => _textComponent.text;
        set { _textComponent.text = Utils.SplitTextIntoLines(value, ARUISettings.OrbMessageMaxCharCountPerLine); }
    }

    public float TextSize
    {
        set { _textComponent.fontSize = value; }
    }

    private Color _isLookingatTextColor = new Color(0.1f, 0.1f, 0.1f);
    public bool IsLookingAtText
    {
        set
        {
            if (value)
            {
                GlowColor = _isLookingatTextColor;
            } else
            {
                GlowColor = _innerGlowColorStartup;
            }
        }
    }

    // Start is called before the first frame update
    void Awake()
    {
        HorizontalLayoutGroup temp = gameObject.GetComponentInChildren<HorizontalLayoutGroup>();
        //init task message group
        _HGroupTaskMessage = temp.gameObject.GetComponent<RectTransform>();
        TMPro.TextMeshProUGUI[] allText = _HGroupTaskMessage.gameObject.GetComponentsInChildren<TMPro.TextMeshProUGUI>();

        _textComponent = allText[0];
        _textComponent.text = "";

        //Init background image
        Image bkgr = _HGroupTaskMessage.GetComponentInChildren<Image>();
        _taskBackgroundMat = new Material(bkgr.material);
        _innerGlowColorStartup = GlowColor;
        _taskBackgroundMat.SetColor("_InnerGlowColor", _innerGlowColorStartup);
        bkgr.material = _taskBackgroundMat;

        _taskMessageCollider = transform.GetComponent<BoxCollider>();

    }

    //public void AddVMNC() => gameObject.AddComponent<VMNonControllable>();

    /// <summary>
    /// Update collider of messagebox based on the how much space the text takes
    /// </summary>
    void Update()
    {
        _taskMessageCollider.size = new Vector3(_HGroupTaskMessage.rect.width, _taskMessageCollider.size.y, _taskMessageCollider.size.z);
        _taskMessageCollider.center = new Vector3(_HGroupTaskMessage.rect.width / 2, 0, 0);
    }

    /// <summary>
    /// Update the anchor of the flexible text container based on the available space in the user's FOV
    /// </summary>
    public void UpdateAnchorInstant()
    {
        _taskMessageCollider.center = new Vector3(_HGroupTaskMessage.rect.width / 2, 0, 0);
        _taskMessageCollider.size = new Vector3(_HGroupTaskMessage.rect.width, _taskMessageCollider.size.y, _taskMessageCollider.size.z);
    }

}
