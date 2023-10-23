using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using System.Collections;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

public enum DwellButtonType
{
    Toggle=0,
    Select =1
}

/// <summary>
/// Button that can be triggered using touch or eye-gaze dwelling
/// </summary>
public class DwellButton : MonoBehaviour, IMixedRealityTouchHandler
{
    private bool _btnInitialized = false;

    public bool IsInteractingWithBtn = false;
    public float Width => _btnCollider.size.y;

    private bool _isLookingAtBtn = false;
    public bool IsLookingAtBtn => _isLookingAtBtn;

    private bool _isTouchingBtn = false;
    private bool _touchable = false;

    private bool _uniqueObj = false;

    private EyeTarget _target;
    private UnityEvent _selectEvent;
    private UnityEvent _quarterSelectEvent;
    private BoxCollider _btnCollider;
    public BoxCollider Collider => _btnCollider;
    private GameObject _btnmesh;

    private DwellButtonType _type = DwellButtonType.Toggle;
    private bool _toggled = false;
    public bool Toggled { 
        get => _toggled;
        set { 
            _toggled = value; 
            SetSelected(value);
        } 
    }

    private bool _isDisabled = false;
    public bool IsDisabled
    {
        get => _isDisabled;
        set { 
            _isDisabled = value;

            _btnmesh.gameObject.SetActive(!value);

            if (_icon)
            {
                if (value)
                    _icon.color = new Color(0.2f, 0.2f, 0.2f);
                else
                    _icon.color = Color.white;
            }
        }
    }

    //*** Btn Dwelling Feedback 
    private Shapes.Disc _loadingDisc;

    private float _startingAngle;

    //*** Btn Push Feedback
    private Shapes.Disc _pushConfiromationDisc;

    //*** Btn Design
    private Material _btnBGMat;

    private Image _icon;
   
    public void InitializeButton(EyeTarget target, UnityAction btnSelectEvent, UnityAction btnHalfSelect, 
        bool touchable, DwellButtonType type, bool isUnique = false)
    {
        Shapes.Disc[] discs = GetComponentsInChildren<Shapes.Disc>(true);
        _loadingDisc = discs[0];
        _pushConfiromationDisc = discs[1];
        _pushConfiromationDisc.enabled = false;

        _startingAngle = _loadingDisc.AngRadiansStart;

        MeshRenderer mr = GetComponentInChildren<MeshRenderer>();
        _btnBGMat = new Material(mr.material);
        _btnBGMat.color = ARUISettings.BtnBaseColor;
        mr.material = _btnBGMat;

        _icon = transform.GetComponentInChildren<Image>();
        if (_icon)
            _icon.color = Color.white;

        _selectEvent = new UnityEvent();
        _quarterSelectEvent = new UnityEvent();

        _btnCollider = GetComponentInChildren<BoxCollider>(true);
        _btnmesh = transform.GetChild(0).gameObject;

        _uniqueObj = isUnique;
        //TODO: FIGURE OUT HOW TO GET RID OF THIS
        _selectEvent = new UnityEvent();
        _quarterSelectEvent = new UnityEvent();
        this._target = target;
        _selectEvent.AddListener(btnSelectEvent);

        if (btnHalfSelect != null)
            _quarterSelectEvent.AddListener(btnHalfSelect);

        this._touchable = touchable;
        this._type = type;

        if (touchable)
            gameObject.AddComponent<NearInteractionTouchable>();

        _btnInitialized = true;
    }

    private void Update()
    {
        if (!_btnInitialized) return;

        UpdateCurrentlyLooking();
        IsInteractingWithBtn = _isTouchingBtn || _isLookingAtBtn || EyeGazeManager.Instance.CurrentHit.Equals(EyeTarget.textConfirmationWindow);
    }

    private void UpdateCurrentlyLooking()
    {
        if (!_btnInitialized) return;

        bool currentLooking = false;

        if (_uniqueObj)
        {
            currentLooking = EyeGazeManager.Instance.CurrentHitObj != null &&
                    EyeGazeManager.Instance.CurrentHitObj.GetInstanceID() == this.gameObject.GetInstanceID();
        }
        else
        {
            currentLooking = EyeGazeManager.Instance.CurrentHit == _target;
        }

        if (currentLooking && !_isLookingAtBtn && !_isTouchingBtn && !_isDisabled)
        {
            _isLookingAtBtn = true;
            StartCoroutine(Dwelling());
        }  

        if (!currentLooking || _isTouchingBtn)
        {
            _isLookingAtBtn = false;
            StopCoroutine(Dwelling());
            _btnBGMat.color = ARUISettings.BtnBaseColor;
        }

        _isLookingAtBtn = currentLooking;
    }
    
    private IEnumerator Dwelling()
    {
        AudioManager.Instance.PlaySound(transform.position, SoundType.confirmation);

        _btnBGMat.color = ARUISettings.BtnActiveColor;

        bool halfEventEvoked = false;
        bool success = false;
        float duration = 6.24f / ARUISettings.EyeDwellTime; //full circle in radians

        float elapsed = 0f;
        while (!_isTouchingBtn && _isLookingAtBtn && !_isDisabled && elapsed < duration)
        {
            if (CoreServices.InputSystem.EyeGazeProvider.GazeTarget == null)
                break;

            elapsed += Time.deltaTime;
            _loadingDisc.AngRadiansEnd = elapsed * ARUISettings.EyeDwellTime;
            _loadingDisc.Color = Color.white;

            if (!halfEventEvoked && _isLookingAtBtn && _quarterSelectEvent != null && elapsed > (duration / 4))
            {
                halfEventEvoked = true;
                _quarterSelectEvent.Invoke();
            }
                
            if (elapsed>duration && _isLookingAtBtn)
                success = true;

            yield return null;
        }

        if (success)
        {
            _selectEvent.Invoke();
            if (_type == DwellButtonType.Toggle)
            {
                _toggled = !_toggled;
                SetSelected(_toggled);
            } else
            {
                _toggled = false;
                SetSelected(false);
            }
        } else
        {
            _btnBGMat.color = ARUISettings.BtnBaseColor;

            if (_type != DwellButtonType.Toggle || (_type == DwellButtonType.Toggle && !_toggled))
                SetSelected(false);
            else if (_type == DwellButtonType.Toggle && _toggled)
                SetSelected(true);
        }
    }

    /// <summary>
    /// Detect the user touching the button
    /// </summary>
    /// <param name="eventData"></param>
    public void OnTouchStarted(HandTrackingInputEventData eventData)
    {
        if (!_touchable || _isDisabled || !_btnInitialized) return;
        _isTouchingBtn = true;
        _btnBGMat.color = ARUISettings.BtnActiveColor;
        _pushConfiromationDisc.enabled = true;
    }

    public void OnTouchCompleted(HandTrackingInputEventData eventData)
    {
        if (!_touchable || _isDisabled || !_btnInitialized) return;
        _isTouchingBtn = false;

        _btnBGMat.color = ARUISettings.BtnBaseColor;
        _btnmesh.transform.localPosition = Vector3.zero;
        _pushConfiromationDisc.enabled = false;
    }

    public void OnTouchUpdated(HandTrackingInputEventData eventData) 
    {
        if (!_touchable || _isDisabled || !_btnInitialized) return;
        _btnmesh.transform.position = eventData.InputData;

        if (_btnmesh.transform.localPosition.z > _pushConfiromationDisc.transform.localPosition.z)
            _pushConfiromationDisc.Color = Color.cyan;
        else _pushConfiromationDisc.Color = Color.white;

        if (_btnmesh.transform.localPosition.z > _pushConfiromationDisc.transform.localPosition.z+0.01f)
            _selectEvent.Invoke();
    }

    private void SetSelected(bool selected)
    {
        if (!_btnInitialized) return;

        if (selected)
        {
            _loadingDisc.AngRadiansEnd = 6.24f;
            _loadingDisc.Color = ARUISettings.BtnLoadingDiscColor;
            _btnBGMat.color = ARUISettings.BtnActiveColor;
        }
        else
        {
            _loadingDisc.AngRadiansEnd = _startingAngle;
            _loadingDisc.Color = Color.white;
            _btnBGMat.color = ARUISettings.BtnBaseColor;
        }
    }
}