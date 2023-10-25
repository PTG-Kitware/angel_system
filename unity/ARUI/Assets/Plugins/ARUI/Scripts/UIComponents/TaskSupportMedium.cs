using Shapes;
using UnityEngine;

public class TaskSupportMedium : MonoBehaviour
{
    private Disc _ringindicator;
    private VMImage _medium;

    private bool _mediumIsActive = false;
    // Start is called before the first frame update
    void Start()
    {
        transform.SetLayerAllChildren(StringResources.LayerToInt(StringResources.UI_layer));

        _medium = transform.GetChild(0).gameObject.GetComponentInChildren<VMImage>();
        _medium.gameObject.SetActive(false);

        _ringindicator = transform.GetChild(1).GetComponent<Disc>();
    }

    // Update is called once per frame
    void Update()
    {
        if (AngelARUI.Instance.ARCamera == null) return;

        transform.LookAt(AngelARUI.Instance.ARCamera.transform.position);

        if (!_mediumIsActive && 
            (EyeGazeManager.Instance.CurrentHit==EyeTarget.ringindicator 
             && EyeGazeManager.Instance.CurrentHitObj.GetInstanceID()==_ringindicator.gameObject.GetInstanceID()) )
        {
            _mediumIsActive = true;
            _medium.gameObject.SetActive(true);
        }

        if (_mediumIsActive && 
            !(EyeGazeManager.Instance.CurrentHit == EyeTarget.ringindicator
             && EyeGazeManager.Instance.CurrentHitObj.GetInstanceID() == _ringindicator.gameObject.GetInstanceID()))
        {
            _mediumIsActive = false;
            _medium.gameObject.SetActive(false);
            _medium.transform.localPosition = Vector3.zero;
        }
    }
}
