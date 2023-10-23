using Microsoft.MixedReality.Toolkit.Utilities.Solvers;
using UnityEngine;

public class ObjectIndicator : MonoBehaviour
{
    private GameObject _halo;
    private DirectionalIndicator _directionalSolverPos;
    private Shapes.Disc _haloouter;
    private Shapes.Disc _haloInner;
    private Shapes.Disc _indicator;

    private bool _isFlat = false;
    public bool IsFlat 
    { 
        get { return _isFlat; }
        set { _isFlat = value; } 
    }

    // Start is called before the first frame update
    void Start()
    {
        Shapes.Disc[] discs = GetComponentsInChildren<Shapes.Disc>(true);
        _haloInner = discs[0];
        _haloouter = discs[1];
        _halo = _haloInner.transform.parent.gameObject;

        _directionalSolverPos = GetComponentInChildren<DirectionalIndicator>();
        _indicator = _directionalSolverPos.transform.GetComponentInChildren<Shapes.Disc>();
    }

    // Update is called once per frame
    void Update()
    {
        Vector3 poiToCam = transform.position - AngelARUI.Instance.ARCamera.transform.position;

        float degangle = Vector3.Angle(AngelARUI.Instance.ARCamera.transform.forward, Vector3.Normalize(poiToCam));
        float alpha = Mathf.Max(0.05f,Mathf.Min(1,(1f / 15f) * (degangle-25f)));
        //Debug.Log(degangle + "  " + alpha);
        _indicator.ColorInner = new Color(1, 1, 1, alpha);
        _haloInner.ColorOuter = new Color(1, 1, 1, 1-alpha);
        _haloouter.ColorInner = new Color(1, 1, 1, 1-alpha);

        // on-screen halo faces the user
        if (!_isFlat)
            _halo.transform.rotation = Quaternion.LookRotation(AngelARUI.Instance.ARCamera.transform.position - _halo.transform.position, Vector3.up);
        else
            _halo.transform.rotation = Quaternion.LookRotation(Vector3.up, Vector3.right);

        _directionalSolverPos.transform.rotation = Quaternion.LookRotation(AngelARUI.Instance.ARCamera.transform.position - _halo.transform.position, Vector3.up);

    }

}
