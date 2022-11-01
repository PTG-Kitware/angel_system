using Microsoft.MixedReality.Toolkit.Utilities.Solvers;
using UnityEngine;

public class ObjectIndicator : MonoBehaviour
{
    private GameObject halo;
    private DirectionalIndicator directionalSolverPos;
    private Shapes.Disc haloouter;
    private Shapes.Disc haloInner;
    private Shapes.Disc indicator;

    private bool isFlat = false;

    // Start is called before the first frame update
    void Start()
    {
        Shapes.Disc[] discs = GetComponentsInChildren<Shapes.Disc>(true);
        haloInner = discs[0];
        haloouter = discs[1];
        halo = haloInner.transform.parent.gameObject;

        directionalSolverPos = GetComponentInChildren<DirectionalIndicator>();
        indicator = directionalSolverPos.transform.GetComponentInChildren<Shapes.Disc>();
    }

    // Update is called once per frame
    void Update()
    {
        Vector3 poiToCam = transform.position - AngelARUI.Instance.mainCamera.transform.position;

        float degangle = Vector3.Angle(AngelARUI.Instance.mainCamera.transform.forward, Vector3.Normalize(poiToCam));
        float alpha = Mathf.Max(0.05f,Mathf.Min(1,(1f / 15f) * (degangle-25f)));
        //Debug.Log(degangle + "  " + alpha);
        indicator.ColorInner = new Color(1, 1, 1, alpha);
        haloInner.ColorOuter = new Color(1, 1, 1, 1-alpha);
        haloouter.ColorInner = new Color(1, 1, 1, 1-alpha);

        // on-screen halo faces the user
        if (!isFlat)
            halo.transform.rotation = Quaternion.LookRotation(AngelARUI.Instance.mainCamera.transform.position - halo.transform.position, Vector3.up);
        else
            halo.transform.rotation = Quaternion.LookRotation(Vector3.up, Vector3.right);

        directionalSolverPos.transform.rotation = Quaternion.LookRotation(AngelARUI.Instance.mainCamera.transform.position - halo.transform.position, Vector3.up);

    }

    public void SetFlat(bool isFlat) => this.isFlat = isFlat;
}
