using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class OrbHandle : MonoBehaviour
{
    private Shapes.Polyline _indicator;
    private bool _isActive = false;
    public bool IsActive
    {
        get => _isActive;
        set => _isActive = value;
    }

    void Start()
    {
        _indicator = gameObject.GetComponentInChildren<Shapes.Polyline>();
        _indicator.gameObject.SetActive(false);
    }

    void Update()
    {
        _indicator.gameObject.SetActive(_isActive);
    }

}
