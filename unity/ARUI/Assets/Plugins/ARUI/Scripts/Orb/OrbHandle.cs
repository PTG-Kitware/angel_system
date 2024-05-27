using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class OrbHandle : MonoBehaviour
{
    private Shapes.Triangle _indicator;
    private bool _isActive = false;

    private float _fixingProgress = 0.0f;

    public void SetHandleProgress(float progress)
    {
        _indicator.ColorA = Color.white * progress;
        _indicator.ColorB = Color.white * progress;
        _indicator.ColorC = Color.white * progress;
        _fixingProgress = progress;
    }

    public void Start()
    {
        _indicator = gameObject.GetComponentInChildren<Shapes.Triangle>();
        _indicator.gameObject.SetActive(false);
    }

    public void Update()
    {
        _isActive = Orb.Instance.OrbBehavior == MovementBehavior.Fixed || _fixingProgress > 0.0f;
        _indicator.gameObject.SetActive(_isActive);
    }

}
