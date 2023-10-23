using Microsoft.MixedReality.Toolkit;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class VMHands : VMNonControllable
{
    // Update is called once per frame
    void Start()
    {
        collider = gameObject.AddComponent<BoxCollider>();

        base.Start();

        Rigidbody rb = gameObject.AddComponent<Rigidbody>();
        rb.isKinematic = true;
        rb.useGravity = false;

        collider.isTrigger = true;
    }

    void OnTriggerEnter(Collider other) { }

    void OnTriggerExit(Collider other) { }
}
