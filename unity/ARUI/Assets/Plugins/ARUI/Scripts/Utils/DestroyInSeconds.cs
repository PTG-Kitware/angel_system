using Microsoft.MixedReality.Toolkit;
using System;
using System.Collections;
using UnityEngine;

public class DestroyInSeconds : MonoBehaviour
{
    void Start() => StartCoroutine(LateDestroy());

    private IEnumerator LateDestroy()
    {
        yield return new WaitForSeconds(0.2f);

        Destroy(gameObject);
    }
}