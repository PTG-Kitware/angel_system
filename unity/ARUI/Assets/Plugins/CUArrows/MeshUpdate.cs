using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cogobyte.ProceduralIndicators;

public class MeshUpdate : MonoBehaviour
{
    // Start is called before the first frame update
    public ArrowObject arrowObject;

    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {

        arrowObject.updateArrowMesh();
    }
}
