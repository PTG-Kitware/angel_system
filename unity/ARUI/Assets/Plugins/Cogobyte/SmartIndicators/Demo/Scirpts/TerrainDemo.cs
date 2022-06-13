using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.Demo.SmartProceduralIndicators
{

    public class TerrainDemo : MonoBehaviour
    {
        public Transform positionHandle;
        public Transform terrainPosition;
        // Start is called before the first frame update
        void Start()
        {

        }

        // Update is called once per frame
        void Update()
        {
            terrainPosition.position = positionHandle.position - new Vector3(50, 12, 50);
        }
    }
}