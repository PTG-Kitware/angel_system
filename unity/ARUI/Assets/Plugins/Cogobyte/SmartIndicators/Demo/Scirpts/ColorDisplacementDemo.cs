using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cogobyte.SmartProceduralIndicators;

namespace Cogobyte.Demo.SmartProceduralIndicators
{

    public class ColorDisplacementDemo : MonoBehaviour
    {
        public float dispSpeed = 1;
        public SmartArrow s;
        // Start is called before the first frame update
        void Start()
        {

        }

        // Update is called once per frame
        void Update()
        {
            s.displacement += Time.deltaTime * dispSpeed;
            s.UpdateArrow();
        }
    }
}