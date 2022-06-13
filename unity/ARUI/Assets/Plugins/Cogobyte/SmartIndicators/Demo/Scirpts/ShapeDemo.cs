using Cogobyte.SmartProceduralIndicators;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;


namespace Cogobyte.Demo.SmartProceduralIndicators
{

    public class ShapeDemo : MonoBehaviour
    {
        public SmartArrow shapeArrow;
        public float displacement = 0;
        public float colorDisplacement = 0;
        public float colorDisplacementSpeed = 10;
        public float displacementSpeed = 2;
        // Start is called before the first frame update
        void Start()
        {

        }

        // Update is called once per frame
        void Update()
        {
            displacement += Time.deltaTime * displacementSpeed;
            colorDisplacement += Time.deltaTime * colorDisplacementSpeed;
            shapeArrow.bodyRenderers[0].displacement = displacement;
            shapeArrow.bodyRenderers[0].colorDisplacement = colorDisplacement;
            shapeArrow.UpdateArrow();
        }
    }
}