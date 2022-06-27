using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cogobyte.ProceduralIndicators;

namespace Cogobyte.Demo.ProceduralIndicators
{
    //Demo for broken arrow mode in FeaturesShowCase
    //Oscilates the path of arrow along x axis 
    namespace Cogobyte.Demo.ProceduralIndicators
    {
        public class BrokenArrowDemo : MonoBehaviour
        {
            public ArrowObject arrowObject;
            float fi = 0;

            void Update()
            {
                arrowObject.arrowPath.startPoint = new Vector3(-10, 0, 0);
                arrowObject.arrowPath.endPoint = new Vector3(8f * Mathf.Sin(fi), 0, 0);
                fi += 2f * Time.deltaTime;
                arrowObject.updateArrowMesh();
            }
        }
    }
}