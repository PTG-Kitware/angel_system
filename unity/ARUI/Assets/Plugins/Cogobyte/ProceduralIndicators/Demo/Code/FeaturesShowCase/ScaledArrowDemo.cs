using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cogobyte.ProceduralIndicators;

namespace Cogobyte.Demo.ProceduralIndicators
{
    //Demo for scale function in FeaturesShowCase
    //Oscilates the path of arrow along x axis 
    //Oscilates the scale function length
    public class ScaledArrowDemo : MonoBehaviour
    {
        public ArrowObject arrowObject;
        float fi = 0;

        void Update()
        {
            arrowObject.arrowPath.startPoint = new Vector3(-10, 0, -2.5f);
            arrowObject.arrowPath.endPoint = new Vector3(8f * Mathf.Sin(fi), 0, -2.5f);
            fi += 2f * Time.deltaTime;
            arrowObject.arrowPath.widthFunctionLength = 2 + 4f * Mathf.Sin(fi);
            arrowObject.arrowPath.heightFunctionLength = 2 + 4f * Mathf.Sin(fi);
            arrowObject.updateArrowMesh();
        }
    }
}