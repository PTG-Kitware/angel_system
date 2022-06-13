using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cogobyte.ProceduralIndicators;

namespace Cogobyte.Demo.ProceduralIndicators
{
    //Demo for shape function in FeaturesShowCase
    //Oscilates the path of arrow along x axis 
    //Oscialtes the shape function length
    public class ShapeChangeDemo : MonoBehaviour
    {
        public ArrowObject arrowObject;
        float fi = 0;

        void Update()
        {
            arrowObject.arrowPath.startPoint = new Vector3(-10, 0, -7);
            arrowObject.arrowPath.endPoint = new Vector3(8f * Mathf.Sin(fi), 0, -7);
            fi += 2f * Time.deltaTime;
            arrowObject.arrowPath.shapeFunctionLength = 6 + 5 * Mathf.Cos(fi);
            arrowObject.updateArrowMesh();
        }
    }
}