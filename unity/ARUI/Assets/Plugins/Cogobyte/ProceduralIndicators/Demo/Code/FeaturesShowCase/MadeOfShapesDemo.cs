using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cogobyte.ProceduralIndicators;

namespace Cogobyte.Demo.ProceduralIndicators
{
    //Demo for made of shapes arrow mode in FeaturesShowCase
    //Oscilates the path of arrow along x axis 
    public class MadeOfShapesDemo : MonoBehaviour
    {
        public ArrowObject arrowObject;
        float fi = 0;

        void Update()
        {
            arrowObject.arrowPath.startPoint = new Vector3(-10, 0, -2.5f);
            arrowObject.arrowPath.endPoint = new Vector3(5 + 8f * Mathf.Sin(fi), 0, -2.5f);
            arrowObject.updateArrowMesh();
            fi += 2f * Time.deltaTime;
        }
    }
}