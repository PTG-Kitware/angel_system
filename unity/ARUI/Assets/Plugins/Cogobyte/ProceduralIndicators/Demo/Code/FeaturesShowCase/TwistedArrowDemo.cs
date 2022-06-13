using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cogobyte.ProceduralIndicators;

namespace Cogobyte.Demo.ProceduralIndicators
{
    //Demo for rotation function in FeaturesShowCase
    //Changes rotation function length over time
    public class TwistedArrowDemo : MonoBehaviour
    {
        public ArrowObject arrowObject;
        float fi = 0;

        void Update()
        {
            arrowObject.arrowPath.startPoint = new Vector3(-10, 0, -5);
            arrowObject.arrowPath.endPoint = new Vector3(8f * Mathf.Sin(fi), 0, -5);
            fi += 2f * Time.deltaTime;
            arrowObject.arrowPath.rotationFunctionLength = 0.5f + 0.5f * Mathf.Sin(fi);
            arrowObject.updateArrowMesh();
        }
    }
}