using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cogobyte.ProceduralIndicators;

namespace Cogobyte.Demo.ProceduralIndicators
{
    //Demo for extrude arrow mode in FeaturesShowCase
    //Oscilates the path of arrow endpoints along z axis 
    public class TrackTwoObjectsDemo : MonoBehaviour
    {
        public ArrowObject arrowObject;
        float fi = 0;

        void Update()
        {
            fi += 1f * Time.deltaTime;
            arrowObject.arrowPath.startPoint = new Vector3(-10, 0, +10 * Mathf.Sin(fi));
            arrowObject.arrowPath.endPoint = new Vector3(10, 0, -10 * Mathf.Sin(fi));
            arrowObject.updateArrowMesh();
        }
    }
}