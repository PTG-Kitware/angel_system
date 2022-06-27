using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cogobyte.ProceduralIndicators;

namespace Cogobyte.Demo.ProceduralIndicators
{
    //Demo for extrude arrow mode in FeaturesShowCase
    //Circles around and oscilates the length of arrow
    public class TrackObjectDemo : MonoBehaviour
    {
        public ArrowObject arrowObject;
        float fi = 0;
        float omega = 0;
        public float distance = 7.1f;
        public float distanceOscilation = 5f;
        public float rotationSpeed = 1f;
        public float distanceChangeSpeed = 5f;

        void Update()
        {
            fi += rotationSpeed * Time.deltaTime;
            omega += distanceChangeSpeed * Time.deltaTime;
            arrowObject.arrowPath.startPoint = new Vector3(0, 0, 0);
            arrowObject.arrowPath.endPoint = new Vector3((Mathf.Sin(omega) * distanceOscilation + distance) * Mathf.Sin(fi), 0, (Mathf.Sin(omega) * distanceOscilation + distance) * Mathf.Cos(fi));
            arrowObject.updateArrowMesh();
        }
    }
}