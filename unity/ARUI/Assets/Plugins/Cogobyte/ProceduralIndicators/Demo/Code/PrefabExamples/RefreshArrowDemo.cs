using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cogobyte.ProceduralIndicators;

namespace Cogobyte.Demo.ProceduralIndicators
{
    //Creates the halfcircle arrow for refresh arrow demo
    public class RefreshArrowDemo : MonoBehaviour
    {
        public ArrowObject arrowObject;
        public float radius;
        public float rotationAngle;
        // Use this for initialization
        void Start()
        {
            Vector3[] points = new Vector3[500];
            for (int i = 0; i < 500; i++)
            {
                points[i] = new Vector3(radius * Mathf.Sin(i / 500f * 0.95f * Mathf.PI), 0, radius * Mathf.Cos(i / 500f * 0.95f * Mathf.PI));
            }
            arrowObject.arrowPath.arrowPathType = ArrowPath.ArrowPathType.PointArray;
            arrowObject.arrowPath.editedPath = new List<Vector3>(points);
            arrowObject.arrowPath.rotationFunctionLength = 1;
            AnimationCurve curve = AnimationCurve.Linear(0f, rotationAngle, 1f, rotationAngle);
            arrowObject.arrowPath.rotateFunction = curve;
            arrowObject.updateArrowMesh();
        }

        // Update is called once per frame
        void Update()
        {

        }
    }
}