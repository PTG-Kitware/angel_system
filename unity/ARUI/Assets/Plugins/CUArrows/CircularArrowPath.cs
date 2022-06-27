using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cogobyte.ProceduralIndicators;

namespace Cogobyte.Demo.ProceduralIndicators
{
    //Demo for broken arrow mode in FeaturesShowCase
    //Oscilates the path of arrow along x axis 
    public class CircularArrowPath : MonoBehaviour
    {
        public ArrowObject arrowObject;
        int firstPoint = 0;
        int lastPoint = 40;
        Vector3[] points = new Vector3[1000];

        void Start()
        {
            float fi = 0;
            for (int i = 0; i < 720; i++)
            {
                points[i] = new Vector3(Mathf.Sin(fi), 0, Mathf.Cos(fi));
                fi += 0.04f;
            }
        }

        void Update()
        {
            int t = 0;
            for (int i = firstPoint; i < lastPoint; i++)
            {
                arrowObject.arrowPath.editedPath[t++] = points[i];
            }
            firstPoint += 1;
            lastPoint += 1;
            if (lastPoint >= 720)
            {
                firstPoint = 0;
                lastPoint = 40;
            }
            arrowObject.updateArrowMesh();
        }
    }
}