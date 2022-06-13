using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cogobyte.ProceduralIndicators;

namespace Cogobyte.Demo.ProceduralIndicators
{
    //Demo for broken arrow mode in FeaturesShowCase
    //Oscilates the path of arrow along x axis 
    public class PointPathDemo : MonoBehaviour
    {
        public ArrowObject arrowObject;
        int firstPoint = 0;
        int lastPoint = 30;
        Vector3[] points = new Vector3[1000];

        void Start()
        {
            for (int i = 0; i < 100; i++)
            {
                points[i] = new Vector3(10 - i / 10f, 0, -3);
            }
            float fi = 0;
            for (int i = 100; i < 900; i++)
            {
                points[i] = new Vector3(Mathf.Sin(fi), (i - 100) / 200f, -4 + Mathf.Cos(fi));
                fi -= 0.05f;
            }
        }

        void Update()
        {
            int t = 0;
            for (int i = firstPoint; i < lastPoint; i++)
            {
                arrowObject.arrowPath.editedPath[t++] = points[i];
            }
            firstPoint += 2;
            lastPoint += 2;
            if (lastPoint >= 900)
            {
                firstPoint = 0;
                lastPoint = 30;
            }
            arrowObject.updateArrowMesh();
        }
    }
}