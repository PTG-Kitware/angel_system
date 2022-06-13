using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cogobyte.ProceduralIndicators;

namespace Cogobyte.Demo.ProceduralIndicators
{
    //Calculates the spath path for demo in PrefabExamples
    public class Spath : MonoBehaviour
    {
        public ArrowObject arrowObject;
        Vector3[] points = new Vector3[1000];
        // Use this for initialization
        void Start()
        {
            points = new Vector3[201];
            points[0] = new Vector3(0, 0, 0);
            points[1] = new Vector3(5, 0, 0);
            for (int i = 0; i < 100; i++)
            {
                points[2 + i] = new Vector3(5 + Mathf.Sin(i / 100f * Mathf.PI), 0, i * 1 / 100f);
            }
            points[102] = new Vector3(2.5f, 0, 2f);
            for (int i = 0; i < 97; i++)
            {
                points[103 + i] = new Vector3(1 - Mathf.Sin(i / 97f * Mathf.PI), 0, 4 - i * 1 / 97f);
            }
            points[199] = new Vector3(1f, 0, 4);
            points[200] = new Vector3(33.5f, 0, 4);
            arrowObject.arrowPath.editedPath = new List<Vector3>(points);
            arrowObject.updateArrowMesh();
        }


    }
}