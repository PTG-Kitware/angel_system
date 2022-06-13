using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cogobyte.ProceduralIndicators;

namespace Cogobyte.Demo.ProceduralIndicators
{
    //Demo for rotation function in FeaturesShowCase
    //Changes rotation function over time as a linear function from fi to fi
    public class CurvePathDemo : MonoBehaviour
    {
        public ArrowObject arrowObject;
        float fi = 0;

        void Update()
        {
            fi += 0.1f * Time.deltaTime;
            Keyframe[] k = new Keyframe[2] { new Keyframe(0, fi), new Keyframe(1, fi) };
            arrowObject.arrowPath.rotateFunction.keys = k;
            arrowObject.updateArrowMesh();
        }
    }
}