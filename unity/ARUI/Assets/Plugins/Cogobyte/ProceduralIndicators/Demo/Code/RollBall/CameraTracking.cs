using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.Demo.ProceduralIndicators
{
    //Camera will track ball object movement
    public class CameraTracking : MonoBehaviour
    {
        public Transform cameraObj;
        public Transform ball;

        void Start()
        {
        }

        void Update()
        {
            cameraObj.position = ball.position + new Vector3(0, 6, -10);
        }
    }
}