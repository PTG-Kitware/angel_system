using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cogobyte.ProceduralIndicators;

namespace Cogobyte.Demo.ProceduralIndicators
{
    //Stores the path travelled by ball and draws an arrow along that path
    public class TraveledPath : MonoBehaviour
    {
        public ArrowObject arrowObject;
        public Transform ball;
        public bool traveling = false;

        void Start()
        {
            arrowObject.arrowPath.editedPath = new List<Vector3>();
            arrowObject.hideIndicator = true;
            StartTravel();
        }

        public void StartTravel()
        {
            arrowObject.arrowPath.editedPath.Clear();
            traveling = true;
        }

        public void EndTravel()
        {
            arrowObject.arrowPath.editedPath.Clear();
            arrowObject.hideIndicator = true;
            traveling = false;
        }

        // Update is called once per frame
        void Update()
        {
            if (traveling)
            {
                if (arrowObject.arrowPath.editedPath.Count == 0 || ((ball.position + new Vector3(0, 1, 0)) - arrowObject.arrowPath.editedPath[arrowObject.arrowPath.editedPath.Count - 1]).magnitude > 0.2f)
                {
                    arrowObject.arrowPath.editedPath.Add(ball.position + new Vector3(0, 1, 0));
                    arrowObject.hideIndicator = false;
                    arrowObject.updateArrowMesh();
                }
            }
        }
    }
}