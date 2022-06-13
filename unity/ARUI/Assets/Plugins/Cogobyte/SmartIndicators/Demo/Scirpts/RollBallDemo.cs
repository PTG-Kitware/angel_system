using Cogobyte.SmartProceduralIndicators;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.Demo.SmartProceduralIndicators
{

    public class RollBallDemo : MonoBehaviour
    {
        public Transform cameraObj;
        public Transform ball;
        public SmartArrow TrailArrow;
        public SmartArrow DistanceArrow;
        public float speed = 20f;
        public bool traveling = false;

        int currentArrow = 0;
        public List<SmartArrow> arrows;

        void Start()
        {
            ((PointListArrowPath)TrailArrow.arrowPath).customPath = new List<Vector3>();
            StartTravel();
        }

        public void StartTravel()
        {
            ((PointListArrowPath)TrailArrow.arrowPath).customPath.Clear();
            traveling = true;
        }

        public void EndTravel()
        {
            ((PointListArrowPath)TrailArrow.arrowPath).customPath.Clear();
            traveling = false;
        }
        void Update()
        {
            //WSAD movement
            if (Input.GetKey(KeyCode.D))
            {
                ball.transform.position += new Vector3(Time.deltaTime * speed, 0, 0);
            }
            if (Input.GetKey(KeyCode.S))
            {
                ball.transform.position -= new Vector3(0, 0, Time.deltaTime * speed);
            }
            if (Input.GetKey(KeyCode.W))
            {
                ball.transform.position += new Vector3(0, 0, Time.deltaTime * speed);
            }
            if (Input.GetKey(KeyCode.A))
            {
                ball.transform.position -= new Vector3(Time.deltaTime * speed, 0, 0);
            }
            if (Input.GetKey(KeyCode.Space))
            {
                ball.transform.position += new Vector3(0, Time.deltaTime * speed, 0);
            }
            if (Input.GetKey(KeyCode.C))
            {
                ball.transform.position -= new Vector3(0, Time.deltaTime * speed, 0);
            }
            if (Input.GetKeyDown(KeyCode.T))
            {
                //Activate/Deactivate travel arrow
                if (traveling)
                {
                    EndTravel();
                }
                else
                {
                    StartTravel();
                }
            }
            if (Input.GetKeyDown(KeyCode.G))
            {
                arrows[currentArrow].gameObject.SetActive(false);
                currentArrow = (currentArrow + 1) % arrows.Count;
                TrailArrow = arrows[currentArrow];
                arrows[currentArrow].gameObject.SetActive(true);
            }

            cameraObj.position = ball.position + new Vector3(0, 6, -10);
            //tra.arrowPath.endPoint = -me.position + ball.position + new Vector3(0, 1f, 0);
            //tra.updateArrowMesh();


            if (traveling)
            {
                if (((PointListArrowPath)TrailArrow.arrowPath).customPath.Count <= 1 || ((ball.position + new Vector3(0, 1, 0)) - ((PointListArrowPath)TrailArrow.arrowPath).customPath[((PointListArrowPath)TrailArrow.arrowPath).customPath.Count - 1]).magnitude > 0.5f)
                {

                    ((PointListArrowPath)TrailArrow.arrowPath).customPath.Add(ball.position + new Vector3(0, 1, 0));
                    if (((PointListArrowPath)TrailArrow.arrowPath).customPath.Count > 1)
                    {
                        TrailArrow.UpdateArrow();
                    }
                }
            }

        }
    }
}