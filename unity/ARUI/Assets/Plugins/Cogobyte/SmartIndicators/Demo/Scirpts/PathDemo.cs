using Cogobyte.SmartProceduralIndicators;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.Demo.SmartProceduralIndicators
{

    public class PathDemo : MonoBehaviour
    {
        public Transform ball;
        public bool traveling = false;
        int currentArrowIndex = 0;
        public SmartArrow currentArrow;
        public List<SmartArrow> arrows;

        void Start()
        {
            ((PointListArrowPath)currentArrow.arrowPath).customPath = new List<Vector3>();
            StartTravel();
        }

        public void StartTravel()
        {
            ((PointListArrowPath)currentArrow.arrowPath).customPath.Clear();
            traveling = true;
        }

        public void EndTravel()
        {
            ((PointListArrowPath)currentArrow.arrowPath).customPath.Clear();
            traveling = false;
        }
        void Update()
        {
            if (Input.GetKeyDown(KeyCode.P))
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
            if (Input.GetKeyDown(KeyCode.U))
            {
                arrows[currentArrowIndex].gameObject.SetActive(false);
                currentArrowIndex = (currentArrowIndex + 1) % arrows.Count;
                currentArrow = arrows[currentArrowIndex];
                arrows[currentArrowIndex].gameObject.SetActive(true);
            }
            if (traveling)
            {
                if (((PointListArrowPath)currentArrow.arrowPath).customPath.Count <= 1 || ((ball.position) - ((PointListArrowPath)currentArrow.arrowPath).customPath[((PointListArrowPath)currentArrow.arrowPath).customPath.Count - 1]).magnitude > 0.05f)
                {
                    ((PointListArrowPath)currentArrow.arrowPath).customPath.Add(ball.position);
                    if (((PointListArrowPath)currentArrow.arrowPath).customPath.Count > 1)
                    {
                        currentArrow.UpdateArrow();
                    }
                }
            }
        }
    }
}