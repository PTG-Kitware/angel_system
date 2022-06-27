using Cogobyte.SmartProceduralIndicators;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.Demo.SmartProceduralIndicators
{

    public class PointDemo : MonoBehaviour
    {
        public SmartArrow TrailArrow;
        public bool drawing = false;

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
        }

        public void EndTravel()
        {
            ((PointListArrowPath)TrailArrow.arrowPath).customPath.Clear();
        }

        void Update()
        {

            if (Input.GetKeyDown(KeyCode.T))
            {
                //Activate/Deactivate travel arrow
                if (drawing)
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

            RaycastHit hit;
            if (Input.GetMouseButton(0) || Input.GetMouseButtonUp(1))
            {
                Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
                if (Physics.Raycast(ray, out hit, 1000.0f))
                {
                    //selecting another object changes the currently selected object
                    if (((PointListArrowPath)TrailArrow.arrowPath).customPath.Count < 1 || (hit.point - ((PointListArrowPath)TrailArrow.arrowPath).customPath[((PointListArrowPath)TrailArrow.arrowPath).customPath.Count - 1]).magnitude > 0.1f)
                    {
                        ((PointListArrowPath)TrailArrow.arrowPath).customPath.Add(hit.point);
                        if (((PointListArrowPath)TrailArrow.arrowPath).customPath.Count > 1)
                        {
                            TrailArrow.UpdateArrow();
                        }
                    }
                }
            }
        }
    }
}