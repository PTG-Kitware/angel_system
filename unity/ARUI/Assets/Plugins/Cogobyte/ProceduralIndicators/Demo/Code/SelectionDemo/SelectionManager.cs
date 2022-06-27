using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cogobyte.ProceduralIndicators;
using Cogobyte.ProceduralLibrary;

namespace Cogobyte.Demo.ProceduralIndicators
{
    public class SelectionManager : MonoBehaviour
    {
        //all indicators used
        public ArrowObject projectedArrowToGround;
        public ArrowObject arrowIndicator;
        public SelectionIndicator hoverIndicator;
        public SelectionIndicator selectedIndicator;
        public SelectionIndicator destinationIndicator;
        //Currently selected object
        public GameObject selectedObject;

        void Start()
        {
            ((projectedArrowToGround.arrowPath.pathArray) as CustomPathArray).customPath[0] = new Vector3(0, 0, 0);
            ((projectedArrowToGround.arrowPath.pathArray) as CustomPathArray).customPath[1] = new Vector3(10, 0, 0);
        }

        void Update()
        {
            RaycastHit hit;
            //Reset indicators
            destinationIndicator.hideIndicator = true;
            arrowIndicator.hideIndicator = true;
            arrowIndicator.updateArrowMesh();
            if (Input.GetMouseButtonDown(0) || Input.GetMouseButton(1) || Input.GetMouseButtonUp(1))
            {
                Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
                //Left mouse to select objects
                if (Input.GetMouseButtonDown(0))
                {
                    if (Physics.Raycast(ray, out hit, 1000.0f))
                        if (hit.collider.gameObject.GetComponent<SelectedObjectOptions>() != null)
                        {
                            //selecting another object changes the currently selected object
                            selectedObject = hit.collider.gameObject;
                            selectedIndicator.hideIndicator = false;
                            selectedIndicator.pathArray = hit.collider.gameObject.GetComponent<SelectedObjectOptions>().myPath;
                            destinationIndicator.pathArray = hit.collider.gameObject.GetComponent<SelectedObjectOptions>().destinationPathArray;
                            arrowIndicator.arrowPath.startPoint = hit.collider.gameObject.transform.position;
                        }
                        else
                        {
                            //Ground deselects
                            selectedIndicator.hideIndicator = true;
                            selectedObject = null;
                        }
                }
                if (Input.GetMouseButton(1) && selectedObject != null)
                {
                    //Right mouse shows the arrow from start object to desitination and draws the destination outline
                    if (Physics.Raycast(ray, out hit, 1000.0f))
                    {
                        arrowIndicator.hideIndicator = false;
                        destinationIndicator.hideIndicator = false;
                        arrowIndicator.arrowPath.startPoint = selectedObject.transform.position;
                        arrowIndicator.arrowPath.endPoint = hit.point;
                        destinationIndicator.pathArray.translation = hit.point;
                        arrowIndicator.updateArrowMesh();
                        destinationIndicator.updateIndicatorMesh();
                    }
                }
                if (Input.GetMouseButtonUp(1) && selectedObject != null)
                {
                    //Right mouse shows the arrow from start object to desitination and draws the destination outline
                    if (Physics.Raycast(ray, out hit, 1000.0f))
                    {
                        ((projectedArrowToGround.arrowPath.pathArray) as CustomPathArray).customPath[0] = selectedObject.transform.position;
                        selectedObject.transform.position = hit.point;
                        ((projectedArrowToGround.arrowPath.pathArray) as CustomPathArray).customPath[1] = selectedObject.transform.position;
                    }
                }

            }
            else
            {
                //hovering over object has another currently hovered selection (broken path selection)
                Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
                if (Physics.Raycast(ray, out hit, 1000.0f))
                    if (hit.collider.gameObject.GetComponent<SelectedObjectOptions>() != null)
                    {
                        hoverIndicator.hideIndicator = false;
                        hoverIndicator.pathArray = hit.collider.gameObject.GetComponent<SelectedObjectOptions>().myPath;
                    }
                    else
                    {
                        hoverIndicator.hideIndicator = true;
                    }
            }
        }
    }
}