using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.Demo.SmartProceduralIndicators
{

    public class DragHandle : MonoBehaviour
    {
        public Transform HandleMainTransform;
        public Camera MainCamera;
        public enum MoveAxis { X, Y, Z, XY, XZ, YZ }
        public MoveAxis moveAxis = MoveAxis.X;
        Vector3 Offset;
        public Plane userPlane = new Plane(Vector3.up, Vector3.zero);

        public float GetSliderValue()
        {
            return 0;
        }

        void OnMouseDown()
        {
            Offset = HandleMainTransform.position - GetPoint();
        }
        void OnMouseDrag()
        {
            HandleMainTransform.position = GetPoint() + Offset;
        }

        public Vector3 GetPoint()
        {
            Ray mouseRay = Camera.main.ScreenPointToRay(Input.mousePosition);
            float dist;
            if (moveAxis == MoveAxis.X || moveAxis == MoveAxis.Z || moveAxis == MoveAxis.XZ)
            {
                userPlane.SetNormalAndPosition(HandleMainTransform.up, HandleMainTransform.position);
                if (userPlane.Raycast(mouseRay, out dist))
                {
                    Vector3 pos = mouseRay.GetPoint(dist);
                    if (moveAxis == MoveAxis.X)
                    {
                        pos = FindNearestPointOnLine(HandleMainTransform.position, HandleMainTransform.right, pos);
                    }
                    if (moveAxis == MoveAxis.Z)
                    {
                        pos = FindNearestPointOnLine(HandleMainTransform.position, HandleMainTransform.forward, pos);
                    }
                    return pos;
                }
            }
            else if (moveAxis == MoveAxis.Y || moveAxis == MoveAxis.YZ)
            {
                userPlane.SetNormalAndPosition(HandleMainTransform.right, HandleMainTransform.position);
                if (userPlane.Raycast(mouseRay, out dist))
                {
                    Vector3 pos = mouseRay.GetPoint(dist);
                    if (moveAxis == MoveAxis.Y)
                    {
                        pos = FindNearestPointOnLine(HandleMainTransform.position, HandleMainTransform.up, pos);
                    }
                    if (moveAxis == MoveAxis.Z)
                    {
                        pos = FindNearestPointOnLine(HandleMainTransform.position, HandleMainTransform.forward, pos);
                    }
                    return pos;
                }
            }
            else
            {
                userPlane.SetNormalAndPosition(HandleMainTransform.forward, HandleMainTransform.position);
                if (userPlane.Raycast(mouseRay, out dist))
                {
                    Vector3 pos = mouseRay.GetPoint(dist);
                    return pos;
                }
            }
            return HandleMainTransform.position;
        }

        Vector3 FindNearestPointOnLine(Vector3 a, Vector3 b, Vector3 p)
        {
            b = a + b;
            return a + Vector3.Project(p - a, b - a);
        }

    }
}