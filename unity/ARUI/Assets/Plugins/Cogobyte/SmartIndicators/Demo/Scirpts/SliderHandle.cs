using Cogobyte.SmartProceduralIndicators;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.Demo.SmartProceduralIndicators
{

    public class SliderHandle : MonoBehaviour
    {
        public Transform HandleMainTransform;
        public Camera MainCamera;
        Plane userPlane;
        public float maxLength = 10;

        public float GetSliderValue()
        {
            return (HandleMainTransform.position - transform.position).magnitude / maxLength;
        }

        void OnMouseDrag()
        {
            transform.position = GetPoint();
        }


        public Vector3 GetPoint()
        {
            Ray mouseRay = Camera.main.ScreenPointToRay(Input.mousePosition);
            float dist;
            userPlane.SetNormalAndPosition(HandleMainTransform.right, HandleMainTransform.position);
            if (userPlane.Raycast(mouseRay, out dist))
            {
                Vector3 pos = mouseRay.GetPoint(dist);
                pos = GetClosestPointOnFiniteLine(pos, HandleMainTransform.position, HandleMainTransform.position + HandleMainTransform.forward * maxLength);
                return pos;
            }
            return transform.position;
        }

        Vector3 GetClosestPointOnFiniteLine(Vector3 point, Vector3 line_start, Vector3 line_end)
        {
            Vector3 line_direction = line_end - line_start;
            float line_length = line_direction.magnitude;
            line_direction.Normalize();
            float project_length = Mathf.Clamp(Vector3.Dot(point - line_start, line_direction), 0f, line_length);
            return line_start + line_direction * project_length;
        }

    }

}