using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.Demo.SmartProceduralIndicators
{

    public class RotateHandle : MonoBehaviour
    {
        public Transform HandleMainTransform;
        public Camera MainCamera;
        Quaternion Offset;
        Vector3 MouseDownPosition;
        public Plane userPlane = new Plane(Vector3.up, Vector3.zero);

        void OnMouseDown()
        {
            Offset = Quaternion.Inverse(Quaternion.LookRotation(HandleMainTransform.forward, GetPoint() - HandleMainTransform.position) * Quaternion.LookRotation(HandleMainTransform.forward, HandleMainTransform.up));

        }
        void OnMouseDrag()
        {
            HandleMainTransform.rotation = Quaternion.Inverse(Offset * Quaternion.LookRotation(HandleMainTransform.forward, GetPoint() - HandleMainTransform.position));
        }

        public Vector3 GetPoint()
        {
            Ray mouseRay = Camera.main.ScreenPointToRay(Input.mousePosition);
            float dist;
            userPlane.SetNormalAndPosition(HandleMainTransform.forward, HandleMainTransform.position);
            if (userPlane.Raycast(mouseRay, out dist))
            {
                Vector3 pos = mouseRay.GetPoint(dist);
                return pos;
            }
            return HandleMainTransform.position + Vector3.up;
        }

    }
}