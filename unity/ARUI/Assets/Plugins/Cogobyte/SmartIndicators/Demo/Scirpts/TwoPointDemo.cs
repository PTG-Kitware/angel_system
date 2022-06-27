using Cogobyte.SmartProceduralIndicators;
using UnityEngine;

namespace Cogobyte.Demo.SmartProceduralIndicators
{
    public class TwoPointDemo : MonoBehaviour
    {
        public SmartArrow smartArrow;
        public Transform pointAHandle;
        public Transform pointBHandle;
        public Transform upDirectionHandle;
        public float positionBounds = 100;

        // Update is called once per frame
        void Update()
        {
            if (pointAHandle.position.magnitude > positionBounds) pointAHandle.position = pointAHandle.position.normalized * positionBounds;
            if (pointBHandle.position.magnitude > positionBounds) pointBHandle.position = pointBHandle.position.normalized * positionBounds;
            ((PointToPointArrowPath)smartArrow.arrowPath).pointA = pointAHandle.position;
            ((PointToPointArrowPath)smartArrow.arrowPath).pointB = pointBHandle.position;
            ((PointToPointArrowPath)smartArrow.arrowPath).upDirection = upDirectionHandle.rotation * Vector3.up;
            upDirectionHandle.position = (pointAHandle.position + pointBHandle.position) / 2;
            upDirectionHandle.rotation = Quaternion.LookRotation(pointBHandle.position - pointAHandle.position, ((PointToPointArrowPath)smartArrow.arrowPath).upDirection);
            smartArrow.UpdateArrow();
        }
    }
}
