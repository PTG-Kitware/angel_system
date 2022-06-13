using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    /// <summary>
    /// Arrow path defined by two points and up direction.
    /// </summary>
    public class PointToPointArrowPath : ArrowPath
    {
        /// <summary>
        /// First point.
        /// </summary>
        public Vector3 pointA = Vector3.zero;
        /// <summary>
        /// Second point.
        /// </summary>
        public Vector3 pointB = Vector3.forward * 10;
        /// <summary>
        /// Path up orientation is defined by vector from pointA to upDirection.
        /// </summary>
        public Vector3 upDirection = Vector3.up;
        /// <summary>
        /// Number of path segments used when obstacle check is turned on.
        /// </summary>
        [Range (1,100)]
        public int obstacleCheckLevelOfDetail = 10;
        
        public override void CalculatePath()
        {
            if (m_myTransform == null) m_myTransform = transform;
            if (calculatedPath == null) calculatedPath = new List<Vector3>();
            else calculatedPath.Clear();
            if (calculatedRotations == null) calculatedRotations = new List<Quaternion>();
            else calculatedRotations.Clear();
            m_pathLength = (pointB - pointA).magnitude;
            Quaternion rot = Quaternion.LookRotation(pointB - pointA, upDirection);
            calculatedPath.Add(pointA);
            calculatedRotations.Add(rot);
            if (obstacleCheck)
            {
                Vector3 temp = pointB - pointA;
                for (int i = 1; i < obstacleCheckLevelOfDetail; i++)
                {
                    calculatedPath.Add(pointA + (i*temp)/obstacleCheckLevelOfDetail);
                    calculatedRotations.Add(rot);
                }
            }
            calculatedRotations.Add(rot);
            calculatedPath.Add(pointB);
            ConvertToLocal();
            ObstacleCheck();
            CalculatePathLength();
        }
    }
}