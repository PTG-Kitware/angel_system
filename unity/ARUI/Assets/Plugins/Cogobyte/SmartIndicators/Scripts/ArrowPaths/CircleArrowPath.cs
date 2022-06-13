using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    /// <summary>
    /// Arrow path defined by a circle.
    /// </summary>
    public class CircleArrowPath : ArrowPath
    {
        /// <summary>
        /// Center of the circle.
        /// </summary>
        public Vector3 center = Vector3.zero;
        /// <summary>
        /// Radius of the circle is the radius and center difference.
        /// </summary>
        public Vector3 radius = Vector3.up;
        /// <summary>
        /// Axis vector around which circle circulates.
        /// </summary>
        public Vector3 axis = Vector3.back;
        /// <summary>
        /// Number of circle segments.
        /// </summary>
        [Range (3,100)]
        public int levelOfDetail = 3;
        /// <summary>
        /// Offset angle from the radius position.
        /// </summary>
        public float startAngle = 0;
        /// <summary>
        /// Draw circle angle length around axis.
        /// </summary>
        public float endAngle = 360;
        /// <summary>
        /// Roll of the path.
        /// </summary>
        public float upDirectionRollAngle = 0;

        public override void CalculatePath()
        {
            if (m_myTransform == null) m_myTransform = transform;
            if (calculatedPath == null) calculatedPath = new List<Vector3>();
            else calculatedPath.Clear();
            if (calculatedRotations == null) calculatedRotations = new List<Quaternion>();
            else calculatedRotations.Clear();
            for (int i = 0; i <= levelOfDetail; i++)
            {
                calculatedPath.Add(center + Quaternion.AngleAxis(startAngle + i * (endAngle-startAngle)/levelOfDetail,axis) * radius);

            }
            for (int i = 0; i < calculatedPath.Count - 1; i++)
            {
                if(calculatedPath[i + 1] - calculatedPath[i] !=Vector3.zero)
                {
                    calculatedRotations.Add(Quaternion.LookRotation(calculatedPath[i + 1] - calculatedPath[i], Quaternion.AngleAxis(upDirectionRollAngle, calculatedPath[i + 1] - calculatedPath[i]) * (calculatedPath[i] - center)));
                }
                else
                {
                    calculatedRotations.Add(Quaternion.identity);
                }
            }
            calculatedRotations.Add(calculatedRotations[calculatedRotations.Count-1]);
            ConvertToLocal();
            ObstacleCheck();
            CalculatePathLength();
        }
    }
}