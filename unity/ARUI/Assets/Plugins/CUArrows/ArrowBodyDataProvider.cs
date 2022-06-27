// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using UnityEngine;

namespace Microsoft.MixedReality.Toolkit.Utilities
{
    /// <summary>
    /// Creates a spline based on control points.
    /// </summary>
    public class ArrowBodyDataProvider : BaseMixedRealityLineDataProvider
    {
        [Tooltip("List of positions and orientations that define control points to generate the spline")]
        [SerializeField]
        public MixedRealityPose[] controlPoints =
        {
            MixedRealityPose.ZeroIdentity,
            new MixedRealityPose(new Vector3(1.0f, 0.5f, 0f), Quaternion.identity),
            new MixedRealityPose(new Vector3(0.0f, -0.5f, 0f), Quaternion.identity),
            new MixedRealityPose(new Vector3(0.5f, 0.0f, 0f), Quaternion.identity),
        };

        /// <summary>
        /// List of positions and orientations that define control points to generate the spline
        /// </summary>
        private MixedRealityPose[] ControlPoints => controlPoints;

        #region BaseMixedRealityLineDataProvider Implementation

        /// <inheritdoc />
        public override int PointCount => controlPoints.Length;

        /// <inheritdoc />
        protected override Vector3 GetPointInternal(float normalizedDistance)
        {
            var totalDistance = normalizedDistance * (PointCount - 1);
            int point1Index = Mathf.FloorToInt(totalDistance);
            point1Index -= point1Index % 3;
            float subDistance = (totalDistance - point1Index) / 3;

            int point2Index;
            int point3Index;
            int point4Index;

            if (!Loops)
            {
                if (point1Index + 3 >= PointCount)
                {
                    return controlPoints[PointCount - 1].Position;
                }

                if (point1Index < 0)
                {
                    return controlPoints[0].Position;
                }

                point2Index = point1Index + 1;
                point3Index = point1Index + 2;
                point4Index = point1Index + 3;
            }
            else
            {
                point2Index = (point1Index + 1) % (PointCount - 1);
                point3Index = (point1Index + 2) % (PointCount - 1);
                point4Index = (point1Index + 3) % (PointCount - 1);
            }

            Vector3 point1 = controlPoints[point1Index].Position;
            Vector3 point2 = controlPoints[point2Index].Position;
            Vector3 point3 = controlPoints[point3Index].Position;
            Vector3 point4 = controlPoints[point4Index].Position;

            return LineUtility.InterpolateBezierPoints(point1, point2, point3, point4, subDistance);
        }

        /// <inheritdoc />
        protected override Vector3 GetPointInternal(int pointIndex)
        {
            if (pointIndex < 0 || pointIndex >= controlPoints.Length)
            {
                Debug.LogError("Invalid point index!");
                return Vector3.zero;
            }

            if (Loops && pointIndex == PointCount - 1)
            {
                controlPoints[pointIndex] = controlPoints[0];
                pointIndex = 0;
            }

            return controlPoints[pointIndex].Position;
        }

        /// <inheritdoc />
        protected override void SetPointInternal(int pointIndex, Vector3 point)
        {
            if (pointIndex < 0 || pointIndex >= controlPoints.Length)
            {
                Debug.LogError("Invalid point index!");
                return;
            }

            if (Loops && pointIndex == PointCount - 1)
            {
                controlPoints[pointIndex] = controlPoints[0];
                pointIndex = 0;
            }
            controlPoints[pointIndex].Position = point;
        }


        /// <inheritdoc />
        protected override float GetUnClampedWorldLengthInternal()
        {
            float distance = 0f;
            Vector3 last = GetPoint(0f);

            for (int i = 1; i < BaseMixedRealityLineDataProvider.UnclampedWorldLengthSearchSteps; i++)
            {
                Vector3 current = GetPoint((float)i / BaseMixedRealityLineDataProvider.UnclampedWorldLengthSearchSteps);
                distance += Vector3.Distance(last, current);
            }

            return distance;
        }

        #endregion BaseMixedRealityLineDataProvider Implementation
    }
}
