using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    /// <summary>
    /// Arrow path defined by a list of points. 
    /// </summary>
    public class PointListArrowPath : ArrowPath
    {
        /// <summary>
        /// List of points of the path.
        /// </summary>
        public List<Vector3> customPath = new List<Vector3>() { new Vector3(0, 0, 0), new Vector3(1, 0, 0) };
        /// <summary>
        /// Up direction candidate vector for all modes.
        /// </summary>
        public Vector3 upDirection = Vector3.up;
        /// <summary>
        /// Orgin point for up vector for relative to point and relative to line mode.
        /// </summary>
        public Vector3 upReferencePoint = Vector3.zero;
        /// <summary>
        /// Up direction mode. Relative to path starts with up direction and calculates up for each point.
        /// Constant up sets up vector for all points to upDirection or sets up vector of a point to negative last direction if it bends 90 degrees of yaw rotation relative to last direction.
        /// Relative to point sets all up vectors to point and up reference point difference.
        /// Relative to line sets all up vectors to point and closest point on up direction line.
        /// </summary>
        public enum UpDirectionMode {RelativeToPath,ConstantUp,RelativeToPoint,RelativeToLine}
        /// <summary>
        /// Current up direction mode
        /// </summary>
        public UpDirectionMode upDirectionType = UpDirectionMode.RelativeToPath;
        /// <summary>
        /// Number of segments between two points when obstacle check is turned on.
        /// </summary>
        [Range (1,100)]
        public int obstacleCheckLevelOfDetail = 10;

        public override void CalculatePath()
        {
            if (m_myTransform == null) m_myTransform = transform;
            if (customPath.Count < 2) return;
            if (calculatedPath == null) calculatedPath = new List<Vector3>();
            else calculatedPath.Clear();
            if (calculatedRotations == null) calculatedRotations = new List<Quaternion>();
            else calculatedRotations.Clear();
            Vector3 upVector = upDirection;
            Vector3 prevDir = customPath[1] - customPath[0];
            Vector3 pathDiff;
            if (prevDir != Vector3.zero)
            {
                upVector = Quaternion.LookRotation(prevDir, upVector) * Vector3.up;
            }
            for (int i = 0; i < customPath.Count - 1; i++)
            {
                pathDiff = customPath[i + 1] - customPath[i];
                calculatedPath.Add(customPath[i]);
                if (prevDir.magnitude < SmartArrowUtilities.Utilities.errorRate) prevDir = pathDiff;
                upVector = GetNextUpVector(customPath[i], prevDir, pathDiff, upVector, upDirection);
                if (pathDiff != Vector3.zero)
                {
                    calculatedRotations.Add(Quaternion.LookRotation(pathDiff, upVector));
                }
                else
                {
                    calculatedRotations.Add(Quaternion.identity);
                }
                upVector = calculatedRotations[calculatedRotations.Count - 1] * Vector3.up;
                if (obstacleCheck)
                {
                    for (int j = 1; j < obstacleCheckLevelOfDetail; j++)
                    {
                        calculatedPath.Add(customPath[i] + (j * pathDiff) / obstacleCheckLevelOfDetail);
                        calculatedRotations.Add(calculatedRotations[calculatedRotations.Count - 1]);
                    }
                }
                if (pathDiff.magnitude > SmartArrowUtilities.Utilities.errorRate)
                prevDir = pathDiff;
            }
            pathDiff = customPath[customPath.Count - 1] - customPath[customPath.Count - 2];
            calculatedPath.Add(customPath[customPath.Count - 1]);
            calculatedRotations.Add(calculatedRotations[calculatedRotations.Count - 1]);
            ConvertToLocal();
            ObstacleCheck();
            CalculatePathLength();
        }

        /// <summary>
        /// Ensures there are always at least two points in the path based on neighbour points and up vector candidate.
        /// </summary>
        void OnValidate()
        {
            if (customPath.Count < 2)
            {
                customPath = new List<Vector3>() { new Vector3(0, 0, 0), new Vector3(1, 0, 0) };
            }
        }
        /// <summary>
        /// Calculates up vector for a given point from the list.
        /// </summary>
        /// <param name="point">Point that needs an up vector.</param>
        /// <param name="previousDirection">Difference vector between previous point and source point.</param>
        /// <param name="nextDirection">Difference vector between source point and next point.</param>
        /// <param name="previousUpVector">Previous calculated vector by GetNextUpVector function.</param>
        /// <param name="upVectorCandidate">Constant up vector.</param>
        /// <returns>Up vector for source point.</returns>
        public Vector3 GetNextUpVector(Vector3 point,Vector3 previousDirection, Vector3 nextDirection, Vector3 previousUpVector, Vector3 upVectorCandidate)
        {
            if (upDirectionType == UpDirectionMode.RelativeToPath)
            {
                if (Vector3.Angle(previousUpVector, nextDirection) < SmartArrowUtilities.Utilities.errorRate) return -previousDirection;
                if (Vector3.Angle(-previousUpVector, nextDirection) < SmartArrowUtilities.Utilities.errorRate) return previousDirection;
                if (Vector3.Angle(previousDirection,nextDirection) <= 90) return Quaternion.LookRotation(nextDirection, previousUpVector)*Vector3.up;
                else return Quaternion.LookRotation(nextDirection, previousUpVector) * Vector3.down;
            }
            else if (upDirectionType == UpDirectionMode.ConstantUp)
            {
                if (Vector3.Angle(previousUpVector, nextDirection) < SmartArrowUtilities.Utilities.errorRate) return -previousDirection;
                if (Vector3.Angle(-previousUpVector, nextDirection) < SmartArrowUtilities.Utilities.errorRate) return previousDirection;
                return Quaternion.LookRotation(nextDirection, upVectorCandidate) * Vector3.up;
            }
            else if (upDirectionType == UpDirectionMode.RelativeToPoint)
            {
                Vector3 temp = (point - upReferencePoint);
                if (Vector3.Angle(temp, nextDirection) < SmartArrowUtilities.Utilities.errorRate) return -previousDirection;
                if (Vector3.Angle(-temp, nextDirection) < SmartArrowUtilities.Utilities.errorRate) return previousDirection;
                return temp;
            }
            else
            {
                Vector3 temp = Vector3.Cross(upVectorCandidate, Vector3.Cross(point - upReferencePoint, upVectorCandidate));
                if (Vector3.Angle(temp, nextDirection) < SmartArrowUtilities.Utilities.errorRate) return -previousDirection;
                if (Vector3.Angle(-temp, nextDirection) < SmartArrowUtilities.Utilities.errorRate) return previousDirection;
                return temp;
            }
        }
    }
}