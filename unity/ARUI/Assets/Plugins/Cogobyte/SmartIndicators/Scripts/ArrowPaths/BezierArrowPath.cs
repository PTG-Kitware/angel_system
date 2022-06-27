using Cogobyte.SmartProceduralIndicators.SmartArrowUtilities;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    /// <summary>
    /// Arrow path defined by bezier splines.
    /// </summary>
    public class BezierArrowPath : ArrowPath
    {
        /// <summary>
        /// Container for all bezier points.
        /// </summary>
        public BezierSpline bezierSpline = new BezierSpline();
        /// <summary>
        /// Number of segments for each spline.
        /// </summary>
        public int levelOfDetail = 10;
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
        /// Define each point uses in reference up and out reference up for rotation generations.
        /// </summary>
        public enum UpDirectionMode { RelativeToPath, DefineEachPoint, ConstantUp, RelativeToPoint, RelativeToLine }
        /// <summary>
        /// Current up direction mode
        /// </summary>
        public UpDirectionMode upDirectionType = UpDirectionMode.RelativeToPath;
        public override void CalculatePath()
        {
            if (m_myTransform == null) m_myTransform = transform;
            if (calculatedPath == null) calculatedPath = new List<Vector3>();
            else calculatedPath.Clear();
            if (calculatedRotations == null) calculatedRotations = new List<Quaternion>();
            else calculatedRotations.Clear();

            Vector3 upVector = upDirection;
            Vector3 pathDiff = Vector3.up;
            Vector3 nextPoint = bezierSpline.GetPoint(0, 0);
            Vector3 prevDir = bezierSpline.GetPoint(0, 1f / levelOfDetail) - nextPoint;
            for (int i = 0; i < bezierSpline.points.Count-1; i++)
            {
                for (int j = 1; j <= levelOfDetail; j++)
                {
                    calculatedPath.Add(nextPoint);
                    nextPoint = bezierSpline.GetPoint(i, j * 1f / levelOfDetail);
                    pathDiff = nextPoint - calculatedPath[calculatedPath.Count-1];
                    if (upDirectionType == UpDirectionMode.DefineEachPoint)
                    {
                        calculatedRotations.Add(Quaternion.LookRotation(pathDiff, nextPoint + Vector3.Lerp(bezierSpline.points[i].outReferenceUp - nextPoint, bezierSpline.points[i+1].inReferenceUp - nextPoint, (j-1) * 1f / levelOfDetail)));
                    }
                    else
                    {
                        upVector = GetNextUpVector(calculatedPath[calculatedPath.Count - 1], prevDir, pathDiff, upVector, upDirection);
                        calculatedRotations.Add(Quaternion.LookRotation(pathDiff, upVector));
                        upVector = calculatedRotations[calculatedRotations.Count - 1] * Vector3.up;
                        prevDir = pathDiff;
                    }
                }
            }
            calculatedPath.Add(nextPoint);
            if (upDirectionType != UpDirectionMode.DefineEachPoint)
            calculatedRotations.Add(calculatedRotations[calculatedRotations.Count - 1]);
            else calculatedRotations.Add(Quaternion.LookRotation(pathDiff, bezierSpline.points[bezierSpline.points.Count - 1].inReferenceUp - nextPoint));
            ConvertToLocal();
            ObstacleCheck();
            CalculatePathLength();
        }
        /// <summary>
        /// Ensures minimum two points and minimum level of detail of 2.
        /// 
        /// </summary>
        void OnValidate()
        {
            levelOfDetail = Mathf.Max(levelOfDetail, 2);
            if (bezierSpline.points.Count < 2) bezierSpline.Reset();
        }
        /// <summary>
        /// Calculates up vector for a given point from the list.
        /// </summary>
        /// <param name="point">Point that needs an up vector.</param>
        /// <param name="previousDirection">Difference vector between previous point and source point.</param>
        /// <param name="nextDirection">Difference vector between source point and next point.</param>
        /// <param name="previousUpVector">Previous point up vector.</param>
        /// <param name="upVectorCandidate">Candidate up vector.</param>
        /// <returns>Up vector for source point.</returns>
        public Vector3 GetNextUpVector(Vector3 point, Vector3 previousDirection, Vector3 nextDirection, Vector3 previousUpVector, Vector3 upVectorCandidate)
        {
            if (upDirectionType == UpDirectionMode.RelativeToPath)
            {
                if (Vector3.Angle(previousUpVector, nextDirection) < SmartArrowUtilities.Utilities.errorRate) return -previousDirection;
                if (Vector3.Angle(-previousUpVector, nextDirection) < SmartArrowUtilities.Utilities.errorRate) return previousDirection;
                if (Vector3.Angle(previousDirection, nextDirection) <= 90) return Quaternion.LookRotation(nextDirection, previousUpVector) * Vector3.up;
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
        /// <summary>
        /// Container class for all splines.
        /// </summary>
        [System.Serializable]
        public class BezierSpline
        {
            /// <summary>
            /// List of bezier points that make the curves.
            /// </summary>
            [SerializeField]
            public List<BezierSplinePoint> points = new List<BezierSplinePoint>();
            public BezierSpline()
            {
                Reset();
            }
            /// <summary>
            /// Getter for control point mode at index.
            /// </summary>
            /// <param name="index">Index of control point.</param>
            /// <returns>Mode of control point at index.</returns>
            public BezierControlPointMode GetControlPointMode(int index)
            {
                return points[index].controlPointMode;
            }
            /// <summary>
            /// Sets control point mode and enforces it
            /// </summary>
            /// <param name="index">Point to set mode.</param>
            /// <param name="mode">Bezier control point mode to enforce.</param>
            public void SetControlPointMode(int index, BezierControlPointMode mode)
            {
                points[index].controlPointMode = mode;
                points[index].MoveInTangentControlPoint(points[index].outTangent);
            }
            /// <summary>
            /// Getter of curve points count.
            /// </summary>
            public int CurveCount
            {
                get
                {
                    return points.Count - 1;
                }
            }
            /// <summary>
            /// Gets point on curve for percentage t.
            /// </summary>
            /// <param name="i">Start curve point.</param>
            /// <param name="t">Percentage at curve.</param>
            /// <returns></returns>
            public Vector3 GetPoint(int i , float t)
            {
                return Bezier.GetPoint(points[i].position, points[i].outTangent, points[i + 1].inTangent, points[i + 1].position, t);
            }
            /// <summary>
            /// Removes a curve point at index.
            /// </summary>
            /// <param name="index">Curve point to delete</param>
            public void DeleteCurve(int index)
            {
                points.RemoveAt(index);
            }
            /// <summary>
            /// //Splits one curve into two in the middle
            /// </summary>
            /// <param name="index">Curve start point to split.</param>
            public void SplitCurve(int index)
            {
                BezierSplinePoint p = new BezierSplinePoint();
                BezierSplinePoint secondPoint = (index != points.Count - 1) ? points[index + 1] : points[0];
                p.position = Bezier.GetPoint(points[index].position, points[index].outTangent, secondPoint.inTangent, secondPoint.position, 0.5f);
                p.inTangent = p.position - Bezier.GetFirstDerivative(points[index].position, points[index].outTangent, secondPoint.inTangent, secondPoint.position, 0.5f).normalized;
                p.outTangent = p.position + Bezier.GetFirstDerivative(points[index].position, points[index].outTangent, secondPoint.inTangent, secondPoint.position, 0.5f).normalized;
                p.controlPointMode = BezierControlPointMode.Free;
                points.Insert(index + 1, p);
            }
            /// <summary>
            /// Adds a next curve
            /// </summary>
            public void AddCurve()
            {
                points.Add(new BezierSplinePoint());
                points[points.Count - 1].position = points[points.Count - 2].position + new Vector3(1, 0, 0);
                points[points.Count - 1].inTangent = points[points.Count - 1].position + new Vector3(-0.5f, 0, 0);
                points[points.Count - 1].outTangent = points[points.Count - 1].position + new Vector3(0.5f, 0, 0);
                points[points.Count - 1].controlPointMode = BezierControlPointMode.Free;
            }
            /// <summary>
            /// Resets to the default curve
            /// </summary>
            public void Reset()
            {
                points = new List<BezierSplinePoint>();
                points.Add(new BezierSplinePoint());
                points[0].position = new Vector3(0, 0, 0);
                points[0].inTangent = new Vector3(-0.5f, 0, 0);
                points[0].outTangent = new Vector3(0.5f, 0, 0);
                points[0].controlPointMode = BezierControlPointMode.Free;
                points.Add(new BezierSplinePoint());
                points[1].position = new Vector3(1, 0, 0);
                points[1].inTangent = new Vector3(0.5f, 0, 0);
                points[1].outTangent = new Vector3(1.5f, 0, 0);
                points[1].controlPointMode = BezierControlPointMode.Free;
            }
        }
        /// <summary>
        /// A bezier point containing two control points for previous and next curve and previous and next rotation candidate.
        /// </summary>
        [System.Serializable]
        public class BezierSplinePoint
        {
            /// <summary>
            /// Bezier point position.
            /// </summary>
            public Vector3 position;
            /// <summary>
            /// Entry tangent of spline composed of previous point and this point.
            /// </summary>
            public Vector3 inTangent;
            /// <summary>
            /// Start tangent of this point.
            /// </summary>
            public Vector3 outTangent;
            /// <summary>
            /// End reference up from previous point to this point.
            /// </summary>
            public Vector3 inReferenceUp;
            /// <summary>
            /// Start reference up from this point.
            /// </summary>
            public Vector3 outReferenceUp;
            /// <summary>
            /// Current control point mode.
            /// </summary>
            public BezierControlPointMode controlPointMode;
            /// <summary>
            /// Move the point to destination and move tangents and up references with it.
            /// </summary>
            /// <param name="destination"></param>
            public void MovePoint(Vector3 destination)
            {
                destination = SmartArrowUtilities.Utilities.RoundVector(destination);
                inTangent += destination - position;
                outTangent += destination - position;
                inReferenceUp += destination - position;
                outReferenceUp += destination - position;
                position = destination;
            }
            /// <summary>
            /// Enforces other control point if one is moved.
            /// </summary>
            /// <param name="destination">Where to move the main point.</param>
            public void MoveInTangentControlPoint(Vector3 destination)
            {
                destination = SmartArrowUtilities.Utilities.RoundVector(destination);
                inTangent = destination;
                if (controlPointMode == BezierControlPointMode.Free) return;
                Vector3 enforcedTangent = position - inTangent;
                if (controlPointMode == BezierControlPointMode.Aligned)
                {
                    enforcedTangent = enforcedTangent.normalized * Vector3.Distance(position, outTangent);
                }
                outTangent = position + enforcedTangent;
            }
            /// <summary>
            /// Enforces other control point if one is moved.
            /// </summary>
            /// <param name="destination">Where to move the point.</param>
            public void MoveOutTangentControlPoint(Vector3 destination)
            {
                destination = SmartArrowUtilities.Utilities.RoundVector(destination);
                outTangent = destination;
                if (controlPointMode == BezierControlPointMode.Free) return;
                Vector3 enforcedTangent = position - outTangent;
                if (controlPointMode == BezierControlPointMode.Aligned)
                {
                    enforcedTangent = enforcedTangent.normalized * Vector3.Distance(position, inTangent);
                }
                inTangent = position + enforcedTangent;
            }
        }
        /// <summary>
        /// Bezier control point modes. Free doesnt adjust control points. Aligned keeps the same in and out tangent direction. Mirrored keeps the same in and out direction and magnitude.
        /// </summary>
        public enum BezierControlPointMode
        {
            Free,
            Aligned,
            Mirrored
        }
    }
}