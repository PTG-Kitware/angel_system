using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    /// <summary>
    /// Abstract class for creating different arrow paths for arrow.
    /// Contains generated points of the path with rotations. 
    /// </summary>
    public abstract class ArrowPath : MonoBehaviour
    {
        /// <summary>
        /// Local coordinate system if true. Global if false.
        /// </summary>
        public bool local = false;
        /// <summary>
        /// Check for each point for obstacles if true.
        /// </summary>
        public bool obstacleCheck = false;
        /// <summary>
        /// Objects that get hit by obstacle check ray.
        /// </summary>
        public LayerMask obstacleCheckLayer;
        /// <summary>
        /// Distance from ray hit point when checking for obstacles.
        /// </summary>
        public float distanceFromObstacle = 1;
        /// <summary>
        /// How far does ray get cast from source point.
        /// </summary>
        public float obstacleCheckRayLength = 1;

        /// <summary>
        /// Final generated points of the path.
        /// </summary>
        protected List<Vector3> calculatedPath;
        /// <summary>
        /// Calculated rotations for each generated path point.
        /// </summary>
        protected List<Quaternion> calculatedRotations;
        /// <summary>
        /// Calculated total length of the path.
        /// </summary>
        protected float m_pathLength;
        /// <summary>
        /// Reference to parent transform of this script.
        /// </summary>
        protected Transform m_myTransform;
        /// <summary>
        /// Getter for calculated path points.
        /// </summary>
        /// <returns>Calculated path points.</returns>
        public List<Vector3> GetCalculatedPath()
        {
            return calculatedPath;
        }
        /// <summary>
        /// Getter for calcualted path rotations.
        /// </summary>
        /// <returns>Calculated rotations.</returns>
        public List<Quaternion> GetCalculatedRotation()
        {
            return calculatedRotations;
        }
        /// <summary>
        /// Getter for path length.
        /// </summary>
        /// <returns>The length of the path.</returns>
        public float GetPathLength() {
            return m_pathLength;
        }
        /// <summary>
        /// Generate start points for both ends of the body and tip ends.
        /// </summary>
        /// <param name="smartArrow">Smart arrow that is using this path.</param>
        /// <param name="bodyTipPathData">Target bodyTipPathData object that gets filled with required data.</param>
        public virtual void GetBodyAndTipData(SmartArrow smartArrow,BodyAndTipPathData bodyTipPathData)
        {
            float tailLength = 0;
            if (smartArrow.arrowTail != null) tailLength = smartArrow.arrowTail.GetLength();
            float headLength = 0;
            if (smartArrow.arrowHead != null) headLength = smartArrow.arrowHead.GetLength();
            bool tailFollowsPath = false;
            if (smartArrow.arrowTail != null) tailFollowsPath = smartArrow.arrowTail.FollowsPath();
            bool headFollowsPath = false;
            if (smartArrow.arrowHead != null) headFollowsPath = smartArrow.arrowHead.FollowsPath();
            bodyTipPathData.noRender = false;
            bodyTipPathData.noBody = false;
            bodyTipPathData.calculatedPath = calculatedPath;
            bodyTipPathData.calculatedRotations = calculatedRotations;
            bodyTipPathData.bodyLength = m_pathLength * (smartArrow.EndPercentage - smartArrow.StartPercentage);
            bodyTipPathData.sizeFactor = 1;
            if (calculatedPath.Count < 2 || smartArrow.StartPercentage>=smartArrow.EndPercentage)
            {
                bodyTipPathData.noRender = true;
                return;
            }
            
            bodyTipPathData.pathStartIndex = calculatedPath.Count - 1;
            bodyTipPathData.PathStartPoint = calculatedPath[calculatedPath.Count -1];
            float currentLength = 0;
            for (int i = 1; i < calculatedPath.Count; i++)
            {
                if ((calculatedPath[i] - calculatedPath[i - 1]).magnitude + currentLength > smartArrow.StartPercentage * m_pathLength)
                {
                    bodyTipPathData.PathStartPoint = calculatedPath[i - 1] + (calculatedPath[i] - calculatedPath[i - 1]).normalized * (smartArrow.StartPercentage * m_pathLength - currentLength);
                    bodyTipPathData.pathStartIndex = i;
                    break;
                }
                else
                {
                    currentLength += (calculatedPath[i] - calculatedPath[i - 1]).magnitude;
                }
                if(i== calculatedPath.Count - 1)
                {
                    bodyTipPathData.noRender = true; 
                    return;
                }
            }

            if (tailLength < SmartArrowUtilities.Utilities.errorRate)
            {
                bodyTipPathData.bodyStartPoint = bodyTipPathData.PathStartPoint;
                bodyTipPathData.bodyStartIndex = bodyTipPathData.pathStartIndex;
                bodyTipPathData.bodyStartRotation = calculatedRotations[bodyTipPathData.bodyStartIndex - 1];
            }
            else
            {
                currentLength = 0;
                if (tailFollowsPath)
                {
                    bodyTipPathData.bodyLength -= tailLength;
                    bodyTipPathData.bodyStartPoint = bodyTipPathData.PathStartPoint;
                    for (int i = bodyTipPathData.pathStartIndex; i < calculatedPath.Count; i++)
                    {
                        if ((calculatedPath[i] - bodyTipPathData.bodyStartPoint).magnitude + currentLength > tailLength)
                        {
                            bodyTipPathData.bodyStartPoint = bodyTipPathData.bodyStartPoint + (calculatedPath[i] - bodyTipPathData.bodyStartPoint).normalized * (tailLength - currentLength);
                            bodyTipPathData.bodyStartIndex = i;
                            break;
                        }
                        else
                        {
                            currentLength += (calculatedPath[i] - calculatedPath[i - 1]).magnitude;
                            bodyTipPathData.bodyStartPoint = calculatedPath[i];
                        }
                        if (i == calculatedPath.Count - 1)
                        {
                            bodyTipPathData.noRender = true;
                            return;
                        }
                    }
                    bodyTipPathData.bodyStartRotation = calculatedRotations[bodyTipPathData.bodyStartIndex];
                }
                else
                {
                    bodyTipPathData.bodyStartPoint = bodyTipPathData.PathStartPoint;

                    for (int i = bodyTipPathData.pathStartIndex; i < calculatedPath.Count; i++)
                    {
                        float distance = Vector3.Distance(calculatedPath[i], bodyTipPathData.PathStartPoint);
                        if (distance >= tailLength)
                        {
                            Vector3 intersectPoint = SmartArrowUtilities.Utilities.LineSphereIntersection(bodyTipPathData.PathStartPoint, tailLength, bodyTipPathData.bodyStartPoint, calculatedPath[i])[0];
                            bodyTipPathData.bodyLength -= currentLength + (bodyTipPathData.bodyStartPoint - intersectPoint).magnitude;
                            bodyTipPathData.bodyStartPoint = intersectPoint;
                            bodyTipPathData.bodyStartIndex = i;
                            break;
                        }
                        currentLength += (calculatedPath[i] - bodyTipPathData.bodyStartPoint).magnitude;
                        bodyTipPathData.bodyStartPoint = calculatedPath[i];
                        if (i == calculatedPath.Count - 1)
                        {
                            bodyTipPathData.noBody = true;
                        }
                    }
                    bodyTipPathData.bodyStartRotation = Quaternion.LookRotation(bodyTipPathData.bodyStartPoint - bodyTipPathData.PathStartPoint, calculatedRotations[bodyTipPathData.bodyStartIndex] * Vector3.up);
                }
            }
            currentLength = m_pathLength;
            for (int i = calculatedPath.Count - 2; i >= 0; i--)
            {
                if (currentLength - (calculatedPath[i] - calculatedPath[i + 1]).magnitude < smartArrow.EndPercentage * m_pathLength)
                {
                    bodyTipPathData.PathEndPoint = calculatedPath[i + 1] + (calculatedPath[i] - calculatedPath[i + 1]).normalized * (currentLength - smartArrow.EndPercentage * m_pathLength);
                    bodyTipPathData.pathEndIndex = i;
                    currentLength -= (calculatedPath[i] - calculatedPath[i + 1]).magnitude;
                    break;
                }
                else
                {
                    currentLength -= (calculatedPath[i] - calculatedPath[i + 1]).magnitude;
                }
                if (i == 0)
                {
                    bodyTipPathData.noRender = true;
                    return;
                }
            }


            if (headLength < SmartArrowUtilities.Utilities.errorRate)
            {
                bodyTipPathData.bodyEndPoint = bodyTipPathData.PathEndPoint;
                bodyTipPathData.bodyEndIndex = bodyTipPathData.pathEndIndex;
                bodyTipPathData.bodyEndRotation = calculatedRotations[bodyTipPathData.bodyEndIndex + 1];
            }
            else
            {
                currentLength = 0;
                if (headFollowsPath)
                {
                    bodyTipPathData.bodyLength -= headLength;
                    bodyTipPathData.bodyEndPoint = bodyTipPathData.PathEndPoint;
                    for (int i = bodyTipPathData.pathEndIndex; i >= 0; i--)
                    {
                        if ((calculatedPath[i] - bodyTipPathData.bodyEndPoint).magnitude + currentLength > headLength)
                        {
                            bodyTipPathData.bodyEndPoint = bodyTipPathData.bodyEndPoint + (calculatedPath[i] - bodyTipPathData.bodyEndPoint).normalized * (headLength - currentLength);
                            bodyTipPathData.bodyEndIndex = i;
                            break;
                        }
                        else
                        {
                            currentLength += (calculatedPath[i] - bodyTipPathData.bodyEndPoint).magnitude;
                            bodyTipPathData.bodyEndPoint = calculatedPath[i];
                        }
                        if (i == 0)
                        {
                            bodyTipPathData.noBody = true;
                        }
                    }
                    bodyTipPathData.bodyEndRotation = calculatedRotations[bodyTipPathData.bodyEndIndex];
                }
                else
                {
                    bodyTipPathData.bodyEndPoint = bodyTipPathData.PathEndPoint;
                    for (int i = bodyTipPathData.pathEndIndex; i >= 0; i--)
                    {
                        if (i == bodyTipPathData.bodyStartIndex - 1)
                        {
                            float dist = Vector3.Distance(bodyTipPathData.bodyStartPoint, bodyTipPathData.PathEndPoint);
                            if (dist >= headLength)
                            {
                                Vector3 intersectPoint = SmartArrowUtilities.Utilities.LineSphereIntersection(bodyTipPathData.PathEndPoint, headLength, bodyTipPathData.bodyEndPoint, bodyTipPathData.bodyStartPoint)[0];
                                bodyTipPathData.bodyLength -= currentLength + (bodyTipPathData.bodyEndPoint - intersectPoint).magnitude;
                                bodyTipPathData.bodyEndPoint = intersectPoint;
                                bodyTipPathData.bodyEndIndex = i;
                                break;
                            }
                            bodyTipPathData.noBody = true;
                            break;
                        }
                        float distance = Vector3.Distance(calculatedPath[i], bodyTipPathData.PathEndPoint);
                        if (distance >= headLength)
                        {
                            Vector3 intersectPoint = SmartArrowUtilities.Utilities.LineSphereIntersection(bodyTipPathData.PathEndPoint, headLength, bodyTipPathData.bodyEndPoint, calculatedPath[i])[0];
                            bodyTipPathData.bodyLength -= currentLength + (bodyTipPathData.bodyEndPoint - intersectPoint).magnitude;
                            bodyTipPathData.bodyEndPoint = intersectPoint;
                            bodyTipPathData.bodyEndIndex = i;
                            break;
                        }
                        currentLength += (calculatedPath[i] - bodyTipPathData.bodyEndPoint).magnitude;
                        bodyTipPathData.bodyEndPoint = calculatedPath[i];
                    }
                    if(bodyTipPathData.PathEndPoint - bodyTipPathData.bodyEndPoint != Vector3.zero)
                    {
                        bodyTipPathData.bodyEndRotation = Quaternion.LookRotation(bodyTipPathData.PathEndPoint - bodyTipPathData.bodyEndPoint, calculatedRotations[bodyTipPathData.bodyEndIndex] * Vector3.up);
                    }
                }
            }



            if (bodyTipPathData.noBody)
            {
                bodyTipPathData.bodyStartPoint = bodyTipPathData.bodyEndPoint = bodyTipPathData.PathStartPoint + (bodyTipPathData.PathEndPoint - bodyTipPathData.PathStartPoint) * 1/2;
                if (tailLength + headLength > 0)
                {
                    bodyTipPathData.sizeFactor = (bodyTipPathData.PathEndPoint - bodyTipPathData.PathStartPoint).magnitude / (tailLength + headLength);
                    bodyTipPathData.bodyStartPoint = bodyTipPathData.bodyEndPoint = bodyTipPathData.PathStartPoint + (bodyTipPathData.PathEndPoint - bodyTipPathData.PathStartPoint) * tailLength/ (tailLength + headLength);
                }
                bodyTipPathData.bodyStartIndex = bodyTipPathData.pathStartIndex;
                bodyTipPathData.bodyEndIndex = bodyTipPathData.pathEndIndex;
                bodyTipPathData.bodyEndRotation = bodyTipPathData.bodyStartRotation = Quaternion.LookRotation(bodyTipPathData.PathEndPoint - bodyTipPathData.PathStartPoint, calculatedRotations[bodyTipPathData.bodyStartIndex] * Vector3.up);
            }
            return;
        }
        /// <summary>
        /// Main path generation method. Genrates the calcualted path points and rotations.
        /// </summary>
        public abstract void CalculatePath();
        /// <summary>
        /// Calculate total length of the path.
        /// </summary>
        public virtual void CalculatePathLength()
        {
            m_pathLength = 0;
            for (int i = 1; i < calculatedPath.Count; i++)
            {
                m_pathLength += (calculatedPath[i] - calculatedPath[i - 1]).magnitude;
            }
        }
        /// <summary>
        /// Cast ray from each point and if it hits object with obstacle check layer move the point distance from obstacle length above hit point. 
        /// </summary>
        public virtual void ObstacleCheck()
        {
            if (obstacleCheck) { 
                RaycastHit hit;
                for (int i = 0; i < calculatedPath.Count; i++)
                {
                    if (Physics.Raycast(calculatedPath[i] + calculatedRotations[i] * Vector3.up, calculatedRotations[i] * Vector3.down, out hit, obstacleCheckRayLength, obstacleCheckLayer, QueryTriggerInteraction.Ignore))
                    {
                        calculatedPath[i] = hit.point + calculatedRotations[i] * Vector3.up * distanceFromObstacle;
                    }
                }
            }
        }
        /// <summary>
        /// Converts path to local coordinate system.
        /// </summary>
        public virtual void ConvertToLocal()
        {
            if (local)
            {
                for (int i = 0; i < calculatedPath.Count; i++)
                {
                    calculatedPath[i] = m_myTransform.rotation * (calculatedPath[i] + m_myTransform.position);
                    calculatedRotations[i] = m_myTransform.rotation * calculatedRotations[i];
                }
            }
        }
    }
}