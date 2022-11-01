// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Microsoft.MixedReality.Toolkit.Utilities.Solvers;
using UnityEngine;
using Microsoft.MixedReality.Toolkit;
using System.Collections;
using System;

/// <summary>
/// Provides a solver for the Orb, using MRTK solver
/// </summary>
public class OrbFollowerSolver : Solver
{
    [Tooltip("Min distance from eye to position element around, i.e. the sphere radius")]
    private float minDistance = 0.6f;
    
    [Tooltip("Max distance from eye to element")]
    private float maxDistance = 1f;

    [Tooltip("The element will stay at least this far away from the center of view")]
    private float minViewDegrees = 0f;

    [Tooltip("The element will stay at least this close to the center of view")]
    private float currentMaxViewDegrees = 12f;
    private float maxViewDegreesRegular = 12f;
    private float maxViewDegreesSticky = 21f;
    private float maxViewDegreesCenter = 5f;

    private float regularMoveLerp = 0.7f;

    private bool paused = false;
    public bool IsPaused
    {
        get { return paused; }
        set { paused = value; }
    }

    private bool stayCenter = false;
    private bool isSticky = false;

    private bool isLookingAtOrbFlag = false;
    private bool lazyEyeDisableProcessing = false;

    private bool outOfFOV = false;
    public bool IsOutOfFOV
    {
        get { return outOfFOV; }
    }

    /// <summary>
    /// Position to the view direction, or the movement direction, or the direction of the view cone.
    /// </summary>
    private Vector3 SolverReferenceDirection => SolverHandler.TransformTarget != null ? SolverHandler.TransformTarget.forward : Vector3.forward;

    private Vector3 ReferencePoint => SolverHandler.TransformTarget != null ? SolverHandler.TransformTarget.position : Vector3.zero;

    private new void Start()
    {
        base.Start();

        MoveLerpTime = regularMoveLerp;
        RotateLerpTime = 0.1f;
        Smoothing = true;
    }

    private void LateUpdate()
    {
        if (isLookingAtOrbFlag && !Orb.Instance.IsLookingAtOrb && !lazyEyeDisableProcessing)
            StartCoroutine(LazyDisableIsLooking());
        else if (!isLookingAtOrbFlag && Orb.Instance.IsLookingAtOrb)
            isLookingAtOrbFlag = true;

    }

    public override void SolverUpdate()
    {
        if (!(paused || isLookingAtOrbFlag))
        {
            Vector3 goalPosition = WorkingPosition;

            //update maxDistance based on spatial map
            float dist = Utils.GetCameraToPosDist(transform.position);
            if (dist != -1)
                maxDistance = Mathf.Max(minDistance, Mathf.Min(dist - 0.05f, 1.0f));

            GetDesiredPos(ref goalPosition);
            GoalPosition = goalPosition;
            transform.rotation = Quaternion.LookRotation(goalPosition - ReferencePoint, Vector3.up);
        }
        else //only update rotation 
        {
            GoalPosition = transform.position;
            transform.rotation = Quaternion.LookRotation(transform.position - ReferencePoint, Vector3.up);
            outOfFOV = false;
        }
          
    }

    /// <summary>
    /// Position update for solver
    /// </summary>
    /// <param name="desiredPos">the reference to the position the orb should be</param>
    /// <returns></returns>
    private void GetDesiredPos(ref Vector3 desiredPos)
    {
        // Determine reference locations and directions
        Vector3 direction = SolverReferenceDirection;
        Vector3 elementPoint = transform.position;
        Vector3 elementDelta = elementPoint - ReferencePoint;
        float elementDist = elementDelta.magnitude;
        Vector3 elementDir = elementDist > 0 ? elementDelta / elementDist : Vector3.one;

        // Generate basis: First get axis perpendicular to reference direction pointing toward element
        Vector3 perpendicularDirection = (elementDir - direction);
        perpendicularDirection -= direction * Vector3.Dot(perpendicularDirection, direction);
        perpendicularDirection.Normalize();

        // Calculate the clamping angles, accounting for aspect (need the angle relative to view plane)
        float heightToViewAngle = Vector3.Angle(perpendicularDirection, Vector3.up);

        //Apply a different clamp to vertical FOV than horizontal. Vertical = Horizontal * aspectV
        float aspectV = 0.7f;
        float verticalAspectScale = Mathf.Lerp(aspectV, 1f, Mathf.Abs(Mathf.Sin(heightToViewAngle * Mathf.Deg2Rad)));

        float currentAngle = Vector3.Angle(elementDir, direction);
        // Calculate the current angle
        float currentAngleClamped = Mathf.Clamp(currentAngle, minViewDegrees, currentMaxViewDegrees * verticalAspectScale);

        if (isSticky || stayCenter)
            currentAngleClamped = currentMaxViewDegrees * verticalAspectScale; 

        // Clamp distance too, if desired
        float clampedDistance = Mathf.Clamp(elementDist, minDistance, maxDistance);

        if (currentAngle > currentMaxViewDegrees * verticalAspectScale || isSticky)
        {
            //Debug.Log("Current:" + currentAngle + ", " + (currentMaxViewDegrees * verticalAspectScale));
            float angRad = currentAngleClamped * Mathf.Deg2Rad;
            desiredPos = ReferencePoint + clampedDistance * (direction * Mathf.Cos(angRad) + perpendicularDirection * Mathf.Sin(angRad));
        }
        else if (!clampedDistance.Equals(elementDist))
        {
            // Only need to apply distance
            desiredPos = ReferencePoint + clampedDistance * elementDir;
        }

        if (currentAngle > 21 * verticalAspectScale)
            outOfFOV = true;
        else
            outOfFOV = false;
    }

    private IEnumerator LazyDisableIsLooking()
    {
        lazyEyeDisableProcessing = true;
        yield return new WaitForSeconds(1f);

        if (!Orb.Instance.IsLookingAtOrb)
            isLookingAtOrbFlag = false;

        lazyEyeDisableProcessing = false;
    }

    #region Sticky or Center pull 

    /// <summary>
    /// For important notifiations
    /// </summary>
    /// <param name="move"></param>
    public void MoveToCenter(bool toCenter)
    {
        this.stayCenter = toCenter;

        if (toCenter)
        {
            if (isSticky)
                SetSticky(false);

            currentMaxViewDegrees = maxViewDegreesCenter;
            MoveLerpTime = 0.3f;
        }
        else
        {
            if (isSticky)
                SetSticky(true);

            currentMaxViewDegrees = maxViewDegreesRegular;
            MoveLerpTime = regularMoveLerp;
            WorkingPosition = transform.position;
        }
    }


    /// <summary>
    /// If true, stick the orb to the edge of the FOV, else run regular solver
    /// </summary>
    /// <param name="isSticky"></param>
    public void SetSticky(bool isSticky)
    {
        this.isSticky = isSticky;

        if (stayCenter) return;

        if (isSticky)
        {
            currentMaxViewDegrees = maxViewDegreesSticky;
            MoveLerpTime = 0.7f;
            StopCoroutine("LazyDisableStickyMode()");
        }
        else
            StartCoroutine(LazyDisableStickyMode(maxViewDegreesSticky, -((maxViewDegreesSticky - maxViewDegreesRegular)) / 5, 5));
    }

    /// <summary>
    /// Routine that disables the sticky mode smoothly.
    /// </summary>
    /// <param name="start"> Start value of MaxViewDegrees</param>
    /// <param name="stepSize"> Step value</param>
    /// <param name="stepNum">Number of steps</param>
    /// <returns></returns>
    private IEnumerator LazyDisableStickyMode(float start, float stepSize, int stepNum)
    {
        yield return new WaitForSeconds(0.5f);
        int counter = 0;
        while (this.isSticky == false && counter < stepNum)
        {
            currentMaxViewDegrees = start+ (stepSize*stepNum);
            counter++;
            yield return new WaitForSeconds(0.1f);
        }

        if (this.isSticky == false)
        {
            WorkingPosition = transform.position;
            currentMaxViewDegrees = maxViewDegreesRegular;
            MoveLerpTime = regularMoveLerp;
        }
        else
        {
            currentMaxViewDegrees = maxViewDegreesSticky;
            MoveLerpTime = 0.7f;
        }
    }

    #endregion
}