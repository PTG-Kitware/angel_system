// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Microsoft.MixedReality.Toolkit.Utilities.Solvers;
using UnityEngine;
using Microsoft.MixedReality.Toolkit;
using System.Collections;
using System;

/// <summary>
/// Provides a solver for the Orb, based on MRTK solver
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
    private float currentMaxViewDegrees = 10f;
    private float maxViewDegreesRegular = 10f;
    private float maxViewDegreesSticky = 23f;

    [HideInInspector]
    public bool useEyeTarget = false;

    private bool lazyEyeDisableProcessing = false;

    [HideInInspector]
    public bool paused = false;
    [HideInInspector]
    public bool isSticky;

    private bool isLookingAtOrbFlag = false;

    /// <summary>
    /// Position to the view direction, or the movement direction, or the direction of the view cone.
    /// </summary>
    private Vector3 SolverReferenceDirection => SolverHandler.TransformTarget != null ? SolverHandler.TransformTarget.forward : Vector3.forward;

    private Vector3 ReferencePoint => SolverHandler.TransformTarget != null ? SolverHandler.TransformTarget.position : Vector3.zero;

    private new void Start()
    {
        base.Start();

        StartCoroutine(RunDistanceUpdate());

        MoveLerpTime = 0.3f;
        RotateLerpTime = 0.1f;
        Smoothing = true;
    }

    public new void Update()
    {
        if (isLookingAtOrbFlag && !Orb.Instance.GetIsUserLookingAtOrb() && !lazyEyeDisableProcessing)
            StartCoroutine(LazyDisableIsLooking());
        else if (!isLookingAtOrbFlag && Orb.Instance.GetIsUserLookingAtOrb())
            isLookingAtOrbFlag = true;
    }

    /// <inheritdoc />
    public override void SolverUpdate()
    {
        if (!(paused || isLookingAtOrbFlag))
        {
            Vector3 goalPosition = WorkingPosition;
            GetDesiredPos(ref goalPosition);
            GoalPosition = goalPosition;
            transform.rotation = Quaternion.LookRotation(goalPosition - ReferencePoint, Vector3.up);
        }
        else
        {
            transform.rotation = Quaternion.LookRotation(transform.position - ReferencePoint, Vector3.up);
        }
    }

    private bool GetDesiredPos(ref Vector3 desiredPos)
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

        // Calculate the current angle
        float currentAngle = Vector3.Angle(elementDir, direction);
        float currentAngleClamped = Mathf.Clamp(currentAngle, minViewDegrees * verticalAspectScale, currentMaxViewDegrees * verticalAspectScale);
        if (isSticky)
            currentAngleClamped = currentMaxViewDegrees * verticalAspectScale;

        // Clamp distance too, if desired
        float clampedDistance = Mathf.Clamp(elementDist, minDistance, maxDistance);

        ///For notifications
        if (useEyeTarget)
        {   // The orb should stay around the eye target
            var eyeGazeProvider = CoreServices.InputSystem?.EyeGazeProvider;
            Vector3 fallbackPos = (AngelARUI.Instance.mainCamera.transform.forward * 0.7f);
            if (eyeGazeProvider != null)
            {
                Vector3 newPos = (eyeGazeProvider.GazeOrigin + (eyeGazeProvider.GazeDirection.normalized * 0.7f));
                if (Utils.InFOV(AngelARUI.Instance.mainCamera, newPos))
                    desiredPos = newPos;
                else
                    desiredPos = fallbackPos;
            }
            else
                desiredPos = fallbackPos;
        }
        else 
        {
            //If the angle was clamped, do some special update stuff
            if (currentAngle != currentAngleClamped || isSticky)
            {
                float angRad = currentAngleClamped * Mathf.Deg2Rad;
                // Calculate new position
                desiredPos = ReferencePoint + clampedDistance * (direction * Mathf.Cos(angRad) + perpendicularDirection * Mathf.Sin(angRad));
            }
            else if (!clampedDistance.Equals(elementDist))
            {
                // Only need to apply distance
                desiredPos = ReferencePoint + clampedDistance * elementDir;
            }
        }

        return false;
    }


    /// <summary>
    /// Update maximum distance between the head and the orb based on the spatial environment, so that the orb is not placed
    /// behind physical objects.
    /// </summary>
    /// <returns></returns>
    private IEnumerator RunDistanceUpdate()
    {
        while (true)
        {
            if (!paused)
            {
                float dist = Utils.GetCameraToPosDist(transform.position);

                if (dist != -1)
                    maxDistance = Mathf.Max(minDistance, Mathf.Min(dist-0.05f, 1.0f));
            }

            yield return new WaitForSeconds(1f);
        }
    }

    public void MoveToEyeTarget(bool move) {
        useEyeTarget = move;
    }

    private IEnumerator LazyDisableIsLooking()
    {
        lazyEyeDisableProcessing = true;
        yield return new WaitForSeconds(0.5f);

        if (!Orb.Instance.GetIsUserLookingAtOrb())
            isLookingAtOrbFlag = false;

        lazyEyeDisableProcessing = false;
    }


    #region Getter and Setter

    public bool GetPaused() => paused;
    public void SetPaused(bool isPaused) => paused = isPaused;

    public void SetSticky(bool isSticky)
    {
        this.isSticky = isSticky;
        
        if (isSticky)
        {
            currentMaxViewDegrees = maxViewDegreesSticky;
            MoveLerpTime = 0.5f;

            StopCoroutine("LazyDisableStickyMode()");
        }
        else
        {
            StartCoroutine(LazyDisableStickyMode(0.5f, (maxViewDegreesSticky-maxViewDegreesRegular)/5, 5,false));
        }
            
    }

    private IEnumerator LazyDisableStickyMode(float start, float stepSize, int stepNum, bool status)
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
            MoveLerpTime = 0.3f;
        } else
        {
            currentMaxViewDegrees = maxViewDegreesSticky;
            MoveLerpTime = 0.5f;
        }
    }

    #endregion
}