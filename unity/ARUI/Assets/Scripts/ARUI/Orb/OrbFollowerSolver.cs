// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Microsoft.MixedReality.Toolkit.Utilities.Solvers;
using UnityEngine;
using Microsoft.MixedReality.Toolkit;
using System.Collections;
using System;
using UnityEditor.Build;
/// <summary>
/// Provides a solver for the Orb, using MRTK solver
/// </summary>
public class OrbFollowerSolver : Solver
{
    private VMControllable thisControllable;

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

    private bool coolDown = false;
    private Vector3 coolDownTarget = Vector3.zero;

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

    public bool vmIsOn = false;

    /// <summary>
    /// Position to the view direction, or the movement direction, or the direction of the view cone.
    /// </summary>
    private Vector3 SolverReferenceDirection => SolverHandler.TransformTarget != null ? SolverHandler.TransformTarget.forward : Vector3.forward;

    private Vector3 ReferencePoint => SolverHandler.TransformTarget != null ? SolverHandler.TransformTarget.position : Vector3.zero;

    private new void Start()
    {
        base.Start();

        thisControllable = GetComponent<VMControllable>();
        if (thisControllable == null)
            thisControllable = gameObject.AddComponent<VMControllable>();

        MoveLerpTime = regularMoveLerp;
        RotateLerpTime = 0.1f;
        Smoothing = true;

        vmIsOn =  AngelARUI.Instance.IsVMActiv;
    }

    private void LateUpdate()
    {
        if (isLookingAtOrbFlag && !Orb.Instance.IsLookingAtOrb(true) && !lazyEyeDisableProcessing)
            StartCoroutine(LazyDisableIsLooking());
        else if (!isLookingAtOrbFlag && Orb.Instance.IsLookingAtOrb(true))
            isLookingAtOrbFlag = true;
    }


    public override void SolverUpdate()
    {
        // Update the collider AABB of the orb based on the position and visibility of the orb message
        thisControllable.UpdateRectBasedOnSubColliders(Orb.Instance.GetCurrentMessageCollider(), Orb.Instance.FaceCollider);
        
        if (!(paused || isLookingAtOrbFlag))
        {
            Vector3 goalPosition = WorkingPosition;

            // Update the collider AABB of the orb based on the position and visibility of the orb message
            thisControllable.UpdateRectBasedOnSubColliders(Orb.Instance.GetCurrentMessageCollider(), Orb.Instance.FaceCollider);

            if (!coolDown)
            {
                //update maxDistance based on spatial map
                float dist = Utils.GetCameraToPosDist(transform.position);
                if (dist != -1)
                    maxDistance = Mathf.Max(minDistance, Mathf.Min(dist - 0.05f, 1.0f));

                bool moving = GetDesiredPos(ref goalPosition);
                if (moving)
                    StartCoroutine(CoolDown());
            }
            else
                goalPosition = coolDownTarget;

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

    public IEnumerator CoolDown()
    {
        coolDown = true;

        float dist = Vector3.Magnitude(transform.position - coolDownTarget);
        while (dist > 0.01f)
        {
            yield return new WaitForEndOfFrame();

            //update maxDistance based on spatial map
            float distance = Utils.GetCameraToPosDist(transform.position);
            if (distance != -1)
                maxDistance = Mathf.Max(minDistance, Mathf.Min(distance - 0.05f, 1.0f));

            if (vmIsOn)
            {
                Rect getBest = ViewManagement.Instance.GetBestEmptyRect(thisControllable);
                if (getBest != Rect.zero)
                {
                    float depth = Mathf.Min(maxDistance, (transform.position - AngelARUI.Instance.ARCamera.transform.position).magnitude);
                    depth = Mathf.Max(depth, minDistance);

                    coolDownTarget = AngelARUI.Instance.ARCamera.ScreenToWorldPoint(Utils.GetRectPivot(getBest) + new Vector3(0, 0, depth));

                    yield return new WaitForSeconds(0.2f);
                }

            }

            dist = Vector3.Magnitude(transform.position - coolDownTarget);

        }

        coolDown = false;
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

        if (AngelARUI.Instance.IsVMActiv)
        {
            Rect getBest = ViewManagement.Instance.GetBestEmptyRect(thisControllable);
            if (getBest != Rect.zero)
            {
                float depth = Mathf.Min(maxDistance, (transform.position - AngelARUI.Instance.ARCamera.transform.position).magnitude);
                depth = Mathf.Max(depth, minDistance);

                coolDownTarget = AngelARUI.Instance.ARCamera.ScreenToWorldPoint(Utils.GetRectPivot(getBest) + new Vector3(0, 0, depth));

                desiredPos = coolDownTarget;
                return true;
            }
            
        }

        if (currentAngle > 21 * verticalAspectScale)
            outOfFOV = true;
        else
            outOfFOV = false;

        return false;
    }

    private IEnumerator LazyDisableIsLooking()
    {
        lazyEyeDisableProcessing = true;
        yield return new WaitForSeconds(1f);

        if (!Orb.Instance.IsLookingAtOrb(true))
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
            currentMaxViewDegrees = start + (stepSize * stepNum);
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