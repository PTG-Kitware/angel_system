// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Microsoft.MixedReality.Toolkit.Utilities.Solvers;
using UnityEngine;
using System.Collections;

/// <summary>
/// Provides a solver for the Orb, using MRTK solver
/// </summary>
public class OrbFollowerSolver : Solver
{
    private VMControllable thisControllable;

    private bool _paused = false;                        /// < If true, the element will follow the user around
    public bool IsPaused
    {
        get => _paused;
        set { _paused = value; }
    }

    //** Radial and positional behavior of the orb
    private float _currentMaxDistance = 0.7f;           /// < Current max distance from eye to element, changes depending on the environment
    private float _currentMaxViewDegrees = 15f;         /// < The element will stay at least this close to the center of view
    
    private bool _coolDown = false;
    private Vector3 _coolDownTarget = Vector3.zero;

    private bool _stayCenter = false;
    private bool _isSticky = false;                     /// < If true, orb stays at the edge of the view cone

    private bool _outOfFOV = false;                     /// < If true, orb is not in the FOV of the user
    public bool IsOutOfFOV => _outOfFOV;

    //** Eye gaze events
    private bool _isLookingAtOrbFlag = false;           /// < If true, user is looking at orb
    private bool _lazyEyeDisableProcessing = false;

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

        MoveLerpTime = ARUISettings.OrbMoveLerpRegular;
        RotateLerpTime = 0.1f;
        Smoothing = true;
    }

    private void LateUpdate()
    {
        if (_isLookingAtOrbFlag && !Orb.Instance.IsLookingAtOrb(true) && !_lazyEyeDisableProcessing)
            StartCoroutine(LazyDisableIsLooking());
        else if (!_isLookingAtOrbFlag && Orb.Instance.IsLookingAtOrb(true))
            _isLookingAtOrbFlag = true;
    }

    public override void SolverUpdate()
    {
        // Update the collider AABB of the orb based on the position and visibility of the orb message
        thisControllable.UpdateRectBasedOnSubColliders(Orb.Instance.AllOrbColliders);

        if (!(_paused || _isLookingAtOrbFlag))
        {
            Vector3 goalPosition = WorkingPosition;

            if (!_coolDown)
            {
                //update maxDistance based on spatial map
                float dist = transform.position.GetCameraToPosDist();
                if (dist != -1)
                    _currentMaxDistance = Mathf.Max(ARUISettings.OrbMinDistToUser, Mathf.Min(dist - 0.05f, ARUISettings.OrbMaxDistToUser));

                bool moving = GetDesiredPos(ref goalPosition);
                if (moving)
                    StartCoroutine(CoolDown());
            }
            else
                goalPosition = _coolDownTarget;

            GoalPosition = goalPosition;
            transform.rotation = Quaternion.LookRotation(goalPosition - ReferencePoint, AngelARUI.Instance.ARCamera.transform.up);
        }
        else if (!(_isLookingAtOrbFlag))
        {
            //only update rotation 
            GoalPosition = transform.position;
            transform.rotation = Quaternion.LookRotation(transform.position - ReferencePoint, AngelARUI.Instance.ARCamera.transform.up);
            _outOfFOV = false;
        }
    }

    public IEnumerator CoolDown()
    {
        _coolDown = true;

        float dist = Vector3.Magnitude(transform.position - _coolDownTarget);
        while (dist > 0.01f)
        {
            yield return new WaitForEndOfFrame();

            //update maxDistance based on spatial map
            float distance = transform.position.GetCameraToPosDist();
            if (distance != -1)
                _currentMaxDistance = Mathf.Max(ARUISettings.OrbMinDistToUser, Mathf.Min(distance - 0.05f, ARUISettings.OrbMaxDistToUser));

            if (AngelARUI.Instance.IsVMActiv && !(_paused || _isLookingAtOrbFlag))
            {
                Rect getBest = ViewManagement.Instance.GetBestEmptyRect(thisControllable);
                if (getBest != Rect.zero)
                {
                    float depth = Mathf.Min(_currentMaxDistance, (transform.position - AngelARUI.Instance.ARCamera.transform.position).magnitude);
                    depth = Mathf.Max(depth, ARUISettings.OrbMinDistToUser);

                    _coolDownTarget = AngelARUI.Instance.ARCamera.ScreenToWorldPoint(Utils.GetRectPivot(getBest) + new Vector3(0, 0, depth));

                    yield return new WaitForSeconds(0.2f);
                }

            }

            dist = Vector3.Magnitude(transform.position - _coolDownTarget);

        }

        _coolDown = false;
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
        float currentAngleClamped = Mathf.Clamp(currentAngle, ARUISettings.OrbMinViewDegrees, _currentMaxViewDegrees * verticalAspectScale);

        if (_isSticky || _stayCenter)
            currentAngleClamped = _currentMaxViewDegrees * verticalAspectScale;

        // Clamp distance too, if desired
        float clampedDistance = Mathf.Clamp(elementDist, ARUISettings.OrbMinDistToUser, _currentMaxDistance);

        if (currentAngle > _currentMaxViewDegrees * verticalAspectScale || _isSticky)
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
                float depth = Mathf.Min(_currentMaxDistance, (transform.position - AngelARUI.Instance.ARCamera.transform.position).magnitude);
                depth = Mathf.Max(depth, ARUISettings.OrbMinDistToUser);

                _coolDownTarget = AngelARUI.Instance.ARCamera.ScreenToWorldPoint(Utils.GetRectPivot(getBest) + new Vector3(0, 0, depth));

                desiredPos = _coolDownTarget;
                return true;
            }

        }

        if (currentAngle > ARUISettings.OrbOutOfFOVThresV * verticalAspectScale)
            _outOfFOV = true;
        else
            _outOfFOV = false;

        return false;
    }

    /// <summary>
    /// Wait for a second before disabling the isLookingAtOrb flag
    /// </summary>
    /// <returns></returns>
    private IEnumerator LazyDisableIsLooking()
    {
        _lazyEyeDisableProcessing = true;
        yield return new WaitForSeconds(1f);

        if (!Orb.Instance.IsLookingAtOrb(true))
            _isLookingAtOrbFlag = false;

        _lazyEyeDisableProcessing = false;
    }

    #region Sticky or Center pull 

    /// <summary>
    /// For important notification, pull the orb to the center of the FOV
    /// </summary>
    /// <param name="move"></param>
    public void MoveToCenter(bool toCenter)
    {
        this._stayCenter = toCenter;

        if (toCenter)
        {
            if (_isSticky)
                SetSticky(false);

            _currentMaxViewDegrees = ARUISettings.OrbMaxViewDegCenter;
            MoveLerpTime = 0.3f;
        }
        else
        {
            _currentMaxViewDegrees = ARUISettings.OrbMaxViewDegRegular;
            MoveLerpTime = ARUISettings.OrbMoveLerpRegular;
            WorkingPosition = transform.position;
        }
    }


    /// <summary>
    /// If true, stick the orb to the edge of the FOV, else run regular solver
    /// </summary>
    /// <param name="isSticky"></param>
    public void SetSticky(bool isSticky)
    {
        this._isSticky = isSticky;

        if (_stayCenter) return;

        if (isSticky)
        {
            _currentMaxViewDegrees = ARUISettings.OrbMaxViewDegSticky;
            MoveLerpTime = ARUISettings.OrbMoveLerpRegular;
            StopCoroutine("LazyDisableStickyMode()");
        }
        else
            StartCoroutine(LazyDisableStickyMode(ARUISettings.OrbMaxViewDegSticky, -((ARUISettings.OrbMaxViewDegSticky - ARUISettings.OrbMaxViewDegRegular)) / 5, 5));
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
        while (this._isSticky == false && counter < stepNum)
        {
            _currentMaxViewDegrees = start + (stepSize * stepNum);
            counter++;
            yield return new WaitForSeconds(0.1f);
        }

        if (this._isSticky == false)
        {
            WorkingPosition = transform.position;
            _currentMaxViewDegrees = ARUISettings.OrbMaxViewDegRegular;
            MoveLerpTime = ARUISettings.OrbMoveLerpRegular;
        }
        else
        {
            _currentMaxViewDegrees = ARUISettings.OrbMaxViewDegSticky;
            MoveLerpTime = ARUISettings.OrbMoveLerpRegular;
        }
    }

    #endregion
}