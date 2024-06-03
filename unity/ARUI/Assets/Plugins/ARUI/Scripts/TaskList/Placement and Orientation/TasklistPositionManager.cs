using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Shapes;
using System.Diagnostics;

public class TasklistPositionManager : Singleton<TasklistPositionManager>
{
    public float movementSpeed = 1.0f;
    //Offset from camera if no objects exist
    public float xOffset;
    public float zOffset;

    private float _lerpTime = 1.0f;

    private Vector3 _lerpStart;
    private Vector3 _lerpEnd;
    private bool _isLerping;

    private float _timeStarted;

    private bool _isLooking = false;
    public bool IsLooking
    {
        get => _isLooking;
        set { _isLooking = value; }
    }

    private float _currentMaxDistance = 1.2f;

    /// <summary>
    /// Function to have the task overview snap to the center of all required objects
    /// This is done when the recipe goes from one step to another. By default, the overview
    /// stays on the center of all required objects
    /// Source -> https://www.blueraja.com/blog/404/how-to-use-unity-3ds-linear-interpolation-vector3-lerp-correctly
    /// </summary>
    public void SnapToCentroid()
    {
        //update maxDistance based on spatial map
        float distance = transform.position.GetDistanceToSpatialMap();
        if (distance != -1)
            _currentMaxDistance = Mathf.Max(ARUISettings.OrbMinDistToUser, Mathf.Min(distance - 0.05f, ARUISettings.OrbMaxDistToUser));

        Vector3 lookDirection = AngelARUI.Instance.ARCamera.transform.forward * _currentMaxDistance;

        Vector3 finalPos = new Vector3(lookDirection.x, AngelARUI.Instance.ARCamera.transform.position.y, lookDirection.z);

        _timeStarted = Time.time;
        _isLerping = true;
        _lerpStart = transform.position;
        _lerpEnd = finalPos;
    }

    public void SetPosition(Vector3 worldPosition)
    {
        transform.position = worldPosition;
    }

    void FixedUpdate()
    {
        if (_isLerping)
        {
            float timeSinceStarted = Time.time - _timeStarted;
            float percentComplete = timeSinceStarted / _lerpTime;
            transform.position = Vector3.Lerp(_lerpStart, _lerpEnd, percentComplete);
            if (percentComplete >= 1.0f)
            {
                _isLerping = false;
            }
        }
    }

}