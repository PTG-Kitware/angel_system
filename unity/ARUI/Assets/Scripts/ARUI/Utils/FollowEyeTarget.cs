using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;
using Microsoft.MixedReality.Toolkit;
using UnityEngine;
using DilmerGames.Core.Singletons;
using System;

public enum EyeTarget
{
    nothing = 0,
    orbFace = 1,
    orbMessage = 2,
    tasklist = 3,
    orbtasklistButton =4
}

public class FollowEyeTarget : Singleton<FollowEyeTarget>
{
    public EyeTarget currentHit = EyeTarget.nothing;
    private MeshRenderer cube;

    private void Awake()
    {
        cube=gameObject.GetComponent<MeshRenderer>();
    }

    private void Update()
    {
        var eyeGazeProvider = CoreServices.InputSystem?.EyeGazeProvider;
        if (eyeGazeProvider != null)
        {
            gameObject.transform.position = eyeGazeProvider.GazeOrigin + eyeGazeProvider.GazeDirection.normalized * 2.0f;

            EyeTrackingTarget lookedAtEyeTarget = EyeTrackingTarget.LookedAtEyeTarget;

            // Update GameObject to the current eye gaze position at a given distance
            if (lookedAtEyeTarget != null)
            {
                Ray rayToCenter = new Ray(CameraCache.Main.transform.position, lookedAtEyeTarget.transform.position - CameraCache.Main.transform.position);
                RaycastHit hitInfo;
                UnityEngine.Physics.Raycast(rayToCenter, out hitInfo);

                float dist = (hitInfo.point - CameraCache.Main.transform.position).magnitude;
                gameObject.transform.position = eyeGazeProvider.GazeOrigin + eyeGazeProvider.GazeDirection.normalized * dist;

                if (lookedAtEyeTarget.gameObject.name.Contains("TextContainer"))
                    currentHit = EyeTarget.orbMessage;

                else if (lookedAtEyeTarget.gameObject.name.Contains("BodyPlacement"))
                    currentHit = EyeTarget.orbFace;

                else if (lookedAtEyeTarget.gameObject.name.Contains("Tasklist"))
                    currentHit = EyeTarget.tasklist;

                else if (lookedAtEyeTarget.gameObject.name.Contains("FaceTaskListButton"))
                    currentHit = EyeTarget.orbtasklistButton;
                else
                    currentHit = EyeTarget.nothing;

                //Debug.Log(lookedAtEyeTarget.gameObject.name);
            }
            else
            {
                // If no target is hit, show the object at a default distance along the gaze ray.
                gameObject.transform.position = eyeGazeProvider.GazeOrigin + eyeGazeProvider.GazeDirection.normalized * 2.0f;
                currentHit = EyeTarget.nothing;
            }
        } else
        {
            currentHit = EyeTarget.nothing;
        }
    }

    public void ShowDebugTarget(bool showEyeGazeTarget) => cube.enabled = showEyeGazeTarget;
}
