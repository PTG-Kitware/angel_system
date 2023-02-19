using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;
using Microsoft.MixedReality.Toolkit;
using UnityEngine;
using DilmerGames.Core.Singletons;
using System;
using System.Diagnostics.Eventing.Reader;

public enum EyeTarget
{
    nothing = 0,
    orbFace = 1,
    orbMessage = 2,
    tasklist = 3,
    orbtasklistButton =4,
    okButton=5,
    cancelButton=6,
}

public class FollowEyeTarget : Singleton<FollowEyeTarget>
{
    public EyeTarget currentHit = EyeTarget.nothing;
    private MeshRenderer cube;

    private bool showRayDebugCube = false;

    private void Awake() => cube = gameObject.GetComponent<MeshRenderer>();

    private void Update()
    {
        var eyeGazeProvider = CoreServices.InputSystem?.EyeGazeProvider;
        if (eyeGazeProvider != null)
        {
            gameObject.transform.position = eyeGazeProvider.GazeOrigin + eyeGazeProvider.GazeDirection.normalized * 2.0f;
            cube.enabled = false;

            Ray rayToCenter = new Ray(eyeGazeProvider.GazeOrigin, eyeGazeProvider.GazeDirection);
            RaycastHit hitInfo;

            int layerMask = 1 << 5; //Ignore everything except layer 5, which is the UI
            UnityEngine.Physics.Raycast(rayToCenter, out hitInfo, 100f, layerMask);

            // Update GameObject to the current eye gaze position at a given distance
            if (hitInfo.collider != null)
            {
                float dist = (hitInfo.point - AngelARUI.Instance.ARCamera.transform.position).magnitude;
                gameObject.transform.position = eyeGazeProvider.GazeOrigin + eyeGazeProvider.GazeDirection.normalized * dist;
                //Debug.Log(hitInfo.collider.gameObject.name);
                string goName = hitInfo.collider.gameObject.name.ToLower();

                if (goName.Contains("flexibletextcontainer_orb"))
                    currentHit = EyeTarget.orbMessage;

                else if (goName.Contains("bodyplacement"))
                    currentHit = EyeTarget.orbFace;

                else if (goName.Contains("tasklistcontainer"))
                    currentHit = EyeTarget.tasklist;

                else if (goName.Contains("facetasklistbutton"))
                    currentHit = EyeTarget.orbtasklistButton;

                else if (goName.Contains("okbutton"))
                    currentHit = EyeTarget.okButton;

                else if (goName.Contains("cancelbutton"))
                    currentHit = EyeTarget.cancelButton;
                else
                    currentHit = EyeTarget.nothing;

                if (currentHit != EyeTarget.nothing && showRayDebugCube)
                    cube.enabled = true;
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

    public void ShowDebugTarget(bool showEyeGazeTarget) => showRayDebugCube = showEyeGazeTarget;
}
