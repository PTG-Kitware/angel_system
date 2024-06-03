using Microsoft.MixedReality.Toolkit;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;

public class EyeGazeManager : Singleton<EyeGazeManager>
{
    public int CurrentHitID = -1;

    private List<int> _eyeTargetIDs = new List<int>();

    /// ** Debug eye gaze target cube
    private MeshRenderer _eyeGazeTargetCube;
    private bool _showRayDebugCube = false;

    private void Awake() => _eyeGazeTargetCube = gameObject.GetComponent<MeshRenderer>();

    private void Update()
    {
        var eyeGazeProvider = CoreServices.InputSystem?.EyeGazeProvider;
        if (eyeGazeProvider != null)
        {
            gameObject.transform.position = eyeGazeProvider.GazeOrigin + eyeGazeProvider.GazeDirection.normalized * 2.0f;
            _eyeGazeTargetCube.enabled = false;

            Ray rayToCenter = new Ray(eyeGazeProvider.GazeOrigin, eyeGazeProvider.GazeDirection);
            RaycastHit hitInfo;

            int layerMask = LayerMask.GetMask(StringResources.UI_layer, StringResources.VM_layer);
            UnityEngine.Physics.Raycast(rayToCenter, out hitInfo, 100f, layerMask);

            // Update GameObject to the current eye gaze position at a given distance
            if (hitInfo.collider != null)
            {
                float dist = (hitInfo.point - AngelARUI.Instance.ARCamera.transform.position).magnitude;
                gameObject.transform.position = eyeGazeProvider.GazeOrigin + eyeGazeProvider.GazeDirection.normalized * dist;

                //UnityEngine.Debug.Log("Currently looking at:" + hitInfo.collider.gameObject.name+" with ID"+ hitInfo.collider.gameObject.GetInstanceID());
                
                if (_eyeTargetIDs.Contains(hitInfo.collider.gameObject.GetInstanceID()))
                {
                    CurrentHitID = hitInfo.collider.gameObject.GetInstanceID();
                    if (_showRayDebugCube)
                    {
                        _eyeGazeTargetCube.enabled = true;
                    }
                } else
                    CurrentHitID = -1;
            }
            else
            {
                // If no target is hit, show the object at a default distance along the gaze ray.
                gameObject.transform.position = eyeGazeProvider.GazeOrigin + eyeGazeProvider.GazeDirection.normalized * 2.0f;
                CurrentHitID = -1;
            }
        }
        else
        {
            CurrentHitID = -1;
        }
    }

    public void RegisterEyeTargetID(GameObject ob)
    {
        AngelARUI.Instance.DebugLogMessage("Registered Collision Events with "+ ob.name+" and ID "+ ob.GetInstanceID(), false);
        _eyeTargetIDs.Add(ob.GetInstanceID());
    }

    public void ShowDebugTarget(bool showEyeGazeTarget) => _showRayDebugCube = showEyeGazeTarget;
}
