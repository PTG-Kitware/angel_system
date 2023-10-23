using Microsoft.MixedReality.Toolkit;
using System.Diagnostics;
using UnityEngine;

public enum EyeTarget
{
    nothing = 0,
    orbFace = 1,
    orbMessage = 2,
    tasklist = 3,
    orbtasklistButton = 4,
    menuBtn = 5,
    listmenuButton_tasks = 6,
    okButton = 7,
    ringindicator = 8,
    textConfirmationWindow = 9,
    listmenuButton_items = 10,
    upButton = 11,
    downButton =12,
    resetButton = 13,
    pieCollider = 14
}

public class EyeGazeManager : Singleton<EyeGazeManager>
{
    public EyeTarget CurrentHit = EyeTarget.nothing;
    public GameObject CurrentHitObj;

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
                //Debug.Log(hitInfo.collider.gameObject.name);
                string goName = hitInfo.collider.gameObject.name.ToLower();

                if (goName.Contains("flexibletextcontainer_orb"))
                    CurrentHit = EyeTarget.orbMessage;

                else if (goName.Contains("bodyplacement"))
                    CurrentHit = EyeTarget.orbFace;

                else if (goName.Contains(StringResources.tasklist_name.ToLower()))
                    CurrentHit = EyeTarget.tasklist;

                else if (goName.Contains("facetasklistbutton"))
                    CurrentHit = EyeTarget.orbtasklistButton;

                else if (goName.Contains("okbutton"))
                    CurrentHit = EyeTarget.okButton;

                else if (goName.Contains("flexibletextcontainer_window"))
                    CurrentHit = EyeTarget.textConfirmationWindow;

                else if (goName.Contains("ringindicator"))
                    CurrentHit = EyeTarget.ringindicator;

                else if (goName.Contains("listmenubutton_tasks"))
                    CurrentHit = EyeTarget.listmenuButton_tasks;

                else if (goName.Contains("listmenubutton_items"))
                    CurrentHit = EyeTarget.listmenuButton_items;

                else if (goName.Contains("upbutton"))
                    CurrentHit = EyeTarget.upButton;

                else if (goName.Contains("downbutton"))
                    CurrentHit = EyeTarget.downButton;

                else if (goName.Contains("resetbutton"))
                    CurrentHit = EyeTarget.resetButton;

                else if (goName.Contains("piecollider"))
                    CurrentHit = EyeTarget.pieCollider;

                else if (goName.Contains("_menu"))
                    CurrentHit = EyeTarget.menuBtn;

                else
                    CurrentHit = EyeTarget.nothing;

                if (CurrentHit != EyeTarget.nothing)
                {
                    CurrentHitObj = hitInfo.collider.gameObject;
                    if (_showRayDebugCube)
                        _eyeGazeTargetCube.enabled = true;
                }
                else if (CurrentHit == EyeTarget.nothing)
                    CurrentHitObj = null;

            }
            else
            {
                // If no target is hit, show the object at a default distance along the gaze ray.
                gameObject.transform.position = eyeGazeProvider.GazeOrigin + eyeGazeProvider.GazeDirection.normalized * 2.0f;
                CurrentHit = EyeTarget.nothing;
                CurrentHitObj = null;
            }
        }
        else
        {
            CurrentHit = EyeTarget.nothing;
            CurrentHitObj = null;
        }
    }

    public void ShowDebugTarget(bool showEyeGazeTarget) => _showRayDebugCube = showEyeGazeTarget;
}
