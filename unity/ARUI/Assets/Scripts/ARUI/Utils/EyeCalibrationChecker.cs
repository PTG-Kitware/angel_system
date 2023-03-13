using Microsoft.MixedReality.Toolkit;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

public class EyeCalibrationChecker : MonoBehaviour
{
    private bool? prevCalibrationStatus = null;

    private void Update()
    {
        bool? calibrationStatus;

        calibrationStatus = CoreServices.InputSystem?.EyeGazeProvider?.IsEyeCalibrationValid;

        if (calibrationStatus.HasValue)
        {
            if (prevCalibrationStatus != calibrationStatus)
            {
                if (!calibrationStatus.Value)
                {
                    AngelARUI.Instance.LogDebugMessage("Eye Tracking Calibrationstatus: false",true);
                }
                else
                {
                    AngelARUI.Instance.LogDebugMessage("Eye Tracking Calibrationstatus: true", true);
                }
                prevCalibrationStatus = calibrationStatus;
            }
        }
    }
}
