using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public static class Utils
{
    public static bool InFOV(Camera cam, Vector3 obj, float thresX, float thresY)
    {
        Vector3 viewPos = cam.WorldToViewportPoint(obj);
        if (viewPos.x >= 0+ thresX && viewPos.x <= 1- thresX && viewPos.y >= 0 && viewPos.y <= (1- (thresY*2)) && viewPos.z > 0)
            return true;

        return false;
    }

    public static bool InFOV(Camera cam, Vector3 obj)
    {
        Vector3 viewPos = cam.WorldToViewportPoint(obj);
        if (viewPos.x >= 0 && viewPos.x <= 1 && viewPos.y >= 0 && viewPos.y <= 1 && viewPos.z > 0)
            return true;

        return false;
    }

    /// <summary>
    /// Returns the distance from the camera to the spatial mesh created by hololens2 towards a given target position
    /// </summary>
    /// <param name="target"></param>
    /// <returns></returns>
    public static float GetCameraToPosDist(Vector3 target)
    {
        // Bit shift the index of the layer (8) to get a bit mask
        int layerMask = 1 << 31;

        RaycastHit hit;
        // Does the ray intersect any objects excluding the player layer
        if (Physics.Raycast(AngelARUI.Instance.ARCamera.transform.position, target - AngelARUI.Instance.transform.position, out hit, Mathf.Infinity, layerMask))
            return Mathf.Abs(hit.distance);

        return -1;
    }

    /// <summary>
    /// Split the given text into lines.
    /// </summary>
    /// <param name="text"></param>
    /// <param name="maxCharCountPerLine">maximum allowed characters per line</param>
    /// <returns></returns>
    public static string SplitTextIntoLines(string text, int maxCharCountPerLine)
    {
        var charCount = 0;
        var lines = text.Split(new string[] { " " }, StringSplitOptions.RemoveEmptyEntries)
                        .GroupBy(w => (charCount += w.Length + 1) / maxCharCountPerLine)
                        .Select(g => string.Join(" ", g));
        return String.Join("\n", lines.ToArray());
    }

}