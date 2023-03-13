using System;
using System.IO;
using System.Linq;
using UnityEngine;

public static class Utils
{
    public static bool InFOV(Camera cam, Vector3 obj, float thresX, float thresY)
    {
        Vector3 viewPos = cam.WorldToViewportPoint(obj);
        if (viewPos.x >= 0 + thresX && viewPos.x <= 1 - thresX && viewPos.y >= 0 && viewPos.y <= (1 - (thresY * 2)) && viewPos.z > 0)
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

    public static bool IsSameColor(Color c1, Color c2)
    {
        float thres = 0.02f;
        return (Math.Abs(c1.r - c2.r) < thres &&
                Math.Abs(c1.g - c2.g) < thres &&
                Math.Abs(c1.b - c2.b) < thres);
    }

    /// <summary>
    /// Save the texture as a file to the project folder
    /// </summary>
    /// <param name="tex"></param>
    public static void SaveCapture(Texture2D tex, string filenameNoExtension)
    {
        byte[] bytes = tex.EncodeToPNG();
        var dirPath = Application.dataPath + "/";
        if (!Directory.Exists(dirPath))
            Directory.CreateDirectory(dirPath);

        File.WriteAllBytes(dirPath + filenameNoExtension + ".png", bytes);
        Debug.Log("Saved to: " + dirPath + "filenameNoExtension" + ".png");
    }

    #region GUI and Screen transformations

    public static Vector3 GetRectPivot(Rect rect)
    {
        return new Vector3(rect.x + rect.width / 2,
            rect.y + rect.height / 2, 0.0000001f);
    }

    /// <summary>
    /// Cap the values of the rectangle to the screen (no negative values, not greated than screen size) in GUI coordinate system
    /// GUI coordinate system = origin is top left
    /// </summary>
    /// <param name="item"> Rectangle in GUI coordinate system </param>
    /// <returns></returns>
    public static int[] GetCappedGUI(Rect GUIRect)
    {
        int xmincap = Mathf.Max(0, (int)GUIRect.x);
        int ymincap = Mathf.Max(0, (int)GUIRect.y);

        int xmaxcap = Mathf.Min(AngelARUI.Instance.ARCamera.pixelWidth, (int)GUIRect.x + (int)GUIRect.width);
        int ymaxcap = Mathf.Min(AngelARUI.Instance.ARCamera.pixelHeight, (int)GUIRect.y + (int)GUIRect.height);

        return new int[] { xmincap, ymincap, xmaxcap, ymaxcap };
    }

    /// <summary>
    /// Cap the values of the rectangle to the screen (no negative values, not greated than screen size) in screen coordinate system
    /// Screen coordinate system = origin is bottom left
    /// </summary>
    /// <param name="item"> Rectangle in GUI coordinate system </param>
    /// <returns>xmin, ymin, xmax, ymax</returns>
    public static int[] GetCappedScreen(Rect item)
    {
        float realY = AngelARUI.Instance.ARCamera.pixelHeight - item.y - item.height;

        int xmincap = Mathf.Max(0, (int)item.x);
        int ymincap = Mathf.Max(0, (int)realY);

        int xmaxcap = Mathf.Min(AngelARUI.Instance.ARCamera.pixelWidth, (int)item.x + (int)item.width);
        int ymaxcap = Mathf.Min(AngelARUI.Instance.ARCamera.pixelHeight, (int)realY + (int)item.height);

        return new int[] { xmincap, ymincap, xmaxcap, ymaxcap };
    }

    #endregion

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

    // Get mean of data.
    public static float GetMean(float[] samples)
    {
        int n = samples.Length;
        float sum = 0;
        for (int i = 0; i < n; i++)
            sum = sum + samples[i];
        return sum / n;
    }

    // Get standard deviation of data
    public static float GetSD(float[] samples)
    {
        int n = samples.Length;
        float sum = 0;
        float mean = GetMean(samples);
        // find standard deviation
        for (int i = 0; i < n; i++)
            sum += (samples[i] - mean) * (samples[i] - mean);

        return (float)Math.Sqrt(sum/n);
    }

    // Get skewness of data
    public static float GetSkewness(float[] samples)
    {
        int n = samples.Length;
        // Find skewness using
        // above formula
        double sum = 0;
        float mean = GetMean(samples);
        float lower = 0;
        float upper = 0;
        for (int i = 0; i < n; i++)
        {
            float lp = (samples[i] - mean) * (samples[i] - mean);
            lower += lp;
            upper += lp * (samples[i] - mean);
        }
        return upper / ( Mathf.Pow(lower,(3/2)));
    }

}