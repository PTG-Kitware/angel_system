using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

public static class Utils
{
    /// <summary>
    /// Check if two colors are essentially the same, given the threshold
    /// </summary>
    /// <param name="c1"></param>
    /// <param name="c2"></param>
    /// <param name="thres"></param>
    /// <returns></returns>
    public static bool IsSameColor(Color c1, Color c2, float thres)
    {
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

    /// <summary>
    /// https://docs.unity3d.com/ScriptReference/LayerMask.GetMask.html
    /// </summary>
    /// <param name="layerName"></param>
    /// <returns></returns>
    public static int GetLayerInt(string layerName) => (int)Mathf.Log(LayerMask.GetMask(layerName), 2);

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

    private static Vector2[] GetScreenCorners(Camera cam, List<BoxCollider> bxcols, ref Vector3[] worldCorners)
    {
        float scalingValue = 1.0f;

        int i = 0;
        int index = 0;
        foreach (var item in bxcols)
        {
            Transform current = item.transform;
            worldCorners[index] = current.TransformPoint(item.center + (new Vector3(-item.size.x * scalingValue, -item.size.y * scalingValue, -item.size.z * scalingValue) * 0.5f));
            worldCorners[index + 1] = current.TransformPoint(item.center + (new Vector3(item.size.x * scalingValue, -item.size.y * scalingValue, -item.size.z * scalingValue) * 0.5f));
            worldCorners[index + 2] = current.TransformPoint(item.center + (new Vector3(item.size.x * scalingValue, -item.size.y * scalingValue, item.size.z * scalingValue) * 0.5f));
            worldCorners[index + 3] = current.TransformPoint(item.center + (new Vector3(-item.size.x * scalingValue, -item.size.y * scalingValue, item.size.z * scalingValue) * 0.5f));
            worldCorners[index + 4] = current.TransformPoint(item.center + (new Vector3(-item.size.x * scalingValue, item.size.y * scalingValue, -item.size.z * scalingValue) * 0.5f));
            worldCorners[index + 5] = current.TransformPoint(item.center + (new Vector3(item.size.x * scalingValue, item.size.y * scalingValue, -item.size.z * scalingValue) * 0.5f));
            worldCorners[index + 6] = current.TransformPoint(item.center + (new Vector3(item.size.x * scalingValue, item.size.y * scalingValue, item.size.z * scalingValue) * 0.5f));
            worldCorners[index + 7] = current.TransformPoint(item.center + (new Vector3(-item.size.x * scalingValue, item.size.y * scalingValue, item.size.z * scalingValue) * 0.5f));

            i += 1;
            index += 8;
        }

        Vector2[] corners = new Vector2[bxcols.Count * 8];
        for (int j = 0; j < corners.Length; j++)
            corners[j] = cam.WorldToScreenPoint(worldCorners[j]);

        return corners;
    }

    /// <summary>
    /// TODO
    /// </summary>
    /// <param name="taskName"></param>
    /// <returns></returns>
    public static int ArrayContainsKey(string[] array, string taskName)
    {
        for (int i = 0; i<array.Length; i++)
        {
            if (array[i].Equals(taskName))
                return i;
        }

        return -1;
    }

    /// <summary>
    /// Cut off string after nuberOfWords.
    /// </summary>
    /// <param name="text"></param>
    /// <returns>the first 'numberOfWords' of the given string 'text' or less</returns>
    public static string GetCappedText(string text, int numberOfWords)
    {
        // Split the input text into words using space as the delimiter
        string[] words = text.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

        // Take the first 'maxWords' words and join them back into a string
        return string.Join(" ", words.Take(numberOfWords));
    }

#if UNITY_EDITOR
    /// <summary>
    /// Adds the layer.
    /// </summary>
    /// <returns><c>true</c>, if layer was added, <c>false</c> otherwise.</returns>
    /// <param name="layerName">Layer name.</param>
    public static bool CreateLayer(string layerName, int layerInt)
    {
        // Open tag manager
        SerializedObject tagManager = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset")[0]);
        // Layers Property
        SerializedProperty layersProp = tagManager.FindProperty("layers");
        if (!PropertyExists(layersProp, layerInt, layerInt, layerName))
        {
            SerializedProperty sp;
            sp = layersProp.GetArrayElementAtIndex(layerInt);
            if (sp.stringValue == "")
            {
                // Assign string value to layer
                sp.stringValue = layerName;
                Debug.Log("Layer: " + layerName + " has been added. Please add layer '"+layerName+"' at "+ layerInt+" to avoid compilation issues later.");
                // Save settings
                tagManager.ApplyModifiedProperties();
                return true;
            }
        }
        else
        {
            Debug.Log ("Layer: " + layerName + " already exists");
        }

        return false;
    }

    /// <summary>
    /// https://forum.unity.com/threads/create-tags-and-layers-in-the-editor-using-script-both-edit-and-runtime-modes.732119/
    /// Checks if the value exists in the property.
    /// </summary>
    /// <returns><c>true</c>, if exists was propertyed, <c>false</c> otherwise.</returns>
    /// <param name="property">Property.</param>
    /// <param name="start">Start.</param>
    /// <param name="end">End.</param>
    /// <param name="value">Value.</param>
    /// 
    private static bool PropertyExists(SerializedProperty property, int start, int end, string value)
    {
        for (int i = start; i < end; i++)
        {
            SerializedProperty t = property.GetArrayElementAtIndex(i);
            if (t.stringValue.Equals(value))
            {
                return true;
            }
        }
        return false;
    }

    #endif

    #region GUI and Screen transformations

    /// <summary>
    /// Get the pivot point of the 2D rectangle
    /// </summary>
    /// <param name="rect"></param>
    /// <returns></returns>
    public static Vector3 GetRectPivot(Rect rect)
    {
        return new Vector3(rect.x + rect.width / 2, rect.y + rect.height / 2, 0.0000001f);
    }

    /// <summary>
    /// Cap the values of the rectangle to the screen (no negative values, not greated than screen size) in GUI coordinate system
    /// GUI coordinate system = origin is top left
    /// </summary>
    /// <param name="item"> Rectangle in GUI coordinate system </param>
    /// <returns>Rectangle in GUI</returns>
    public static int[] GUIGetCappedGUI(Rect GUIRect)
    {
        int xmincap = Mathf.Max(0, (int)GUIRect.x);
        int ymincap = Mathf.Max(0, (int)GUIRect.y);

        int xmaxcap = Mathf.Min(AngelARUI.Instance.ARCamera.pixelWidth, (int)GUIRect.x + (int)GUIRect.width);
        int ymaxcap = Mathf.Min(AngelARUI.Instance.ARCamera.pixelHeight, (int)GUIRect.y + (int)GUIRect.height);

        return new int[] { xmincap, ymincap, xmaxcap, ymaxcap };
    }

    public static int[] ScreenToGUI(Rect item)
    {
        float realY = AngelARUI.Instance.ARCamera.pixelHeight - item.y - item.height;

        int xmincap = Mathf.Max(0, (int)item.x);
        int ymincap = Mathf.Max(0, (int)realY);

        int xmaxcap = Mathf.Min(AngelARUI.Instance.ARCamera.pixelWidth, (int)item.x + (int)item.width);
        int ymaxcap = Mathf.Min(AngelARUI.Instance.ARCamera.pixelHeight, (int)realY + (int)item.height);

        return new int[] { xmincap, ymincap, xmaxcap, ymaxcap };
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

    public static Rect GUItoScreenRect(Rect item)
    {
        int[] screen = GetCappedScreen(item);
        return new Rect(screen[0], screen[1], screen[2] - screen[0], screen[3] - screen[1]);
    }

    /// <summary>
    /// https://forum.unity.com/threads/how-do-i-get-the-oriented-bounding-box-values.538239/
    /// </summary>
    /// <param name="t"></param>
    /// <param name="cam"></param>
    /// <returns>Rect of given collider in </returns>
    public static Rect RectFromObjs(this Transform t1, Camera cam, List<BoxCollider> bxcols)
    {
        List<BoxCollider> activeColliders = new List<BoxCollider>();
        foreach (var col in bxcols)
        {
            if (col.gameObject.activeInHierarchy)
                activeColliders.Add(col);
        }

        Vector3[] worldCorners = new Vector3[activeColliders.Count * 8];
        var extentPoints = GetScreenCorners(cam, activeColliders, ref worldCorners);

        return GetGUIRectFromExtents(extentPoints, worldCorners, cam);
    }

    private static Rect GetGUIRectFromExtents(Vector2[] extentPoints, Vector3[] worldCorners, Camera cam)
    {
        //screen coordinate - Screen space is 0,0 at bottom left.
        Vector2 min = extentPoints[0];
        Vector2 max = extentPoints[0];

        int inFOVCount = 0;
        for (int i = 0; i < extentPoints.Length; i++)
        {
            if (inFOVCount == 0 && worldCorners[i].InFOV(cam))
                inFOVCount++;

            min = Vector2.Min(min, extentPoints[i]);
            max = Vector2.Max(max, extentPoints[i]);
        }

        //from screen to GUI
        float box_y_min = cam.pixelHeight - max.y;

        if (inFOVCount == 0 ||
            min.x <= 0 && box_y_min <= 0 && max.x >= AngelARUI.Instance.ARCamera.pixelWidth && max.y >= AngelARUI.Instance.ARCamera.pixelHeight)
            return Rect.zero;

        //GUI coordinates
        return new Rect(min.x, box_y_min, max.x - min.x, max.y - min.y);
    }

    #endregion

    #region Vector3 and Transform Extensions

    public static void SetLayerAllChildren(this Transform root, int layer)
    {
        var children = root.GetComponentsInChildren<Transform>(includeInactive: true);
        foreach (var child in children)
            child.gameObject.layer = layer;
    }

    /// <summary>
    /// TODO
    /// </summary>
    /// <param name="root"></param>
    /// <param name="component"></param>
    public static Transform[] GetAllDescendents(this Transform root)
    {
        return root.GetComponentsInChildren<Transform>(includeInactive: true);
    }

    /// <summary>
    /// Returns true if given position is close to zero, given a threshold
    /// </summary>
    /// <param name="position"></param>
    /// <param name="thres"></param>
    /// <returns></returns>
    public static bool IsCloseToZero(this Vector3 position, float thres)
    {
        return position.x < thres && position.y < thres && position.z < thres &&
            position.x > -thres && position.y > -thres && position.z > -thres;
    }

    /// <summary>
    /// Get the left handed vector from the given right handed vector
    /// (for now, the -z is inverted)
    /// </summary>
    /// <param name="rightHandedVector"></param>
    /// <returns></returns>
    public static Vector3 ConvertRightHandedToLeftHanded(this Vector3 rightHandedVector)
    {
        return new Vector3(rightHandedVector.x, rightHandedVector.y, -rightHandedVector.z);
    }

    /// <summary>
    /// Check if a 3D point is in the camera's field of view.
    /// Returns true if point is, else false.
    /// NOTE: does not check for the extent of the object at this point
    /// </summary>
    /// <param name="cam">the camera reference</param>
    /// <param name="point">the point that we check if this is in FOV of cam</param>
    /// <returns></returns>
    public static bool InFOV(this Vector3 point, Camera cam)
    {
        Vector3 viewPos = cam.WorldToViewportPoint(point);
        return (viewPos.x >= 0 && viewPos.x <= 1 && viewPos.y >= 0 && viewPos.y <= 1 && viewPos.z > 0);
    }

    /// <summary>
    /// Check if position of given transform is in front of given camera cam
    /// </summary>
    /// <param name="t"></param>
    /// <param name="cam"></param>
    /// <returns>true if t is in front of camera, else false</returns>
    public static bool InFrontOfCamera(this Transform t, Camera cam)
    {
        return Vector3.Dot(cam.transform.forward, t.position - cam.transform.position) > 0;
    }

    /// <summary>
    /// Returns the distance from the camera to the spatial mesh created by hololens2 towards a given target position
    /// </summary>
    /// <param name="target"></param>
    /// <returns></returns>
    public static float GetCameraToPosDist(this Vector3 target)
    {
        int layerMask = 1 << StringResources.LayerToInt(StringResources.spatialAwareness_layer);

        RaycastHit hit;
        // Does the ray intersect any objects excluding the player layer
        if (Physics.Raycast(AngelARUI.Instance.ARCamera.transform.position, target - AngelARUI.Instance.transform.position, out hit, Mathf.Infinity, layerMask))
            return Mathf.Abs(hit.distance);

        return -1;
    }

    public static Vector3 GetWorldIntersectPoint(this Vector3 pos)
    {
        int layerMask = 1 << StringResources.LayerToInt(StringResources.spatialAwareness_layer);

        RaycastHit hit;
        if (Physics.Raycast(pos, Vector3.down, out hit, Mathf.Infinity, layerMask))
            return hit.point;
        return Vector3.zero;
    }

    public static void SetXPos(this Transform t, float value)
    {
        Vector3 v = t.position;
        v.x = value;
        t.position = v;
    }

    public static void SetYPos(this Transform t, float value)
    {
        Vector3 v = t.position;
        v.y = value;
        t.position = v;
    }

    public static void SetZPos(this Transform t, float value)
    {
        Vector3 v = t.position;
        v.z = value;
        t.position = v;
    }

    public static void SetLocalXPos(this Transform t, float value)
    {
        Vector3 v = t.localPosition;
        v.x = value;
        t.localPosition = v;
    }

    public static void SetLocalYPos(this Transform t, float value)
    {
        Vector3 v = t.localPosition;
        v.y = value;
        t.localPosition = v;
    }

    public static void SetLocalZPos(this Transform t, float value)
    {
        Vector3 v = t.localPosition;
        v.z = value;
        t.localPosition = v;
    }

    public static void SetXScale(this Transform t, float value)
    {
        Vector3 v = t.localScale;
        v.x = value;
        t.localScale = v;
    }

    public static void SetYScale(this Transform t, float value)
    {
        Vector3 v = t.localScale;
        v.y = value;
        t.localScale = v;
    }

    public static void SetZScale(this Transform t, float value)
    {
        Vector3 v = t.localScale;
        v.z = value;
        t.localScale = v;
    }

    #endregion
}