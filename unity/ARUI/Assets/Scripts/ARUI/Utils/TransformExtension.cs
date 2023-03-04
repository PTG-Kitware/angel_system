using System;
using System.Reflection;
using UnityEngine;

public static class TransformExtension
{
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
    /// https://forum.unity.com/threads/how-do-i-get-the-oriented-bounding-box-values.538239/
    /// </summary>
    /// <param name="t"></param>
    /// <param name="cam"></param>
    /// <returns></returns>
    public static Rect RectFromObj(this Transform t, Camera cam, BoxCollider bxcol)
    {
        Vector3[] worldCorners = new Vector3[8];
        var extentPoints = GetScreenCorners(new Transform[1] { t }, cam, new BoxCollider[1] { bxcol }, ref worldCorners);

        return GetGUIRectFromExtents(extentPoints, worldCorners, cam);
    }

    /// <summary>
    /// https://forum.unity.com/threads/how-do-i-get-the-oriented-bounding-box-values.538239/
    /// </summary>
    /// <param name="t"></param>
    /// <param name="cam"></param>
    /// <returns>Rect of given collider in </returns>
    public static Rect RectFromObjs(this Transform t1, Camera cam, BoxCollider[] bxcols, Transform[] transforms)
    {
        Vector3[] worldCorners = new Vector3[16];
        var extentPoints = GetScreenCorners(transforms, cam, bxcols, ref worldCorners);

        return GetGUIRectFromExtents(extentPoints, worldCorners, cam);
    }

    public static Rect RectFromHands(this Transform t, Camera cam, Bounds bounds)
    {
        var extentPoints = GetScreenCorners(t, cam, bounds);

        return GetGUIRectFromExtents(extentPoints, cam);

    }

    private static Rect GetGUIRectFromExtents(Vector2[] extentPoints, Camera cam)
    {
        Vector2 min = extentPoints[0];
        Vector2 max = extentPoints[0];
        foreach (Vector2 v in extentPoints)
        {
            min = Vector2.Min(min, v);
            max = Vector2.Max(max, v);
        }

        //screen to GUI
        float box_y_min = cam.pixelHeight - max.y;
        return new Rect(min.x, box_y_min, max.x - min.x, max.y - min.y);
    }

    private static Rect GetGUIRectFromExtents(Vector2[] extentPoints, Vector3[] worldCorners, Camera cam)
    {
        //screen coordinate - Screen space is 0,0 at bottom left.
        Vector2 min = extentPoints[0];
        Vector2 max = extentPoints[0];

        int inFOVCount = 0;
        for (int i = 0; i < extentPoints.Length; i++)
        {
            if (inFOVCount == 0 && Utils.InFOV(cam, worldCorners[i]))
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

    private static Vector2[] GetScreenCorners(Transform[] tr, Camera cam, BoxCollider[] bxcols, ref Vector3[] worldCorners)
    {
        float scalingValue = 1.0f;

        int i=0;
        int index = 0;
        foreach (var item in bxcols)
        {
            Transform current = tr[i]; 
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

        Vector2[] corners = new Vector2[bxcols.Length * 8];
        for (int j = 0; j < corners.Length; j++)
            corners[j] = cam.WorldToScreenPoint(worldCorners[j]);

        return corners;
    }

    private static Vector2[] GetScreenCorners(Transform t, Camera cam, Bounds bxcol)
    {
        return new Vector2[]
        {
                cam.WorldToScreenPoint(t.TransformPoint(bxcol.center + new Vector3(-bxcol.size.x, -bxcol.size.y, -bxcol.size.z) * 0.5f)),
                cam.WorldToScreenPoint(t.TransformPoint(bxcol.center + new Vector3(bxcol.size.x, -bxcol.size.y, -bxcol.size.z) * 0.5f)),
                cam.WorldToScreenPoint(t.TransformPoint(bxcol.center + new Vector3(bxcol.size.x, -bxcol.size.y, bxcol.size.z) * 0.5f)),
                cam.WorldToScreenPoint(t.TransformPoint(bxcol.center + new Vector3(-bxcol.size.x, -bxcol.size.y, bxcol.size.z) * 0.5f)),
                cam.WorldToScreenPoint(t.TransformPoint(bxcol.center + new Vector3(-bxcol.size.x, bxcol.size.y, -bxcol.size.z) * 0.5f)),
                cam.WorldToScreenPoint(t.TransformPoint(bxcol.center + new Vector3(bxcol.size.x, bxcol.size.y, -bxcol.size.z) * 0.5f)),
                cam.WorldToScreenPoint(t.TransformPoint(bxcol.center + new Vector3(bxcol.size.x, bxcol.size.y, bxcol.size.z) * 0.5f)),
                cam.WorldToScreenPoint(t.TransformPoint(bxcol.center + new Vector3(-bxcol.size.x, bxcol.size.y, bxcol.size.z) * 0.5f))
        };
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
}