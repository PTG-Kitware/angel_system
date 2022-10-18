using System;
using UnityEngine;

public static class TransformExtension
{
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
        var extentPoints = GetCorners(t, cam, bxcol, ref worldCorners);

        Vector2 min = extentPoints[0];
        Vector2 max = extentPoints[0];

        int inFOVCount = 0;
        for (int i = 0; i < extentPoints.Length; i++)
        {
            if (inFOVCount==0 && Utils.InFOV(cam, worldCorners[i])) {
                inFOVCount++;
            }
            min = Vector2.Min(min, extentPoints[i]);
            max = Vector2.Max(max, extentPoints[i]);
        }

        if (inFOVCount == 0)
            return Rect.zero;

        float box_y_min = cam.pixelHeight - max.y;
        //return new Rect(min.x, box_y_min, max.x - min.x, max.y - min.y);

        return new Rect(min.x, box_y_min, max.x - min.x, max.y - min.y);

    }

    public static Rect RectFromHands(this Transform t, Camera cam, Bounds bounds)
    {
        var extentPoints = GetCorners(t, cam,bounds);

        Vector2 min = extentPoints[0];
        Vector2 max = extentPoints[0];
        foreach (Vector2 v in extentPoints)
        {
            min = Vector2.Min(min, v);
            max = Vector2.Max(max, v);
        }

        //gui to screen
        float box_y_min = cam.pixelHeight - max.y;
        return new Rect(min.x, box_y_min, max.x - min.x, max.y - min.y);

    }

    ////https://stackoverflow.com/questions/65417634/draw-bounding-rectangle-screen-space-around-a-game-object-with-a-renderer-wor
    //private static Rect RendererBoundsInScreenSpace(Bounds bounds)
    //{
    //    // This is the space occupied by the object's visuals
    //    // in WORLD space.
    //    Bounds bigBounds = bounds;

    //    Vector3[] screenSpaceCorners = new Vector3[8];

    //    Camera theCamera = Camera.main;

    //    float scaleValue = 0.9f;

    //    // For each of the 8 corners of our renderer's world space bounding box,
    //    // convert those corners into screen space.
    //    screenSpaceCorners[0] = theCamera.WorldToScreenPoint(new Vector3(bigBounds.center.x + bigBounds.extents.x*scaleValue, bigBounds.center.y + bigBounds.extents.y * scaleValue, bigBounds.center.z + bigBounds.extents.z * scaleValue));
    //    screenSpaceCorners[1] = theCamera.WorldToScreenPoint(new Vector3(bigBounds.center.x + bigBounds.extents.x * scaleValue, bigBounds.center.y + bigBounds.extents.y * scaleValue, bigBounds.center.z - bigBounds.extents.z * scaleValue));
    //    screenSpaceCorners[2] = theCamera.WorldToScreenPoint(new Vector3(bigBounds.center.x + bigBounds.extents.x * scaleValue, bigBounds.center.y - bigBounds.extents.y * scaleValue, bigBounds.center.z + bigBounds.extents.z * scaleValue));
    //    screenSpaceCorners[3] = theCamera.WorldToScreenPoint(new Vector3(bigBounds.center.x + bigBounds.extents.x * scaleValue, bigBounds.center.y - bigBounds.extents.y * scaleValue, bigBounds.center.z - bigBounds.extents.z * scaleValue));
    //    screenSpaceCorners[4] = theCamera.WorldToScreenPoint(new Vector3(bigBounds.center.x - bigBounds.extents.x * scaleValue, bigBounds.center.y + bigBounds.extents.y * scaleValue, bigBounds.center.z + bigBounds.extents.z * scaleValue));
    //    screenSpaceCorners[5] = theCamera.WorldToScreenPoint(new Vector3(bigBounds.center.x - bigBounds.extents.x * scaleValue, bigBounds.center.y + bigBounds.extents.y * scaleValue, bigBounds.center.z - bigBounds.extents.z * scaleValue));
    //    screenSpaceCorners[6] = theCamera.WorldToScreenPoint(new Vector3(bigBounds.center.x - bigBounds.extents.x * scaleValue, bigBounds.center.y - bigBounds.extents.y * scaleValue, bigBounds.center.z + bigBounds.extents.z * scaleValue));
    //    screenSpaceCorners[7] = theCamera.WorldToScreenPoint(new Vector3(bigBounds.center.x - bigBounds.extents.x * scaleValue, bigBounds.center.y - bigBounds.extents.y * scaleValue, bigBounds.center.z - bigBounds.extents.z * scaleValue));

    //    // Now find the min/max X & Y of these screen space corners.
    //    float min_x = screenSpaceCorners[0].x;
    //    float min_y = screenSpaceCorners[0].y;
    //    float max_x = screenSpaceCorners[0].x;
    //    float max_y = screenSpaceCorners[0].y;

    //    for (int i = 1; i < 8; i++)
    //    {
    //        if (screenSpaceCorners[i].x < min_x)
    //        {
    //            min_x = screenSpaceCorners[i].x;
    //        }
    //        if (screenSpaceCorners[i].y < min_y)
    //        {
    //            min_y = screenSpaceCorners[i].y;
    //        }
    //        if (screenSpaceCorners[i].x > max_x)
    //        {
    //            max_x = screenSpaceCorners[i].x;
    //        }
    //        if (screenSpaceCorners[i].y > max_y)
    //        {
    //            max_y = screenSpaceCorners[i].y;
    //        }
    //    }

    //    return Rect.MinMaxRect(min_x, min_y, max_x, max_y);

    //}

    private static Vector2[] GetCorners(Transform t, Camera cam, BoxCollider bxcol, ref Vector3[] worldCorners)
    {
        float scalingValue = 0.9f;

        worldCorners[0] = t.TransformPoint(bxcol.center + (new Vector3(-bxcol.size.x * scalingValue, -bxcol.size.y * scalingValue, -bxcol.size.z * scalingValue) * 0.5f));
        worldCorners[1] = t.TransformPoint(bxcol.center + (new Vector3(bxcol.size.x * scalingValue, -bxcol.size.y * scalingValue, -bxcol.size.z * scalingValue) * 0.5f));
        worldCorners[2] = t.TransformPoint(bxcol.center + (new Vector3(bxcol.size.x * scalingValue, -bxcol.size.y * scalingValue, bxcol.size.z * scalingValue) * 0.5f));
        worldCorners[3] = t.TransformPoint(bxcol.center + (new Vector3(-bxcol.size.x * scalingValue, -bxcol.size.y * scalingValue, bxcol.size.z * scalingValue) * 0.5f));
        worldCorners[4] = t.TransformPoint(bxcol.center + (new Vector3(-bxcol.size.x * scalingValue, bxcol.size.y * scalingValue, -bxcol.size.z * scalingValue) * 0.5f));
        worldCorners[5] = t.TransformPoint(bxcol.center + (new Vector3(bxcol.size.x * scalingValue, bxcol.size.y * scalingValue, -bxcol.size.z * scalingValue) * 0.5f));
        worldCorners[6] = t.TransformPoint(bxcol.center + (new Vector3(bxcol.size.x * scalingValue, bxcol.size.y * scalingValue, bxcol.size.z * scalingValue) * 0.5f));
        worldCorners[7] = t.TransformPoint(bxcol.center + (new Vector3(-bxcol.size.x * scalingValue, bxcol.size.y * scalingValue, bxcol.size.z * scalingValue) * 0.5f));
        return new Vector2[]
        {
                cam.WorldToScreenPoint(worldCorners[0]),
                cam.WorldToScreenPoint(worldCorners[1]),
                cam.WorldToScreenPoint(worldCorners[2]),
                cam.WorldToScreenPoint(worldCorners[3]),
                cam.WorldToScreenPoint(worldCorners[4]),
                cam.WorldToScreenPoint(worldCorners[5]),
                cam.WorldToScreenPoint(worldCorners[6]),
                cam.WorldToScreenPoint(worldCorners[7])
        };
    }


    private static Vector2[] GetCorners(Transform t, Camera cam, Bounds bxcol)
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