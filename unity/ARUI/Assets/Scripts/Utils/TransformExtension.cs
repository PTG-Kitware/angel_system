using UnityEngine;

public static class TransformExtension
{
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