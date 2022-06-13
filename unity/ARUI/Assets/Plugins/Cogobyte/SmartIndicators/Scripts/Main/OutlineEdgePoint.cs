using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    /// <summary>
    /// Outline point 2D information. Outline gets rendered into a 3D vertex.
    /// </summary>
    [System.Serializable]
    public class OutlineEdgePoint
    {
        public Vector2 position = Vector2.zero;
        public Vector2 normal = Vector2.up;
        public Color32 color = Color.white;
        public OutlineEdgePoint() {
        }
        public OutlineEdgePoint(OutlineEdgePoint point)
        {
            position = point.position;
            normal = point.normal;
            color = point.color;
        }
    }
}