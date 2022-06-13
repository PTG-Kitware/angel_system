using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    /// <summary>
    /// A 2D outline edge that can be extruded to a 3D mesh. 
    /// </summary>
    [System.Serializable]
    public class OutlineEdge
    {
        /// <summary>
        /// Mesh filter index to be rendered to.
        /// </summary>
        [Min(0)]
        public int meshIndex = 0;
        /// <summary>
        /// List of all edge points, ordered from first point to end point.
        /// </summary>
        public List<OutlineEdgePoint> points = new List<OutlineEdgePoint>() {
            new OutlineEdgePoint() {position=new Vector2(-0.5f,0)},
            new OutlineEdgePoint() {position=new Vector2(0.5f,0)}
        };
        /// <summary>
        /// Edge default color of all points.
        /// </summary>
        public Color32 color = Color.white;
        public OutlineEdge(){}
        public OutlineEdge(OutlineEdge edge)
        {
            meshIndex = edge.meshIndex;
            points = new List<OutlineEdgePoint>();
            foreach(OutlineEdgePoint p in edge.points)
            {
                points.Add(new OutlineEdgePoint(p));
            }
            color = edge.color;
        }
        /// <summary>
        /// Auto calculates normals for all points. Start and end point normals are perpendicular and others are average of their neighbour normals to make a smooth curve.
        /// </summary>
        public void CaluclateNormals()
        {
            Vector3 lastnorm = Vector3.Cross(new Vector3((points[0].position - points[1].position).x, (points[0].position - points[1].position).y, 0), Vector3.forward).normalized;
            for (int i = 0; i < points.Count - 1; i++)
            {
                Vector3 n = Vector3.Cross(new Vector3((points[i].position - points[i + 1].position).x, (points[i].position - points[i + 1].position).y, 0), Vector3.forward).normalized;
                points[i].normal = (new Vector2(n.x, n.y) + new Vector2(lastnorm.x, lastnorm.y)).normalized;
                lastnorm = n;
            }
            points[points.Count - 1].normal = Vector3.Cross(new Vector3((points[points.Count - 2].position - points[points.Count - 1].position).x, (points[points.Count - 2].position - points[points.Count - 1].position).y, 0), Vector3.forward).normalized;
            if (Vector2.Distance(points[0].position, points[points.Count - 1].position) < 0.0001)
            {
                points[0].normal = points[points.Count - 1].normal = (points[points.Count - 1].normal + points[0].normal).normalized;
            }
        }
        /// <summary>
        /// Sets the color of all points to input color.
        /// </summary>
        /// <param name="color">Color to set all points to.</param>
        public void SetPointsColors(Color32 color)
        {
            foreach (OutlineEdgePoint p in points)
            {
                p.color = color;
            }
        }
        /// <summary>
        /// Sets color of all points from start point to end point using a gradient.
        /// </summary>
        /// <param name="gradient">Color range from start point to end point.</param>
        public void SetPointsColors(Gradient gradient)
        {
            points[0].color = gradient.Evaluate(0);
            float edgeLength = 0;
            for(int i = 1; i < points.Count; i++)
            {
                edgeLength += (points[i].position - points[i-1].position).magnitude;
            }
            float travelEdgeLength = 0;
            for (int i = 1; i < points.Count; i++)
            {
                travelEdgeLength += (points[i].position - points[i - 1].position).magnitude;
                points[i].color = gradient.Evaluate(travelEdgeLength/edgeLength);
            }
        }
        /// <summary>
        /// Flips the x component and order of all points, flipping the edge left right.
        /// </summary>
        public void FlipHorizontal()
        {
            foreach (OutlineEdgePoint p in points)
            {
                p.position.x = -p.position.x;
                p.normal.x = -p.normal.x;
            }
            points.Reverse();
        }
        /// <summary>
        /// Flips the y component and order of all points, turning the edge upside down.
        /// </summary>
        public void FlipVertical()
        {
            foreach (OutlineEdgePoint p in points)
            {
                p.position.y = -p.position.y;
                p.normal.y = -p.normal.y;
            }
            points.Reverse();
        }
        /// <summary>
        /// Swaps x and y on all points which turns horizontal edge into vertical and vice versa.
        /// </summary>
        public void FlipAxes()
        {
            foreach (OutlineEdgePoint p in points)
            {
                float temp = p.position.x;
                p.position.x = p.position.y;
                p.position.y = temp;
                temp = p.normal.x;
                p.normal.x = p.normal.y;
                p.normal.y = temp;
            }
        }
    }
}