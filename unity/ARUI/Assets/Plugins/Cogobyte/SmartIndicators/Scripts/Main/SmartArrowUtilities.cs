using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators.SmartArrowUtilities
{
    /// <summary>
    /// Class for ear clipping algorithm triangulation.
    /// </summary>
    public class EarClipping {
        /// <summary>
        /// Struct class for vertices wotj convex information and orginal vertice reference.
        /// </summary>
        public class EarVertex
        {
            /// <summary>
            /// Vertex coordinate.
            /// </summary>
            public readonly Vector3 position;
            /// <summary>
            /// Position in the original vertex list.
            /// </summary>
            public readonly int index;
            /// <summary>
            /// Is the vertex convex with adjecent vertices.
            /// </summary>
            public bool isConvex = false;
            public EarVertex(Vector3 position, int index, bool isConvex)
            {
                this.position = position;
                this.index = index;
                this.isConvex = isConvex;
            }
        }

        public class ColoredVertex{
            public ColoredVertex(Vector3 pos,Color32 col)
            {
                position = pos;
                color = col;
            }
            public Vector3 position;
            public Color32 color;
        }

        //Calculate which side of plane the point is on
        //-1 is that point lies on plane, 0 for normal side and 1 for counter normal side
        public static int CalculatePointSideOfPlane(Vector3 point, Vector3 planePoint, Vector3 planeNormal)
        {
            float val = Vector3.Dot(point, planeNormal) - Vector3.Dot(planePoint, planeNormal);
            if (Mathf.Abs(Utilities.errorRate - val) < Utilities.errorRate) return -1;
            if (val > 0) return 0;
            return 1;
        }

        public class HoleShape {
            public List<Vector3> holeVertices;
            public float maxX = 0;
            public int maxIndex;
        }
        public static void TriangulateFaceWithHoles(List<int> tris, List<ColoredVertex> points, List<List<ColoredVertex>> holes)
        { 
            int holeNum = points.Count;
            LinkedList<EarVertex> unusedPoints = FormList(points);
            List<HoleShape> holeShapes = new List<HoleShape>();
            for(int i = 0; i < holes.Count; i++)
            {
                float max = holes[i][0].position.x;
                int maxIndex = 0;
                holeShapes.Add(new HoleShape());
                holeShapes[i].holeVertices = new List<Vector3>();
                for (int j = 0; j < holes[i].Count; j++)
                {
                    holeShapes[i].holeVertices.Add(holes[i][j].position);
                    if (max < holes[i][j].position.x) { max = holes[i][j].position.x; maxIndex = j;}
                }
                holeShapes[i].maxIndex = maxIndex;
                holeShapes[i].maxX = max;
            }
            holeShapes.Sort((x, y) => (x.maxX > y.maxX) ? -1 : 1);

            foreach (HoleShape holeData in holeShapes)
            {

                // Find first edge which intersects with rightwards ray originating at the hole bridge point.
                Vector2 rayIntersectPoint = new Vector2(float.MaxValue, holeData.holeVertices[holeData.maxIndex].y);
                List<LinkedListNode<EarVertex>> hullNodesPotentiallyInBridgeTriangle = new List<LinkedListNode<EarVertex>>();
                LinkedListNode<EarVertex> initialBridgeNodeOnHull = null;
                LinkedListNode<EarVertex> currentNode = unusedPoints.First;
                while (currentNode != null)
                {
                    LinkedListNode<EarVertex> nextNode = (currentNode.Next == null) ? unusedPoints.First : currentNode.Next;
                    Vector2 p0 = currentNode.Value.position;
                    Vector2 p1 = nextNode.Value.position;

                    // at least one point must be to right of holeData.bridgePoint for intersection with ray to be possible
                    if (p0.x > holeData.holeVertices[holeData.maxIndex].x || p1.x > holeData.holeVertices[holeData.maxIndex].x)
                    {
                        // one point is above, one point is below
                        if (p0.y > holeData.holeVertices[holeData.maxIndex].y != p1.y > holeData.holeVertices[holeData.maxIndex].y)
                        {
                            float rayIntersectX = p1.x; // only true if line p0,p1 is vertical
                            if (!Mathf.Approximately(p0.x, p1.x))
                            {
                                float intersectY = holeData.holeVertices[holeData.maxIndex].y;
                                float gradient = (p0.y - p1.y) / (p0.x - p1.x);
                                float c = p1.y - gradient * p1.x;
                                rayIntersectX = (intersectY - c) / gradient;
                            }

                            // intersection must be to right of bridge point
                            if (rayIntersectX > holeData.holeVertices[holeData.maxIndex].x)
                            {
                                LinkedListNode<EarVertex> potentialNewBridgeNode = (p0.x > p1.x) ? currentNode : nextNode;
                                // if two intersections occur at same x position this means is duplicate edge
                                // duplicate edges occur where a hole has been joined to the outer polygon
                                bool isDuplicateEdge = Mathf.Approximately(rayIntersectX, rayIntersectPoint.x);

                                // connect to duplicate edge (the one that leads away from the other, already connected hole, and back to the original hull) if the
                                // current hole's bridge point is higher up than the bridge point of the other hole (so that the new bridge connection doesn't intersect).
                                bool connectToThisDuplicateEdge = holeData.holeVertices[holeData.maxIndex].y > potentialNewBridgeNode.Previous.Value.position.y;

                                if (!isDuplicateEdge || connectToThisDuplicateEdge)
                                {
                                    // if this is the closest ray intersection thus far, set bridge hull node to point in line having greater x pos (since def to right of hole).
                                    if (rayIntersectX < rayIntersectPoint.x || isDuplicateEdge)
                                    {
                                        rayIntersectPoint.x = rayIntersectX;
                                        initialBridgeNodeOnHull = potentialNewBridgeNode;
                                    }
                                }
                            }
                        }
                    }

                    // Determine if current node might lie inside the triangle formed by holeBridgePoint, rayIntersection, and bridgeNodeOnHull
                    // We only need consider those which are reflex, since only these will be candidates for visibility from holeBridgePoint.
                    // A list of these nodes is kept so that in next step it is not necessary to iterate over all nodes again.
                    if (currentNode != initialBridgeNodeOnHull)
                    {
                        if (!currentNode.Value.isConvex && p0.x > holeData.holeVertices[holeData.maxIndex].x)
                        {
                            hullNodesPotentiallyInBridgeTriangle.Add(currentNode);
                        }
                    }
                    currentNode = currentNode.Next;
                }

                // Check triangle formed by hullBridgePoint, rayIntersection, and bridgeNodeOnHull.
                // If this triangle contains any points, those points compete to become new bridgeNodeOnHull
                LinkedListNode<EarVertex> validBridgeNodeOnHull = initialBridgeNodeOnHull;
                foreach (LinkedListNode<EarVertex> nodePotentiallyInTriangle in hullNodesPotentiallyInBridgeTriangle)
                {
                    if (nodePotentiallyInTriangle.Value.index == initialBridgeNodeOnHull.Value.index)
                    {
                        continue;
                    }
                    // if there is a point inside triangle, this invalidates the current bridge node on hull.
                    if (PointInTriangle(holeData.holeVertices[holeData.maxIndex], rayIntersectPoint, initialBridgeNodeOnHull.Value.position, nodePotentiallyInTriangle.Value.position))
                    {
                        // Duplicate points occur at hole and hull bridge points.
                        bool isDuplicatePoint = validBridgeNodeOnHull.Value.position == nodePotentiallyInTriangle.Value.position;

                        // if multiple nodes inside triangle, we want to choose the one with smallest angle from holeBridgeNode.
                        // if is a duplicate point, then use the one occurring later in the list
                        float currentDstFromHoleBridgeY = Mathf.Abs(holeData.holeVertices[holeData.maxIndex].y - validBridgeNodeOnHull.Value.position.y);
                        float pointInTriDstFromHoleBridgeY = Mathf.Abs(holeData.holeVertices[holeData.maxIndex].y - nodePotentiallyInTriangle.Value.position.y);

                        if (pointInTriDstFromHoleBridgeY < currentDstFromHoleBridgeY || isDuplicatePoint)
                        {
                            validBridgeNodeOnHull = nodePotentiallyInTriangle;

                        }
                    }
                }

                // Insert hole points (starting at holeBridgeNode) into vertex list at validBridgeNodeOnHull
                currentNode = validBridgeNodeOnHull;
                for (int i = holeData.maxIndex; i <= holeData.holeVertices.Count + holeData.maxIndex; i++)
                {
                    Vector3 previousIndex = currentNode.Value.position;
                    Vector3 currentIndex = holeData.holeVertices[i % holeData.holeVertices.Count];
                    Vector3 nextIndex = holeData.holeVertices[(i + 1) % holeData.holeVertices.Count];

                    if (i == holeData.holeVertices.Count + holeData.maxIndex) // have come back to starting point
                    {
                        nextIndex = validBridgeNodeOnHull.Value.position; 
                    }

                    bool vertexIsConvex = IsConvex(previousIndex, currentIndex, nextIndex);
                    EarVertex holeVertex = new EarVertex(currentIndex, holeNum + i % holeData.holeVertices.Count, vertexIsConvex);
                    currentNode = unusedPoints.AddAfter(currentNode, holeVertex);
                }

                // Add duplicate hull bridge vert now that we've come all the way around. Also set its concavity
                Vector2 nextVertexPos = (currentNode.Next == null) ? unusedPoints.First.Value.position : currentNode.Next.Value.position;
                bool isConvex = IsConvex(holeData.holeVertices[holeData.maxIndex], validBridgeNodeOnHull.Value.position, nextVertexPos);
                EarVertex repeatStartHullVert = new EarVertex(validBridgeNodeOnHull.Value.position, validBridgeNodeOnHull.Value.index, isConvex);
                unusedPoints.AddAfter(currentNode, repeatStartHullVert);

                //Set concavity of initial hull bridge vert, since it may have changed now that it leads to hole vert
                LinkedListNode<EarVertex> nodeBeforeStartBridgeNodeOnHull = (validBridgeNodeOnHull.Previous == null) ? unusedPoints.Last : validBridgeNodeOnHull.Previous;
                LinkedListNode<EarVertex> nodeAfterStartBridgeNodeOnHull = (validBridgeNodeOnHull.Next == null) ? unusedPoints.First : validBridgeNodeOnHull.Next;
                validBridgeNodeOnHull.Value.isConvex = IsConvex(nodeBeforeStartBridgeNodeOnHull.Value.position, validBridgeNodeOnHull.Value.position, nodeAfterStartBridgeNodeOnHull.Value.position);
                holeNum += holeData.holeVertices.Count;
            }
            EarClip(tris, unusedPoints);
        }
        public static LinkedList<EarVertex> FormList(List<ColoredVertex> points)
        {
            LinkedList<EarVertex> unusedPoints = new LinkedList<EarVertex>();
            LinkedListNode<EarVertex> currentNode = null;
            for (int i = 0; i < points.Count; i++)
            {
                int prevPointIndex = (i - 1 + points.Count) % points.Count;
                int nextPointIndex = (i + 1) % points.Count;
                bool vertexIsConvex = IsConvex(points[prevPointIndex].position, points[i].position, points[nextPointIndex].position);
                if (currentNode == null)
                {
                    unusedPoints.AddFirst(new EarVertex(points[i].position, i, vertexIsConvex));
                    currentNode = unusedPoints.First;
                }
                else
                {
                    unusedPoints.AddAfter(currentNode, new EarVertex(points[i].position, i, vertexIsConvex));
                    currentNode = currentNode.Next;
                }
            }
            return unusedPoints;
        }
        public static void TriangulateFace(List<int> tris,List<ColoredVertex> points)
        {
            LinkedList<EarVertex>  unusedPoints = FormList(points);
            EarClip(tris, unusedPoints);
        }
        public static void EarClip(List<int> tris, LinkedList<EarVertex> unusedPoints)
        {
            while (unusedPoints.Count >= 3)
            {
                bool hasRemovedEarThisIteration = false;
                LinkedListNode<EarVertex> vertexNode = unusedPoints.First;
                for (int i = 0; i < unusedPoints.Count; i++)
                {
                    LinkedListNode<EarVertex> prevVertexNode = vertexNode.Previous ?? unusedPoints.Last;
                    LinkedListNode<EarVertex> nextVertexNode = vertexNode.Next ?? unusedPoints.First;

                    if (vertexNode.Value.isConvex)
                    {
                        if (!IsEar(unusedPoints,prevVertexNode.Value, vertexNode.Value, nextVertexNode.Value))
                        {
                            // check if removal of ear makes prev/next vertex convex (if was previously reflex)
                            if (!prevVertexNode.Value.isConvex)
                            {
                                LinkedListNode<EarVertex> prevOfPrev = prevVertexNode.Previous ?? unusedPoints.Last;

                                prevVertexNode.Value.isConvex = IsConvex(prevOfPrev.Value.position, prevVertexNode.Value.position, nextVertexNode.Value.position);
                            }
                            if (!nextVertexNode.Value.isConvex)
                            {
                                LinkedListNode<EarVertex> nextOfNext = nextVertexNode.Next ?? unusedPoints.First;
                                nextVertexNode.Value.isConvex = IsConvex(prevVertexNode.Value.position, nextVertexNode.Value.position, nextOfNext.Value.position);
                            }
                            tris.Add(nextVertexNode.Value.index);
                            tris.Add(vertexNode.Value.index);
                            tris.Add(prevVertexNode.Value.index);
                            hasRemovedEarThisIteration = true;
                            unusedPoints.Remove(vertexNode);
                            break;
                        }
                    }
                    vertexNode = nextVertexNode;
                }

                if (!hasRemovedEarThisIteration)
                {
                    Debug.LogError("Error triangulating mesh. Aborted.");
                    return;
                }
            }
        }
        //Ear test for ear clipping algorithm
        static bool IsEar(LinkedList<EarVertex> unusedPoints, EarVertex a, EarVertex b, EarVertex c)
        {
            LinkedListNode<EarVertex> currentNode = unusedPoints.First;
            for (int i = 0; i < unusedPoints.Count; i++)
            {
                if (!currentNode.Value.isConvex) // convex verts will never be inside triangle
                {
                    if (currentNode.Value.index != a.index && currentNode.Value.index != b.index && currentNode.Value.index != c.index)
                    {
                        if (PointInTriangle(a.position, b.position, c.position, currentNode.Value.position)) return true;
                    }
                }
                currentNode = currentNode.Next;
            }
            return false;
        }
        //Convex test for ear clipping algorithm
        public static bool IsConvex(Vector3 a, Vector3 b, Vector3 c)
        {
            return Mathf.Sign((a.x * (c.y - b.y)) + (b.x * (a.y - c.y)) + (c.x * (b.y - a.y))) < 0;
        }
        //Point in triangle test for ear clipping algorithm
        public static bool PointInTriangle(Vector3 a, Vector3 b, Vector3 c, Vector3 p)
        {
            float area = 0.5f * (-b.y * c.x + a.y * (-b.x + c.x) + a.x * (b.y - c.y) + b.x * c.y);
            float s = 1 / (2 * area) * (a.y * c.x - a.x * c.y + (c.y - a.y) * p.x + (a.x - c.x) * p.y);
            float t = 1 / (2 * area) * (a.x * b.y - a.y * b.x + (a.y - b.y) * p.x + (b.x - a.x) * p.y);
            return s >= 0 && t >= 0 && (s + t) <= 1;
        }
    }

    /// <summary>
    /// Helper math functions for procedural indicators calcualations.
    /// </summary>
    public class Utilities
    {
        /// <summary>
        /// Floating point tolerance for all smart arrow scripts.
        /// </summary>
        public static float errorRate = 0.00001f;
        /// <summary>
        /// World precision of position floats for position rounding.
        /// </summary>
        public static int worldPositionPrecision = 10000;
        /// <summary>
        /// Round position to world position precition.
        /// </summary>
        /// <param name="position">Vector3 to round</param>
        /// <returns>Rounded vector</returns>
        public static Vector3 RoundVector(Vector3 position)
        {
            return new Vector3(Mathf.Round(position.x * worldPositionPrecision) / worldPositionPrecision,
                                Mathf.Round(position.y * worldPositionPrecision) / worldPositionPrecision,
                                Mathf.Round(position.z * worldPositionPrecision) / worldPositionPrecision);
        }
        /// <summary>
        /// Finds intersection points of line and a sphere.
        /// </summary>
        /// <param name="center">Center of the sphere.</param>
        /// <param name="radius">Radius of the sphere.</param>
        /// <param name="lineStartPoint">Start point of the line.</param>
        /// <param name="lineEndPoint">End point of the line.</param>
        /// <returns>Array of intersection points.</returns>
        public static Vector3[] LineSphereIntersection(Vector3 center, float radius, Vector3 lineStartPoint, Vector3 lineEndPoint)
        {
            Vector3 directionRay = lineEndPoint - lineStartPoint;
            Vector3 centerToRayStart = lineStartPoint - center;

            float a = Vector3.Dot(directionRay, directionRay);
            float b = 2 * Vector3.Dot(centerToRayStart, directionRay);
            float c = Vector3.Dot(centerToRayStart, centerToRayStart) - (radius * radius);

            float discriminant = (b * b) - (4 * a * c);
            if (discriminant >= 0)
            {
                //Ray did not miss
                discriminant = Mathf.Sqrt(discriminant);

                //How far on ray the intersections happen
                float t1 = (-b - discriminant) / (2 * a);
                float t2 = (-b + discriminant) / (2 * a);

                Vector3[] hitPoints;

                if (t1 >= 0 && t2 >= 0)
                {
                    //total intersection, return both points
                    hitPoints = new Vector3[2];
                    hitPoints[0] = lineStartPoint + (directionRay * t1);
                    hitPoints[1] = lineStartPoint + (directionRay * t2);
                }
                else
                {
                    //Only one intersected, return one point
                    hitPoints = new Vector3[1];
                    if (t1 >= 0)
                    {
                        hitPoints[0] = lineStartPoint + (directionRay * t1);
                    }
                    else if (t2 >= 0)
                    {
                        hitPoints[0] = lineStartPoint + (directionRay * t2);
                    }
                }
                return hitPoints;
            }
            //No hits
            return null;
        }
    }

    /// <summary>
    /// List of functions for bezier calculations
    /// </summary>
    public static class Bezier
    {
        /// <summary>
        /// Calculate position within bezier spline for given path percentage.
        /// </summary>
        /// <param name="p0">Start bezier point.</param>
        /// <param name="p1">Bezier control point.</param>
        /// <param name="p2">End bezier point.</param>
        /// <param name="t">Percentage on the curve.</param>
        /// <returns>Bezier spline point at percentage.</returns>
        public static Vector3 GetPoint(Vector3 p0, Vector3 p1, Vector3 p2, float t)
        {
            t = Mathf.Clamp01(t);
            float oneMinusT = 1f - t;
            return
                oneMinusT * oneMinusT * p0 +
                2f * oneMinusT * t * p1 +
                t * t * p2;
        }
        /// <summary>
        /// Calculate first derivative of bezier spline for given path percentage.
        /// </summary>
        /// <param name="p0">Start bezier point.</param>
        /// <param name="p1">Bezier control point.</param>
        /// <param name="p2">End bezier point.</param>
        /// <param name="t">Percentage on the curve.</param>
        /// <returns>Bezier spline derivative at percentage.</returns>
        public static Vector3 GetFirstDerivative(Vector3 p0, Vector3 p1, Vector3 p2, float t)
        {
            return
                2f * (1f - t) * (p1 - p0) +
                2f * t * (p2 - p1);
        }
        /// <summary>
        /// alculate position within bezier spline with two control points for given path percentage.
        /// </summary>
        /// <param name="p0">Start bezier point.</param>
        /// <param name="p1">First bezier control point.</param>
        /// <param name="p2">Second bezier control point.</param>
        /// <param name="p3">End bezier point.</param>
        /// <param name="t">Percentage on the curve.</param>
        /// <returns>Bezier spline point at percentage.</returns>
        public static Vector3 GetPoint(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t)
        {
            t = Mathf.Clamp01(t);
            float OneMinusT = 1f - t;
            return
                OneMinusT * OneMinusT * OneMinusT * p0 +
                3f * OneMinusT * OneMinusT * t * p1 +
                3f * OneMinusT * t * t * p2 +
                t * t * t * p3;
        }
        /// <summary>
        /// Calculate first derivative of bezier spline for given path percentage.
        /// </summary>
        /// <param name="p0">Start bezier point.</param>
        /// <param name="p1">First bezier control point.</param>
        /// <param name="p2">Second bezier control point.</param>
        /// <param name="p3">End bezier point.</param>
        /// <param name="t">Percentage on the curve.</param>
        /// <returns>Bezier spline derivative at percentage.</returns>
        public static Vector3 GetFirstDerivative(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t)
        {
            t = Mathf.Clamp01(t);
            float oneMinusT = 1f - t;
            return
                3f * oneMinusT * oneMinusT * (p1 - p0) +
                6f * oneMinusT * t * (p2 - p1) +
                3f * t * t * (p3 - p2);
        }
    }
}