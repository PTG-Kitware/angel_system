using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    /// <summary>
    ///Triangulated 2D shape outline with a back and front cap (face) for extruding. 
    /// </summary>
    [System.Serializable]
    public class Outline
    {
        /// <summary>
        /// List of lines that get extruded.
        /// </summary>
        public List<OutlineEdge> edges = new List<OutlineEdge>(){new OutlineEdge()};
        /// <summary>
        /// Color of all points in outline.
        /// </summary>
        public Color32 color = Color.white;
        /// <summary>
        /// Back cap mesh.
        /// </summary>
        public List<MeshItem> backFaceMeshes = new List<MeshItem>();
        /// <summary>
        /// Front cap mesh.
        /// </summary>
        public List<MeshItem> frontFaceMeshes = new List<MeshItem>();

        public Outline()
        {
        }
        public Outline(Outline outline)
        {
            CopyOutline(outline);
        }
        /// <summary>
        /// Copy all data from one outline to another.
        /// </summary>
        /// <param name="outline">Source outline.</param>
        public void CopyOutline(Outline outline)
        {
            color = outline.color;
            edges = new List<OutlineEdge>();
            for (int i = 0; i < outline.edges.Count; i++)
            {
                edges.Add(new OutlineEdge(outline.edges[i]));
            }
            backFaceMeshes = new List<MeshItem>();
            for(int i = 0; i < outline.backFaceMeshes.Count; i++)
            {
                backFaceMeshes.Add(new MeshItem(outline.backFaceMeshes[i]));
            }
            frontFaceMeshes = new List<MeshItem>();
            for (int i = 0; i < outline.frontFaceMeshes.Count; i++)
            {
                frontFaceMeshes.Add(new MeshItem(outline.frontFaceMeshes[i]));
            }
        }
        /// <summary>
        /// Sets all point colors to target color.
        /// </summary>
        /// <param name="color">Target color.</param>
        public void SetPointsColors(Color32 color)
        {
            foreach(OutlineEdge e in edges)
            {
                e.SetPointsColors(color);
            }
        }
        /// <summary>
        /// Calculates normals for each edge.
        /// </summary>
        public void CalculateNormals()
        {
            foreach (OutlineEdge e in edges)
            {
                e.CaluclateNormals();
            }
        }
        /// <summary>
        /// Calculates normals for each edge and then averages normals on shared edge points.
        /// </summary>
        public void CalculateSmoothNormals()
        {
            foreach(OutlineEdge e in edges)
            {
                CalculateSmoothNormals(e);
            }
        }
        /// <summary>
        /// Calculates normals for target edge only and then averages first and last normal if they are sharing position with other edge vertices.
        /// </summary>
        /// <param name="targetEdge">Target outline edge within outline.</param>
        public void CalculateSmoothNormals(OutlineEdge targetEdge)
        {
            targetEdge.CaluclateNormals();
            foreach(OutlineEdge testEdge in edges)
            {
                if (Vector2.Distance(testEdge.points[0].position, targetEdge.points[targetEdge.points.Count-1].position) < SmartArrowUtilities.Utilities.errorRate)
                {
                    testEdge.points[0].normal = targetEdge.points[targetEdge.points.Count - 1].normal = (Vector3.Cross(new Vector3((testEdge.points[0].position - testEdge.points[1].position).x, (testEdge.points[0].position - testEdge.points[1].position).y, 0), Vector3.forward).normalized + Vector3.Cross(new Vector3((targetEdge.points[targetEdge.points.Count-2].position - targetEdge.points[targetEdge.points.Count - 1].position).x, (targetEdge.points[targetEdge.points.Count - 2].position - targetEdge.points[targetEdge.points.Count - 1].position).y, 0), Vector3.forward).normalized).normalized;
                }
                if (Vector2.Distance(targetEdge.points[0].position, testEdge.points[testEdge.points.Count-1].position) < SmartArrowUtilities.Utilities.errorRate)
                {
                    testEdge.points[testEdge.points.Count - 1].normal = targetEdge.points[0].normal = (Vector3.Cross(new Vector3((targetEdge.points[0].position - targetEdge.points[1].position).x, (targetEdge.points[0].position - targetEdge.points[1].position).y, 0), Vector3.forward).normalized + Vector3.Cross(new Vector3((testEdge.points[testEdge.points.Count - 2].position - testEdge.points[testEdge.points.Count - 1].position).x, (testEdge.points[testEdge.points.Count - 2].position - testEdge.points[testEdge.points.Count - 1].position).y, 0), Vector3.forward).normalized).normalized;
                }
            }
        }
        /// <summary>
        /// Flips all edge vertices and normals horizontally.
        /// </summary>
        public void FlipHorizontal()
        {
            foreach (OutlineEdge e in edges)
            {
                e.FlipHorizontal();
            }
            edges.Reverse();
        }
        /// <summary>
        /// Flips all edge vertices and normals vertically.
        /// </summary>
        public void FlipVertical()
        {
            foreach (OutlineEdge e in edges)
            {
                e.FlipVertical();
            }
            edges.Reverse();
        }
        /// <summary>
        /// Generates front or back face mesh for source oultine and target mesh item.
        /// </summary>
        /// <param name="meshItem">Target mesh item that contains target mesh.</param>
        /// <param name="outline">Source outline with vertices to triangulate.</param>
        /// <param name="front">Is it a front face or back face.</param>
        public static void GenerateFaceMesh(MeshItem meshItem, Outline outline, bool front)
        {
            if (meshItem.mesh.mesh == null) meshItem.mesh.mesh = new Mesh();
            GenerateFaceMesh(meshItem.mesh.mesh, outline, front);
            meshItem.LoadMesh(meshItem.mesh.mesh);
        }
        /// <summary>
        ///  Generates front or back face mesh for source oultine and target mesh.
        /// </summary>
        /// <param name="mesh">Target mesh to store resulting vertices and triangles.</param>
        /// <param name="outline">Source outline with vertices to triangulate.</param>
        /// <param name="front">Is it a front face or back face.</param>
        public static void GenerateFaceMesh(Mesh mesh,Outline outline, bool front)
        {
            List<Vector3> vertices = new List<Vector3>();
            List<Vector3> normals = new List<Vector3>();
            List<Vector2> uvs = new List<Vector2>();
            List<Color32> colors = new List<Color32>();
            List<int> triangles = new List<int>();
            List<List<SmartArrowUtilities.EarClipping.ColoredVertex>> edges = FormOutlineList(outline);
            int vertCount = 0;
            for (int i = 0; i < edges.Count; i++)
            {
                if (edges[i].Count > 2 && Vector3.Distance(edges[i][edges[i].Count - 1].position, edges[i][0].position) < SmartArrowUtilities.Utilities.errorRate)
                {
                    edges[i].RemoveAt(edges[i].Count - 1);
                }
                edges[i].Reverse();
                for (int k = 0; k < edges[i].Count; k++)
                {
                    vertices.Add(edges[i][k].position);
                    colors.Add(edges[i][k].color);
                }
                if (edges[i].Count > 2)
                {
                    SmartArrowUtilities.EarClipping.TriangulateFace(triangles, edges[i]);
                    for (int t = 0; t < triangles.Count; t += 3)
                    {
                        triangles[t] += vertCount;
                        triangles[t + 1] += vertCount;
                        triangles[t + 2] += vertCount;
                        if (front)
                        {
                            int tempI = triangles[t + 1];
                            triangles[t + 1] = triangles[t + 2];
                            triangles[t + 2] = tempI;
                        }
                    }
                    vertCount += edges[i].Count;
                }
            }
            if (vertices != null)
            {
                for (int i = 0; i < vertices.Count; i++)
                {
                    if (front) normals.Add(Vector3.forward);
                    else normals.Add(Vector3.back);
                    uvs.Add(new Vector2(vertices[i].x, vertices[i].y));
                }
                mesh.Clear();
                mesh.name = (front)?"Front":"Back"+"FaceMesh";
                mesh.SetVertices(vertices);
                mesh.SetNormals(normals);
                mesh.SetColors(colors);
                mesh.SetUVs(0, uvs);
                mesh.SetTriangles(triangles, 0);
                mesh.RecalculateBounds();
                mesh.RecalculateTangents();
                mesh.UploadMeshData(false);
            }
        }
        /// <summary>
        /// Generates vertices and triangles that connect two outlines. Outline edges need to form one shape. Inner edges can form multiple shape holes within outer outline.
        /// </summary>
        /// <param name="connector">Target mesh item container for mesh result.</param>
        /// <param name="outer">Source outer outline</param>
        /// <param name="inner">Source inner outline</param>
        /// <param name="tail">Is it an arrow tail or head.</param>
        public static void GenerateOutlineToOutlineFace(BodyTipMeshItem connector,Outline outer, Outline inner, bool tail)
        {
            if (connector.mesh.mesh == null) connector.mesh.mesh = new Mesh();
            outer = new Outline(outer);
            inner = new Outline(inner);
            if (tail)
            {
                outer.FlipHorizontal();
                inner.FlipHorizontal();
            }
            if (connector.mesh.mesh == null) connector.mesh.mesh = new Mesh();
            List<Vector3> vertices = new List<Vector3>();
            List<Vector3> normals = new List<Vector3>();
            List<Vector2> uvs = new List<Vector2>();
            List<Color32> colors = new List<Color32>();
            List<int> triangles = new List<int>();
            connector.innerVertices = new List<int>();
            List<List<SmartArrowUtilities.EarClipping.ColoredVertex>> outerOutline = FormOutlineList(outer);
            List<List<SmartArrowUtilities.EarClipping.ColoredVertex>> innerOutline = FormOutlineList(inner);
            if (outerOutline[0].Count > 2 && Vector3.Distance(outerOutline[0][outerOutline[0].Count - 1].position, outerOutline[0][0].position) < SmartArrowUtilities.Utilities.errorRate)
            {
                outerOutline[0].RemoveAt(outerOutline[0].Count - 1);
            }
            outerOutline[0].Reverse();
            float minOuterLength = 100;
            for (int i = 0; i < outerOutline[0].Count; i++)
            {
                vertices.Add(outerOutline[0][i].position);
                if (outerOutline[0][i].position.magnitude < minOuterLength) minOuterLength = outerOutline[0][i].position.magnitude;
                if ((outerOutline[0][i].position + outerOutline[0][(i+1)%outerOutline[0].Count].position).magnitude/2 < minOuterLength) minOuterLength = (outerOutline[0][i].position + outerOutline[0][(i + 1) % outerOutline[0].Count].position).magnitude / 2;
            }
            float maxInnerLen = 0;
            for (int i = 0; i < innerOutline.Count; i++)
            {
                if (innerOutline[i].Count > 2 && Vector3.Distance(innerOutline[i][innerOutline[i].Count - 1].position, innerOutline[i][0].position) < SmartArrowUtilities.Utilities.errorRate)
                {
                    innerOutline[i].RemoveAt(innerOutline[i].Count - 1);
                }
                for (int j = 0; j < innerOutline[i].Count; j++)
                {
                    vertices.Add(innerOutline[i][j].position);
                    connector.innerVertices.Add(vertices.Count - 1);
                    if (innerOutline[i][j].position.magnitude > maxInnerLen) maxInnerLen = innerOutline[i][j].position.magnitude;
                }
            }
            if (maxInnerLen > minOuterLength)
                minOuterLength = maxInnerLen / minOuterLength + 0.5f;
            else minOuterLength = 1.02f;
            for (int i = 0; i < outerOutline[0].Count; i++)
            {
                outerOutline[0][i].position *= minOuterLength;
            }
            SmartArrowUtilities.EarClipping.TriangulateFaceWithHoles(triangles, outerOutline[0], innerOutline);
            for (int t = 0; t < triangles.Count; t += 3)
            {
                if (tail)
                {
                    int tempI = triangles[t + 1];
                    triangles[t + 1] = triangles[t + 2];
                    triangles[t + 2] = tempI;
                }
            }
            if (vertices != null)
            {
                for (int i = 0; i < vertices.Count; i++)
                {
                    if (tail) normals.Add(Vector3.forward);
                    else normals.Add(Vector3.back);
                    colors.Add(Color.white);
                    uvs.Add(new Vector2(vertices[i].x, vertices[i].y));
                }
                connector.mesh.mesh.Clear();
                connector.mesh.mesh.name = "FaceMesh";
                connector.mesh.mesh.SetVertices(vertices);
                connector.mesh.mesh.SetNormals(normals);
                connector.mesh.mesh.SetUVs(0, uvs);
                connector.mesh.mesh.SetColors(colors);
                connector.mesh.mesh.SetTriangles(triangles, 0);
                connector.mesh.mesh.RecalculateBounds();
                connector.mesh.mesh.RecalculateTangents();
                connector.mesh.mesh.UploadMeshData(false);
                connector.LoadMesh(connector.mesh.mesh);
            }
        }
        /// <summary>
        /// Prepares vertices for the outline triangulation. Combines edges if they have shared vertices.
        /// </summary>
        /// <param name="outline">Source outline.</param>
        /// <returns>List of individual shape oultines.</returns>
        static List<List<SmartArrowUtilities.EarClipping.ColoredVertex>> FormOutlineList(Outline outline)
        {
            List<List<SmartArrowUtilities.EarClipping.ColoredVertex>> edges = new List<List<SmartArrowUtilities.EarClipping.ColoredVertex>>();
            for (int i = 0; i < outline.edges.Count; i++)
            {
                edges.Add(new List<SmartArrowUtilities.EarClipping.ColoredVertex>());
                for (int j = 0; j < outline.edges[i].points.Count; j++)
                {
                    edges[i].Add(new SmartArrowUtilities.EarClipping.ColoredVertex(outline.edges[i].points[j].position, outline.edges[i].points[j].color));
                }

            }

            for (int i = 0; i < edges.Count; i++)
            {
                for (int j = i + 1; j < edges.Count; j++)
                {
                    if (Vector3.Distance(edges[i][edges[i].Count - 1].position, edges[j][0].position) < SmartArrowUtilities.Utilities.errorRate)
                    {
                        for (int k = 1; k < edges[j].Count; k++)
                        {
                            edges[i].Add(edges[j][k]);
                        }
                        edges.RemoveAt(j);
                        i--;
                        break;
                    }
                    else if (Vector3.Distance(edges[j][edges[j].Count - 1].position, edges[i][0].position) < SmartArrowUtilities.Utilities.errorRate)
                    {
                        for (int k = 1; k < edges[i].Count; k++)
                        {
                            edges[j].Add(edges[i][k]);
                        }
                        edges.RemoveAt(i);
                        i--;
                        break;
                    }
                }
            }
            return edges;
        }
    }
}