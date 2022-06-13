using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    /// <summary>
    /// Containes procedural mesh data that can be uploaded to unity mesh.
    /// </summary>
    [System.Serializable]
    public class ArrowMesh
    {
        /// <summary>
        /// Vertex position data.
        /// </summary>
        public List<Vector3> vertices = new List<Vector3>();
        /// <summary>
        /// Triangle data.
        /// </summary>
        public List<int> triangles = new List<int>();
        /// <summary>
        /// Vertex uv data.
        /// </summary>
        public List<Vector2> uvs = new List<Vector2>();
        /// <summary>
        /// Vertex color data.
        /// </summary>
        public List<Color32> colors = new List<Color32>();
        /// <summary>
        /// Vertex normal data.
        /// </summary>
        public List<Vector3> normals = new List<Vector3>();
        /// <summary>
        /// Procedural mesh name.
        /// </summary>
        public string name;
        /// <summary>
        /// Index of vertex that is used as start for triangle upload. Used to append vertex and trianle data to arrow mesh.
        /// </summary>
        public int startVertexIndex = 0;
        /// <summary>
        /// Reference to mesh where mesh data will be uploaded to construct unity mesh.
        /// </summary>
        public Mesh mesh;

        public ArrowMesh()
        {

        }
        public ArrowMesh(ArrowMesh m)
        {
            vertices = new List<Vector3>(m.vertices);
            triangles = new List<int>(m.triangles);
            uvs = new List<Vector2>(m.uvs);
            colors = new List<Color32>(m.colors);
            normals = new List<Vector3>(m.normals);
            name = m.name;
            startVertexIndex = m.startVertexIndex;
            mesh = new Mesh();
        }
        /// <summary>
        /// Clear all data from the arrow mesh.
        /// </summary>
        public void ClearMesh()
        {
            startVertexIndex = 0;
            vertices.Clear();
            triangles.Clear();
            uvs.Clear();
            normals.Clear();
            colors.Clear();
        }
        /// <summary>
        /// Uploades arrow mesh data to unity mesh.
        /// </summary>
        public void GenerateMesh()
        {
            if(mesh == null) mesh = new Mesh();
            mesh.name = name;
            mesh.Clear();
            mesh.SetVertices(vertices);
            mesh.SetColors(colors);
            mesh.SetTriangles(triangles, 0);
            mesh.SetUVs(0, uvs);
            mesh.SetNormals(normals);
            mesh.RecalculateBounds();
            mesh.RecalculateTangents();
            mesh.UploadMeshData(false);
        }
        /// <summary>
        /// Generates a unity mesh then sets it to a provided mesh filter.
        /// </summary>
        /// <param name="meshFilter">Target mesh filter.</param>
        public void PushToMesh(MeshFilter meshFilter)
        {
            GenerateMesh();
            meshFilter.sharedMesh = mesh;
        }
        /// <summary>
        /// Adds vertex data to arrow mesh.
        /// </summary>
        /// <param name="position">Vertex position.</param>
        /// <param name="uv">Vertex 2D uv coordinate.</param>
        /// <param name="normal">Vertex normal.</param>
        /// <param name="col">Vertex color.</param>
        public void AddVertex(Vector3 position, Vector2 uv, Vector3 normal, Color32 col)
        {
            vertices.Add(position);
            uvs.Add(uv);
            normals.Add(normal);
            colors.Add(col);
        }
        /// <summary>
        /// Appends triangle data to arrow mesh. Start vertex index will be used as first vertex for triangle.
        /// </summary>
        /// <param name="a">First vertex.</param>
        /// <param name="b">Second vertex.</param>
        /// <param name="c">Third vertex.</param>
        public void AddTriangle(int a, int b, int c)
        {
            triangles.Add(a + startVertexIndex);
            triangles.Add(b + startVertexIndex);
            triangles.Add(c + startVertexIndex);
        }
        /// <summary>
        /// Resets start vertex index so next mesh can be appended without changing the triangle data first. 
        /// </summary>
        public void UpdateStartVertexIndex()
        {
            startVertexIndex = vertices.Count;
        }
    }
}