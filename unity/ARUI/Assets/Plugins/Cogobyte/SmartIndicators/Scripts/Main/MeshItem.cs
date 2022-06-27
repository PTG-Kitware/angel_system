using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    /// <summary>
    /// Container for arrow mesh with index and coloring options.
    /// </summary>
    [System.Serializable]
    public class MeshItem
    {
        /// <summary>
        /// Mesh filter index within arrow renderer list.
        /// </summary>
        [Min(0)]
        public int meshIndex = 0;
        /// <summary>
        /// Arrow mesh container that holds mesh data.
        /// </summary>
        public ArrowMesh mesh = new ArrowMesh();
        /// <summary>
        /// Coloring modes. Input color is loaded from color parameter off AddToMesh and AddToMeshInnerOuter, single color uses color property and vertex color uses arrow mesh vertex colors.
        /// </summary>
        public enum MeshItemColorMode {InputColor,SingleColor,VertexColor};
        /// <summary>
        /// Current color mode.
        /// </summary>
        public MeshItemColorMode colorMode = MeshItemColorMode.InputColor;
        /// <summary>
        /// Color of all vertices if color mode is set to single color.
        /// </summary>
        public Color color = Color.white;
        public MeshItem() { }
        public MeshItem(MeshItem m) {
            meshIndex = m.meshIndex;
            mesh = new ArrowMesh(m.mesh);
            colorMode = m.colorMode;
            color = m.color;
            mesh.GenerateMesh();
        }
        /// <summary>
        /// Loads data from unity mesh into arrow mesh.
        /// </summary>
        /// <param name="inputMesh">Source unity mesh.</param>
        public void LoadMesh(Mesh inputMesh)
        {
            mesh.name = inputMesh.name;
            mesh.vertices = new List<Vector3>(inputMesh.vertices);
            mesh.normals = new List<Vector3>(inputMesh.normals);
            mesh.colors = new List<Color32>(inputMesh.colors32);
            mesh.triangles = new List<int>();
            mesh.uvs = new List<Vector2>();
            mesh.triangles = new List<int>(inputMesh.triangles);
            mesh.uvs= new List<Vector2>(inputMesh.uv);
            if (mesh.colors == null) mesh.colors = new List<Color32>();
            if (mesh.colors.Count != mesh.vertices.Count)
            {
                mesh.colors.Clear();
                for(int i = 0; i < mesh.vertices.Count; i++)
                {
                    mesh.colors.Add(Color.white);
                }
            }
        }
        /// <summary>
        /// Appends mesh item mesh data to arrow mesh mesh data.
        /// </summary>
        /// <param name="destinationMesh">Target arrow mesh data.</param>
        /// <param name="position">Pivot of the source mesh.</param>
        /// <param name="rotation">Rotation of the source mesh.</param>
        /// <param name="size">Scale of the source mesh.</param>
        /// <param name="color">Color for input color mode.</param>
        public void AddToMesh(ArrowMesh destinationMesh, Vector3 position, Quaternion rotation, Vector3 size, Color32 color)
        {
            destinationMesh.UpdateStartVertexIndex();
            if (colorMode == MeshItemColorMode.SingleColor)
            {
                color = this.color;
            }
            for (int i = 0; i < mesh.vertices.Count; i++)
            {
                if (colorMode == MeshItemColorMode.VertexColor)
                {
                    color = mesh.colors[i];
                }
                destinationMesh.AddVertex(position + rotation * Vector3.Scale(size, mesh.vertices[i]), mesh.uvs[i], rotation * mesh.normals[i], color);
            }
            for(int i = 0; i < mesh.triangles.Count; i+=3)
            {
                destinationMesh.AddTriangle(mesh.triangles[i], mesh.triangles[i + 1], mesh.triangles[i + 2]);
            }
        }
        /// <summary>
        /// Appends mesh item mesh data to arrow mesh mesh data and scales inner vertices.
        /// </summary>
        /// <param name="destinationMesh">Target arrow mesh data.</param>
        /// <param name="pos">Pivot of the source mesh.</param>
        /// <param name="rot">Rotation of the source mesh</param>
        /// <param name="innerSize">Scale of inner vertices of the soruce mesh.</param>
        /// <param name="outerSize">Scale of outer vertices of the soruce mesh.</param>
        /// <param name="color">Color for input color mode.</param>
        /// <param name="inner">References to inner vertices of the source mesh.</param>
        public void AddToMeshInnerOuter(ArrowMesh destinationMesh, Vector3 pos, Quaternion rot, Vector3 innerSize, Vector3 outerSize, Color32 color,List<int> inner)
        {
            destinationMesh.UpdateStartVertexIndex();
            int k = 0;
            if (colorMode == MeshItemColorMode.SingleColor)
            {
                color = this.color;
            }
            for (int i = 0; i < mesh.vertices.Count; i++)
            {
                if (colorMode == MeshItemColorMode.VertexColor)
                {
                    color = mesh.colors[i];
                }
                if (k<inner.Count && i == inner[k])
                {
                    destinationMesh.AddVertex(pos + rot * Vector3.Scale(innerSize, mesh.vertices[i]), innerSize * mesh.uvs[i], rot * mesh.normals[i], color);
                    k++;
                }
                else
                {
                    destinationMesh.AddVertex(pos + rot * Vector3.Scale(outerSize, mesh.vertices[i]), outerSize * mesh.uvs[i], rot * mesh.normals[i], color);
                }
            }
            for (int i = 0; i < mesh.triangles.Count; i+=3)
            {
                destinationMesh.AddTriangle(mesh.triangles[i], mesh.triangles[i + 1], mesh.triangles[i + 2]);
            }
        }
    }

    /// <summary>
    /// Mesh item with refereces to inner vertices so they can be scaled independently from other vertices.
    /// </summary>
    [System.Serializable]
    public class BodyTipMeshItem:MeshItem
    {
        /// <summary>
        /// Reference to inner outline vertices.
        /// </summary>
        public List<int> innerVertices = new List<int>();
    }
}