using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    /// <summary>
    /// Tip that is rendered using an extruded outline along z axis with a front and back face.
    /// </summary>
    public class OutlineTip : ArrowTip
    {
        /// <summary>
        /// If false it will be rendered directly from start to end. If true it will use arrow path.
        /// </summary>
        public bool followPath = false;
        /// <summary>
        /// Outline to extrude.
        /// </summary>
        public Outline outline = new Outline();
        /// <summary>
        /// Scale at the end of the tip.
        /// </summary>
        public Vector2 endPointSize = new Vector2(0,0);
        /// <summary>
        /// Color mode. Use outline color or outline edge color, outline edge point color, gradient of two points or color provided by the arrow body.
        /// </summary>
        public enum OutlineTipColorMode { InputColor, TwoColorGradient, OutlineColor, ColorPerEdge, ColorPerPoint}
        /// <summary>
        /// Current color mode.
        /// </summary>
        public OutlineTipColorMode colorMode = OutlineTipColorMode.InputColor;
        /// <summary>
        /// Gradient start color.
        /// </summary>
        public Color32 startPointColor = Color.white;
        /// <summary>
        /// Gradient end color.
        /// </summary>
        public Color32 endPointColor = Color.white;
        /// <summary>
        /// Number of points for each edge.
        /// </summary>
        List<int> vertexCount = new List<int>();
        /// <summary>
        /// Number of already added points for each edge.
        /// </summary>
        List<int> blockCounters = new List<int>();

        public override bool FollowsPath()
        {
            return followPath;
        }
        public override int MaxMeshIndex()
        {
            int maxMeshIndex = 0;
            foreach (MeshItem m in outline.backFaceMeshes)
            {
                if (m.meshIndex > maxMeshIndex) maxMeshIndex = m.meshIndex;
            }
            foreach (MeshItem m in outline.frontFaceMeshes)
            {
                if (m.meshIndex > maxMeshIndex) maxMeshIndex = m.meshIndex;
            }
            foreach (OutlineEdge e in outline.edges)
            {
                if (e.meshIndex > maxMeshIndex) maxMeshIndex = e.meshIndex;
            }
            return maxMeshIndex;
        }
        public override void GenerateTip(List<ArrowRenderer> arrowRenderers, BodyAndTipPathData tipData, bool tail, bool renderBackFace, Color32 inputColor, float inputRoll)
        {
            Vector3 trueStartSize = new Vector3(size.x,size.y,1) * tipData.sizeFactor;
            Vector3 trueEndSize = new Vector3(endPointSize.x, endPointSize.y, 1) * tipData.sizeFactor;
            float trueLength = size.z * tipData.sizeFactor;
            Color32 startColor = startPointColor;
            Color32 endColor = endPointColor;
            if (colorMode == OutlineTipColorMode.InputColor)
            {
                startColor = inputColor;
                endColor = inputColor;
            }
            else if (colorMode == OutlineTipColorMode.TwoColorGradient)
            {
                startColor = startPointColor;
                endColor = endPointColor;
            }
            if (followPath)
            {
                #region Get outline vertices count
                if (vertexCount == null) vertexCount = new List<int>();
                if (blockCounters == null) blockCounters = new List<int>();
                if(vertexCount.Count != arrowRenderers.Count || blockCounters.Count != arrowRenderers.Count)
                {
                    vertexCount.Clear();
                    blockCounters.Clear();
                }
                while (vertexCount.Count < arrowRenderers.Count)
                {
                    vertexCount.Add(0);
                }
                while (blockCounters.Count < arrowRenderers.Count)
                {
                    blockCounters.Add(0);
                }
                for (int t = 0; t < outline.edges.Count; t++)
                {
                    vertexCount[outline.edges[t].meshIndex] = 0;
                    blockCounters[outline.edges[t].meshIndex] = 0;
                }
                for (int t = 0; t < outline.edges.Count; t++)
                {
                    for (int j = 0; j < outline.edges[t].points.Count; j++)
                    {
                        vertexCount[outline.edges[t].meshIndex]++;
                    }
                }
                #endregion 
                float tipTravel = 0;
                Vector3 position;
                Quaternion rotiation;
                int startIndex;
                int endIndex;
                int incrementer = 1;
                if (tail)
                {
                    position = tipData.bodyStartPoint;
                    rotiation = Quaternion.LookRotation(tipData.bodyStartRotation * Vector3.back, tipData.bodyStartRotation * Vector3.up);
                    startIndex = tipData.bodyStartIndex - 1;
                    endIndex = tipData.pathStartIndex - 1;
                    incrementer = -1;
                    inputRoll = -inputRoll;
                    if (renderBackFace)
                    {
                        GenerateBackFace(arrowRenderers,tipData.bodyStartPoint,Quaternion.LookRotation(tipData.bodyStartRotation * Vector3.back, tipData.bodyStartRotation * Vector3.up) * Quaternion.AngleAxis(-inputRoll, Vector3.forward), trueStartSize,startColor);
                    }
                }
                else
                {
                    position = tipData.bodyEndPoint;
                    rotiation = tipData.bodyEndRotation;
                    startIndex = tipData.bodyEndIndex + 1;
                    endIndex = tipData.pathEndIndex + 1;
                    if (renderBackFace)
                    {
                        GenerateBackFace(arrowRenderers,tipData.bodyEndPoint,tipData.bodyEndRotation * Quaternion.AngleAxis(inputRoll, Vector3.forward), trueStartSize,startColor);
                    }
                }
                RenderVertices(arrowRenderers, position, rotiation, trueStartSize, startColor, tipTravel, inputRoll);
                while (startIndex != endIndex)
                {
                    tipTravel += (position - tipData.calculatedPath[startIndex]).magnitude;
                    position = tipData.calculatedPath[startIndex];
                    rotiation = tipData.calculatedRotations[startIndex];
                    if (tail)
                    {
                        rotiation = Quaternion.LookRotation(rotiation * Vector3.back, rotiation * Vector3.up);
                    }
                    RenderVertices(arrowRenderers, position, rotiation, Vector2.Lerp(trueStartSize, trueEndSize, tipTravel / trueLength), Color32.Lerp(startColor, endColor, tipTravel / trueLength), tipTravel, inputRoll);
                    RenderTriangles(arrowRenderers);
                    startIndex += incrementer;
                }
                tipTravel = trueLength;
                if (tail)
                {
                    position = tipData.PathStartPoint;
                }
                else
                {
                    position = tipData.PathEndPoint;
                }
                RenderVertices(arrowRenderers, position, rotiation, trueEndSize, endColor, tipTravel, inputRoll);
                RenderTriangles(arrowRenderers);
                GenerateFrontFace(arrowRenderers,position,rotiation * Quaternion.AngleAxis(inputRoll, Vector3.forward), trueEndSize,endColor);
            }
            else
            {
                if (tail)
                {
                    if (renderBackFace)
                    {
                        GenerateBackFace(arrowRenderers, tipData.bodyStartPoint, Quaternion.LookRotation(tipData.bodyStartRotation * Vector3.back, tipData.bodyStartRotation * Vector3.up) * Quaternion.AngleAxis(-inputRoll, Vector3.forward), trueStartSize, startColor);
                    }
                    RenderStraightBlock(arrowRenderers,tipData.bodyStartPoint,tipData.PathStartPoint,Quaternion.LookRotation(tipData.bodyStartRotation * Vector3.back, tipData.bodyStartRotation * Vector3.up) * Quaternion.AngleAxis(-inputRoll, Vector3.forward), 0, -trueLength, trueStartSize, trueEndSize,startColor,endColor);
                    GenerateFrontFace(arrowRenderers,tipData.PathStartPoint,Quaternion.LookRotation(tipData.bodyStartRotation * Vector3.back, tipData.bodyStartRotation * Vector3.up) * Quaternion.AngleAxis(-inputRoll, Vector3.forward), trueEndSize,endColor);
                }
                else
                {
                    if (renderBackFace)
                    {
                        GenerateBackFace(arrowRenderers, tipData.bodyEndPoint, tipData.bodyEndRotation * Quaternion.AngleAxis(inputRoll, Vector3.forward), trueStartSize, startColor);
                    }
                    RenderStraightBlock(arrowRenderers,tipData.bodyEndPoint,tipData.PathEndPoint,tipData.bodyEndRotation * Quaternion.AngleAxis(inputRoll, Vector3.forward), tipData.bodyLength, tipData.bodyLength+trueLength,trueStartSize,trueEndSize,startColor,endColor);
                    GenerateFrontFace(arrowRenderers,tipData.PathEndPoint,tipData.bodyEndRotation * Quaternion.AngleAxis(inputRoll, Vector3.forward), trueEndSize,endColor);
                }
            }
        }
        public override Color32 GetDefaultInputColor()
        {
            return outline.color;
        }
        /// <summary>
        /// Adds an outline vertices for the extrude chain.
        /// </summary>
        /// <param name="arrowRenderers">Target arrow renderers with an arrow mesh.</param>
        /// <param name="position"> Pivot position of the outline.</param>
        /// <param name="rotation">Rotation of the outline.</param>
        /// <param name="scale">Scale of the outline.</param>
        /// <param name="inputColor">Inherrited color that can be overriden by outline colors.</param>
        /// <param name="tipTravel">How far is the pivot from the tip start. Used for Uv.y calcualtion.</param>
        /// <param name="inputRoll">Inherited roll from the arrow body.</param>
        void RenderVertices(List<ArrowRenderer> arrowRenderers, Vector3 position, Quaternion rotation, Vector2 scale, Color32 inputColor, float tipTravel, float inputRoll)
        {
            if (colorMode == OutlineTipColorMode.OutlineColor)
            {
                inputColor = outline.color;
            }
            for (int t = 0; t < outline.edges.Count; t++)
            {
                float uvX = -scale.x * outline.edges[t].points[0].position.x;
                if (colorMode == OutlineTipColorMode.ColorPerEdge)
                {
                    inputColor = outline.edges[t].color;
                }
                for (int j = 0; j < outline.edges[t].points.Count; j++)
                {
                    if (colorMode == OutlineTipColorMode.ColorPerPoint)
                    {
                        inputColor = outline.edges[t].points[j].color;
                    }
                    arrowRenderers[outline.edges[t].meshIndex].arrowMesh.AddVertex(position + rotation * Quaternion.AngleAxis(inputRoll, Vector3.forward) * new Vector3(outline.edges[t].points[j].position.x * scale.x, outline.edges[t].points[j].position.y * scale.y, 0), new Vector2(uvX, tipTravel), rotation * outline.edges[t].points[j].normal, inputColor);
                    uvX += (Vector2.Scale(scale, outline.edges[t].points[j].position) - Vector2.Scale(scale, outline.edges[t].points[(j + 1) % outline.edges[t].points.Count].position)).magnitude;
                }
            }
        }
        /// <summary>
        /// Adds outline triangles to arrow renderers between two outline vertices in the extrude chain.
        /// </summary>
        /// <param name="arrowRenderers">Target arrow renderers with an arrow mesh.</param>
        void RenderTriangles(List<ArrowRenderer> arrowRenderers)
        {
            for (int t = 0; t < arrowRenderers.Count; t++)
            {
                arrowRenderers[t].arrowMesh.startVertexIndex = arrowRenderers[t].arrowMesh.vertices.Count - 2 * vertexCount[t];
            }
            for (int i = 0; i < blockCounters.Count; i++)
            {
                blockCounters[i] = 0;
            }
            for (int t = 0; t < outline.edges.Count; t++)
            {
                for (int j = 0; j < outline.edges[t].points.Count - 1; j++)
                {
                    arrowRenderers[outline.edges[t].meshIndex].arrowMesh.AddTriangle(blockCounters[outline.edges[t].meshIndex] + j, blockCounters[outline.edges[t].meshIndex] + j + vertexCount[outline.edges[t].meshIndex], blockCounters[outline.edges[t].meshIndex] + j + 1);
                    arrowRenderers[outline.edges[t].meshIndex].arrowMesh.AddTriangle(blockCounters[outline.edges[t].meshIndex] + j + 1, blockCounters[outline.edges[t].meshIndex] + j + vertexCount[outline.edges[t].meshIndex], blockCounters[outline.edges[t].meshIndex] + j + 1 + vertexCount[outline.edges[t].meshIndex]);
                }
                blockCounters[outline.edges[t].meshIndex] += outline.edges[t].points.Count;
            }
        }
        /// <summary>
        /// Creates extrude chain of outline vertices and connects them with triangles.
        /// Goes straight from start point to end point. 
        /// </summary>
        /// <param name="arrowRenderers">Target arrow renderer with an arrow mesh.</param>
        /// <param name="startPoint">Pivot position start of the tip.</param>
        /// <param name="endPoint">Last outline pivot point.</param>
        /// <param name="rotation">Rotation of the tip.</param>
        /// <param name="startUv">Start uv y position.</param>
        /// <param name="endUv">End uv y position.</param>
        /// <param name="startScale">Scale of the first outline in the extrude chain.</param>
        /// <param name="endScale">Scale of the last outline in the extrude chain.</param>
        /// <param name="startColor">Color of the first outline in the extrude chain. </param>
        /// <param name="endColor">Color of the last outline in the extrude chain.</param>
        void RenderStraightBlock(List<ArrowRenderer> arrowRenderers, Vector3 startPoint, Vector3 endPoint, Quaternion rotation, float startUv,float endUv, Vector2 startScale, Vector2 endScale, Color32 startColor, Color32 endColor)
        {
            foreach (ArrowRenderer r in arrowRenderers)
            {
                r.arrowMesh.UpdateStartVertexIndex();
            }
            if (colorMode == OutlineTipColorMode.OutlineColor)
            {
                startColor = outline.color;
                endColor = outline.color;
            }
            for (int t = 0; t < outline.edges.Count; t++)
            {
                OutlineEdge e = outline.edges[t];
                if (colorMode == OutlineTipColorMode.ColorPerEdge)
                {
                    startColor = e.color;
                    endColor = e.color;
                }
                arrowRenderers[e.meshIndex].arrowMesh.UpdateStartVertexIndex();

                float uvX = -startScale.x * e.points[0].position.x;
                for (int j = 0; j < e.points.Count; j++)
                {
                    if (colorMode == OutlineTipColorMode.ColorPerPoint)
                    {
                        startColor = e.points[j].color;
                    }
                    arrowRenderers[e.meshIndex].arrowMesh.AddVertex(startPoint + rotation * new Vector3(e.points[j].position.x * startScale.x, e.points[j].position.y * startScale.y, 0), new Vector2(uvX, startUv), rotation * e.points[j].normal, startColor);
                    uvX += (Vector2.Scale(startScale, e.points[j].position) - Vector2.Scale(startScale, e.points[(j + 1)%e.points.Count].position)).magnitude;
                }
                uvX = -startScale.x * e.points[0].position.x;
                for (int j = 0; j < e.points.Count; j++)
                {
                    if (colorMode == OutlineTipColorMode.ColorPerPoint)
                    {
                        endColor = e.points[j].color;
                    }
                    arrowRenderers[e.meshIndex].arrowMesh.AddVertex(endPoint + rotation * new Vector3(e.points[j].position.x * endScale.x, e.points[j].position.y * endScale.y, 0), new Vector2(uvX, endUv), rotation * e.points[j].normal, endColor);
                    uvX += (Vector2.Scale(endScale, e.points[j].position) - Vector2.Scale(endScale, e.points[(j + 1) % e.points.Count].position)).magnitude;
                }
                for (int j = 0; j < e.points.Count - 1; j++)
                {
                    arrowRenderers[e.meshIndex].arrowMesh.AddTriangle(e.points.Count + j, j + 1, j);
                    arrowRenderers[e.meshIndex].arrowMesh.AddTriangle(j + e.points.Count, e.points.Count + j + 1, j + 1);
                }
            }
        }
        /// <summary>
        /// Adds the back face mesh to the arrow renderers arrow mesh.
        /// </summary>
        /// <param name="arrowRenderers">Target arrow renderer with arrow mesh.</param>
        /// <param name="position">Pivot point where to generate mesh.</param>
        /// <param name="rotation">Orientation of the input mesh.</param>
        /// <param name="scale">Input mesh scale.</param>
        /// <param name="color">Color for the mesh vertices.</param>
        void GenerateBackFace(List<ArrowRenderer> arrowRenderers, Vector3 position, Quaternion rotation, Vector3 scale, Color32 color)
        {
            for (int t = 0; t < outline.backFaceMeshes.Count; t++)
            {
                outline.backFaceMeshes[t].AddToMesh(arrowRenderers[outline.backFaceMeshes[t].meshIndex].arrowMesh, position, rotation, scale, color);
            }
        }
        /// <summary>
        /// Adds the front face mesh to the arrow renderers arrow mesh.
        /// </summary>
        /// <param name="arrowRenderers">Target arrow renderer with arrow mesh.</param>
        /// <param name="position">Pivot point where to generate mesh.</param>
        /// <param name="rotation">Orientation of the input mesh.</param>
        /// <param name="scale">Input mesh scale.</param>
        /// <param name="color">Color for the mesh vertices.</param>
        void GenerateFrontFace(List<ArrowRenderer> arrowRenderers, Vector3 position,  Quaternion rotation, Vector3 scale,Color32 color)
        {
            for (int t = 0; t < outline.frontFaceMeshes.Count; t++)
            {
                outline.frontFaceMeshes[t].AddToMesh(arrowRenderers[outline.frontFaceMeshes[t].meshIndex].arrowMesh, position, rotation, scale, color);
            }
        }
    }
}