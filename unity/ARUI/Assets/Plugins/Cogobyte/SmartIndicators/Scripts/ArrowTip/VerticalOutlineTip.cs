using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    /// <summary>
    /// Tip that is rendered using an extruded outline along Y axis with a back face mesh.
    /// </summary>
    public class VerticalOutlineTip : ArrowTip
    {
        /// <summary>
        /// Scales the outline along the path (z-axis of the tip).
        /// </summary>
        [Min(0)]
        public float lengthScale = 1;
        /// <summary>
        /// Outline to extrude.
        /// </summary>
        public Outline outline = new Outline();
        /// <summary>
        /// Pre loaded unity mesh rendered when the space behind tip is without a body.
        /// </summary>
        public List<MeshItem> backFaceMeshes = new List<MeshItem>() { };
        /// <summary>
        /// Color modes. Color of outine, outline edge or outline point or last color that was used by the arrow body or first in case it is a tail.
        /// </summary>
        public enum VerticalOutlineTipColorMode { InputColor, OutlineColor, ColorPerEdge, ColorPerPoint }
        /// <summary>
        /// Current color mode.
        /// </summary>
        public VerticalOutlineTipColorMode colorMode = VerticalOutlineTipColorMode.InputColor;      
        public override int MaxMeshIndex()
        {
            int t = 0;
            foreach (OutlineEdge e in outline.edges)
            {
                if (e.meshIndex > t) t = e.meshIndex;
            }
            foreach (MeshItem m in backFaceMeshes)
            {
                if (m.meshIndex > t) t = m.meshIndex;
            }
            return t;
        }
        public override void GenerateTip(List<ArrowRenderer> arrowRenderers, BodyAndTipPathData tipData, bool tail, bool renderBackFace, Color32 inputColor, float inputRoll)
        {
            Vector3 pos = tipData.bodyEndPoint;
            Quaternion outlineVerticalRotation; 
            Quaternion backfaceRotation;
            if (tail)
            {
                pos = tipData.bodyStartPoint;

                outlineVerticalRotation = tipData.bodyStartRotation * Quaternion.AngleAxis(90, Vector3.left);
                outlineVerticalRotation = Quaternion.LookRotation(outlineVerticalRotation * Vector3.back, outlineVerticalRotation * Vector3.up);
                outlineVerticalRotation = outlineVerticalRotation * Quaternion.AngleAxis(-inputRoll, Vector3.forward);
                backfaceRotation = tipData.bodyStartRotation;
                backfaceRotation = Quaternion.LookRotation(backfaceRotation * Vector3.back, backfaceRotation * Vector3.up);
                backfaceRotation = backfaceRotation * Quaternion.AngleAxis(-inputRoll, Vector3.forward);
            }
            else
            {
                outlineVerticalRotation = tipData.bodyEndRotation * Quaternion.AngleAxis(90, Vector3.right) * Quaternion.AngleAxis(inputRoll, Vector3.forward);
                backfaceRotation = tipData.bodyEndRotation * Quaternion.AngleAxis(inputRoll, Vector3.forward);
            }
            Vector3 invertedSize = new Vector3(size.x,lengthScale,size.y) * tipData.sizeFactor;
            Vector3 trueSize = new Vector3(invertedSize.x, invertedSize.z, invertedSize.y);
            //Extrude from lowerpos to upper pos on y
            Vector3 lowerPos = pos + outlineVerticalRotation * Vector3.back * invertedSize.z * 0.5f;
            Vector3 upperPos = pos + outlineVerticalRotation * Vector3.forward * invertedSize.z * 0.5f;
            if (renderBackFace)
            {
                for (int t = 0; t < backFaceMeshes.Count; t++)
                {
                    backFaceMeshes[t].AddToMesh(arrowRenderers[backFaceMeshes[t].meshIndex].arrowMesh, pos, backfaceRotation, trueSize, inputColor);
                }
            }
            foreach (ArrowRenderer r in arrowRenderers)
            {
                r.arrowMesh.UpdateStartVertexIndex();
            }
            if (colorMode == VerticalOutlineTipColorMode.OutlineColor)
            {
                inputColor = outline.color;
            }
            //Extrude
            for (int t = 0; t < outline.edges.Count; t++)
            {
                OutlineEdge e = outline.edges[t];
                if (colorMode == VerticalOutlineTipColorMode.ColorPerEdge)
                {
                    inputColor = e.color;
                }
                arrowRenderers[e.meshIndex].arrowMesh.UpdateStartVertexIndex();
                float uvX = -invertedSize.x * e.points[0].position.x;
                for (int j = 0; j < e.points.Count; j++)
                {
                    if (colorMode == VerticalOutlineTipColorMode.ColorPerPoint)
                    {
                        inputColor = e.points[j].color;
                    }
                    arrowRenderers[e.meshIndex].arrowMesh.AddVertex(lowerPos + outlineVerticalRotation * new Vector3(e.points[j].position.x * invertedSize.x, e.points[j].position.y * invertedSize.y, 0), new Vector2(uvX, 0), outlineVerticalRotation * e.points[j].normal, inputColor);
                    uvX += (Vector2.Scale(invertedSize, e.points[j].position) - Vector2.Scale(invertedSize, e.points[(j + 1) % e.points.Count].position)).magnitude;
                }
                uvX = -invertedSize.x * e.points[0].position.x;
                for (int j = 0; j < e.points.Count; j++)
                {
                    if (colorMode == VerticalOutlineTipColorMode.ColorPerPoint)
                    {
                        inputColor = e.points[j].color;
                    }
                    arrowRenderers[e.meshIndex].arrowMesh.AddVertex(upperPos + outlineVerticalRotation * new Vector3(e.points[j].position.x * invertedSize.x, e.points[j].position.y * invertedSize.y, 0), new Vector2(uvX, trueSize.y), outlineVerticalRotation * e.points[j].normal, inputColor);
                    uvX += (Vector2.Scale(invertedSize, e.points[j].position) - Vector2.Scale(invertedSize, e.points[(j + 1) % e.points.Count].position)).magnitude;
                }
                for (int j = 0; j < e.points.Count - 1; j++)
                {
                    arrowRenderers[e.meshIndex].arrowMesh.AddTriangle(e.points.Count + j, j + 1, j);
                    arrowRenderers[e.meshIndex].arrowMesh.AddTriangle(j + e.points.Count, e.points.Count + j + 1, j + 1);
                }
            }
            //render bottom cap
            for (int t = 0; t < outline.backFaceMeshes.Count; t++)
            {
                outline.backFaceMeshes[t].AddToMesh(arrowRenderers[outline.backFaceMeshes[t].meshIndex].arrowMesh, lowerPos, outlineVerticalRotation , invertedSize, inputColor);
            }
            //render top cap
            for (int t = 0; t < outline.frontFaceMeshes.Count; t++)
            {
                outline.frontFaceMeshes[t].AddToMesh(arrowRenderers[outline.frontFaceMeshes[t].meshIndex].arrowMesh, upperPos, outlineVerticalRotation, invertedSize, inputColor);
            }
        }
        public override Color32 GetDefaultInputColor()
        {
            return outline.color;
        }
    }
}