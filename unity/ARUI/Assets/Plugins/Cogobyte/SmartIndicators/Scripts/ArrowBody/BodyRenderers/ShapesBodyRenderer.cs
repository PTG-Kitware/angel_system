using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    /// <summary>
    /// Renders meshes along the body.
    /// </summary>
    public class ShapesBodyRenderer : BodyRenderer
    {
        /// <summary>
        /// Mesh item with adittional information.
        /// </summary>
        [System.Serializable]
        public class MeshShape
        {
            /// <summary>
            /// Empty space on the body before the mesh position.
            /// </summary>
            public float spaceBefore = 0.5f;
            /// <summary>
            /// Empty space on the body after the mesh position.
            /// </summary>
            public float spaceAfter = 0.5f;
            /// <summary>
            /// Position offset of the mesh.
            /// </summary>
            public Vector3 offset = Vector3.zero;
            /// <summary>
            /// Rotation offset of the mesh.
            /// </summary>
            public Vector3 rotation = Vector3.zero;
            /// <summary>
            /// Scale of the mesh.
            /// </summary>
            public Vector3 scale = Vector3.one;
            /// <summary>
            /// List of mesh items that form the main shape mesh.
            /// </summary>
            public List<MeshItem> meshes = new List<MeshItem>();
        }
        /// <summary>
        /// List of shape templates that get rendered. After the last shape the first shape is the next.
        /// </summary>
        public List<MeshShape> shapes = new List<MeshShape>();

        /// <summary>
        /// Color mode for shapes. Use path colors or gradient.
        /// </summary>
        public enum ShapesBodyRendererColorMode {ColorAlongBody, ColorAlongBodyGradient }
        /// <summary>
        /// Current color mode. It can be overriden by individual shape meshes.
        /// </summary>
        public ShapesBodyRendererColorMode colorMode = ShapesBodyRendererColorMode.ColorAlongBodyGradient;
        /// <summary>
        /// List of path colors for color along body mode.
        /// </summary>
        public List<PathColor> colors = new List<PathColor>() { new PathColor() { } };
        /// <summary>
        /// Color gradinet for color along body gradient mode.
        /// </summary>
        public Gradient colorsGradient = new Gradient()
        {
            colorKeys = new GradientColorKey[2] {
            new GradientColorKey(Color.white, 0),
            new GradientColorKey(Color.white, 1)
            },
            alphaKeys = new GradientAlphaKey[2] {
            new GradientAlphaKey(1, 0),
            new GradientAlphaKey(1, 1)
            }
        };
        /// <summary>
        /// Remaining space mode defines what to do if there is no space before or after the shape. Scale will scale the mesh, hide will not render it and render mode will render it regardless of remaining space.
        /// </summary>
        public enum RemainingSpaceMode { Scale, Hide, Render }
        /// <summary>
        /// Current remaining space mode.
        /// </summary>
        public RemainingSpaceMode remainingSpaceMode = RemainingSpaceMode.Scale;

        /// <summary>
        /// Current remaining chunk of path color.
        /// </summary>
        float m_colorLength =0;
        /// <summary>
        /// Current path color index in colors list.
        /// </summary>
        int m_colorIndex=0;
        /// <summary>
        /// Current shape index in shape list.
        /// </summary>
        int m_shapeIndex = 0;
        /// <summary>
        /// Current remaining space to next shape render.
        /// </summary>
        float m_shapePathChunk = 0;
        /// <summary>
        /// How much of the body path was consumed so far.
        /// </summary>
        float m_traveledBodyLength = 0;

        /// <summary>
        /// Ensures shapes dont have 0 length in total or they will render infinitely.
        /// </summary>
        public void OnValidate()
        {
            if (shapes == null) shapes = new List<MeshShape>();
            foreach (MeshShape s in shapes)
            {
                if (s.spaceBefore <= 0) s.spaceBefore = 0;
                if (s.spaceAfter <= 0) s.spaceAfter = 0;
                if (s.spaceAfter + s.spaceBefore <= 0)
                {
                    s.spaceBefore = 0.1f;
                    s.spaceAfter = 0.1f;
                }
            }
        }

        public override float GetCurrentRoll()
        {
            return 0;
        }
        public override Color32 GetCurrentColor()
        {
            return Color32.Lerp(colors[m_colorIndex].startColor, colors[m_colorIndex].endColor, 1 - m_colorLength / colors[m_colorIndex].length);
        }
        public override int MaxMeshIndex()
        {
            int t = 0;
            foreach(MeshShape s in shapes)
            {
                foreach(MeshItem m in s.meshes)
                {
                    if (m.meshIndex > t) t = m.meshIndex;
                }
            }
            return t;
        }
        public override void InitializeOutline(List<ArrowRenderer> arrowRenderers, float bodyDisplacement, float colorDisplacement, float bodyLength)
        {
            if (shapes.Count == 0) return;
            m_shapeIndex = 0;
            m_shapePathChunk = shapes[0].spaceBefore;
            
            if (bodyDisplacement < 0)
            {
                bodyDisplacement = -bodyDisplacement;
                m_shapeIndex = 0;
                while (bodyDisplacement >= shapes[m_shapeIndex].spaceBefore + shapes[m_shapeIndex].spaceAfter)
                {
                    bodyDisplacement -= shapes[m_shapeIndex].spaceBefore + shapes[m_shapeIndex].spaceAfter;
                    m_shapeIndex = (m_shapeIndex + 1) % shapes.Count;
                }
                if (bodyDisplacement > shapes[m_shapeIndex].spaceBefore)
                {
                    bodyDisplacement -= shapes[m_shapeIndex].spaceBefore;
                    m_shapePathChunk = shapes[m_shapeIndex].spaceAfter - bodyDisplacement;
                    m_shapeIndex = (m_shapeIndex + 1) % shapes.Count;
                    m_shapePathChunk += shapes[m_shapeIndex].spaceBefore;
                }
                else
                {
                    m_shapePathChunk = shapes[m_shapeIndex].spaceBefore - bodyDisplacement;
                }
            }
            else if(bodyDisplacement > 0)
            {
                int lastShapeIndex = shapes.Count-1;
                m_shapeIndex = 0;
                bodyDisplacement += shapes[m_shapeIndex].spaceBefore;
                m_shapePathChunk = shapes[lastShapeIndex].spaceAfter + shapes[m_shapeIndex].spaceBefore;
                while (bodyDisplacement >= m_shapePathChunk)
                {
                    bodyDisplacement -= m_shapePathChunk;
                    m_shapeIndex = lastShapeIndex;
                    lastShapeIndex = (lastShapeIndex-1+shapes.Count)%shapes.Count;
                    m_shapePathChunk = shapes[lastShapeIndex].spaceAfter + shapes[m_shapeIndex].spaceBefore;
                }
                m_shapePathChunk = bodyDisplacement;
            }
            for (int i = 0; i < colors.Count; i++)
            {
                if (colors[i].colorMode == PathColor.PathColorMode.Percentage)
                {
                    colors[i].length = colors[i].percentage * bodyLength;
                }
            }
            m_traveledBodyLength = 0;
            m_colorIndex = 0;
            m_colorLength = colors[m_colorIndex].length;
            if (colorDisplacement < 0)
            {
                colorDisplacement = -colorDisplacement;
                while (colorDisplacement > m_colorLength)
                {
                    colorDisplacement -= m_colorLength;
                    m_colorIndex = (m_colorIndex + 1) % colors.Count;
                    m_colorLength = colors[m_colorIndex].length;
                }
                m_colorLength -= colorDisplacement;
            }
            else if(colorDisplacement>0){
                m_colorIndex = colors.Count-1;
                m_colorLength = colors[m_colorIndex].length;
                while (colorDisplacement > m_colorLength)
                {
                    colorDisplacement -= m_colorLength;
                    m_colorIndex = (m_colorIndex - 1 + colors.Count) % colors.Count;
                    m_colorLength = colors[m_colorIndex].length;
                }
                m_colorLength += colorDisplacement;
            }
        }
        public override void GenerateBody(List<ArrowRenderer> arrowRenderers, BodyRendererInputData bodyChunkData, float generateLength)
        {
            Vector3 pos = bodyChunkData.startPosition;
            Vector3 end;
            Vector3 pathSection = bodyChunkData.endPosition - bodyChunkData.startPosition;
            float startGenerateLength = generateLength;
            if (shapes.Count == 0) return;
            while (generateLength > m_shapePathChunk)
            {
                end = pos + pathSection.normalized * m_shapePathChunk;
                float pLength = (end - pos).magnitude;
                m_traveledBodyLength += pLength;
                while (pLength > m_colorLength)
                {
                    pLength -= m_colorLength;
                    m_colorIndex = (m_colorIndex + 1) % colors.Count;
                    m_colorLength = colors[m_colorIndex].length;
                }
                m_colorLength -= pLength;

                generateLength -= m_shapePathChunk;
                if (remainingSpaceMode == RemainingSpaceMode.Render || (m_shapePathChunk + m_traveledBodyLength>= shapes[m_shapeIndex].spaceBefore)
                    && (m_traveledBodyLength + shapes[m_shapeIndex].spaceAfter <= bodyChunkData.bodyLength))
                {
                    for(int i=0;i< shapes[m_shapeIndex].meshes.Count; i++)
                    {
                        shapes[m_shapeIndex].meshes[i].AddToMesh(arrowRenderers[shapes[m_shapeIndex].meshes[i].meshIndex].arrowMesh, end + bodyChunkData.rotation * shapes[m_shapeIndex].offset, bodyChunkData.rotation * Quaternion.Euler(shapes[m_shapeIndex].rotation), shapes[m_shapeIndex].scale, Color32.Lerp(colors[m_colorIndex].startColor, colors[m_colorIndex].endColor, 1 - m_colorLength / colors[m_colorIndex].length));
                    }
                }
                else
                {
                    if (remainingSpaceMode == RemainingSpaceMode.Scale)
                    {
                        float afterScale = 1;
                        float beforeScale = 1;

                        afterScale = m_traveledBodyLength + shapes[m_shapeIndex].spaceAfter - bodyChunkData.bodyLength;
                        if (afterScale > 0)
                            afterScale = 1 - afterScale / shapes[m_shapeIndex].spaceAfter;
                        else afterScale = 1;
                        beforeScale = (m_shapePathChunk + m_traveledBodyLength) / shapes[m_shapeIndex].spaceBefore;
                        if (beforeScale < afterScale) afterScale = beforeScale;
                        for (int i = 0; i < shapes[m_shapeIndex].meshes.Count; i++)
                        {
                            shapes[m_shapeIndex].meshes[i].AddToMesh(arrowRenderers[shapes[m_shapeIndex].meshes[i].meshIndex].arrowMesh, end + bodyChunkData.rotation * shapes[m_shapeIndex].offset, bodyChunkData.rotation * Quaternion.Euler(shapes[m_shapeIndex].rotation), shapes[m_shapeIndex].scale * afterScale, Color32.Lerp(colors[m_colorIndex].startColor, colors[m_colorIndex].endColor, 1 - m_colorLength / colors[m_colorIndex].length));
                        }
                    }
                }

                m_shapePathChunk = shapes[m_shapeIndex].spaceAfter;
                m_shapeIndex = (m_shapeIndex + 1) % shapes.Count;
                m_shapePathChunk += shapes[m_shapeIndex].spaceBefore;
                pos = end;
            }
            m_shapePathChunk -= generateLength;
            m_traveledBodyLength += generateLength;
            while (generateLength > m_colorLength)
            {
                generateLength -= m_colorLength;
                m_colorIndex = (m_colorIndex + 1) % colors.Count;
                m_colorLength = colors[m_colorIndex].length;
            }
            m_colorLength -= generateLength;
        }
        public override void GenerateBodyBreak(List<ArrowRenderer> arrowRenderers, BodyRendererInputData bodyChunkData, bool front){}
        public override bool GenerateTailConnection(List<ArrowRenderer> arrowRenderers, BodyRendererInputData bodyChunkData, ArrowTip arrowTip)
        {
            return true;
        }
        public override bool GenerateHeadConnection(List<ArrowRenderer> arrowRenderers, BodyRendererInputData bodyChunkData, ArrowTip arrowTip)
        {
            return true;
        }
    }
} 