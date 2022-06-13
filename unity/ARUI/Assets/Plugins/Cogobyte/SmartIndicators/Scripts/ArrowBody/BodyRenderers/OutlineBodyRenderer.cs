using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    /// <summary>
    /// Extrudes an outline along the path.
    /// </summary>
    public class OutlineBodyRenderer : BodyRenderer
    {
        /// <summary>
        /// Outline to extrude.
        /// </summary>
        public Outline outline = new Outline();
        #region colorOptions
        /// <summary>
        /// Color modes for outline renderer. Color along body uses path colors, color along body gradient uses a gradient and others use outline,edge and point colors.
        /// </summary>
        public enum OutlineBodyRendererColorMode { ColorAlongBody, ColorAlongBodyGradient, OutlineColor, ColorPerEdge, ColorPerPoint}
        /// <summary>
        /// Current color mode.
        /// </summary>
        public OutlineBodyRendererColorMode colorMode = OutlineBodyRendererColorMode.OutlineColor;
        /// <summary>
        /// Path color list for color along body mode.
        /// </summary>
        public List<PathColor> colors = new List<PathColor>() { new PathColor() { } };
        /// <summary>
        /// Color gradient for color along body gradient mode.
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
        #endregion colorOptions
        #region curveOptions
        /// <summary>
        /// Fixed width and height of outline.
        /// </summary>
        public Vector2 size = Vector2.one;
        /// <summary>
        /// Fixed roll of the outline.
        /// </summary>
        public float roll = 0;
        /// <summary>
        /// Animation curve for width,height and roll along body.
        /// </summary>
        public AnimationCurve[] curveFunction = new AnimationCurve[3] { AnimationCurve.Linear(0, 0.1f, 1, 0.1f), AnimationCurve.Linear(0, 0.1f, 1, 0.1f), AnimationCurve.Linear(0, 0.1f, 1, 0.1f) };
        /// <summary>
        /// Size and roll modes. Fixed uses fixed values. Function uses animation curves and scales the curve to body length. Function length uses animation curves where time equals body length.
        /// </summary>
        public enum CurveFunctionMode {Fixed, Function, FunctionLength}
        /// <summary>
        /// Current curve mode for width, height and roll.
        /// </summary>
        public CurveFunctionMode[] curveModes = new CurveFunctionMode[] { CurveFunctionMode.Fixed, CurveFunctionMode.Fixed, CurveFunctionMode.Fixed };
        /// <summary>
        /// Body chunk minimum number of segments.
        /// </summary>
        public int lod = 1;
        #endregion curveOptions
        #region cornerOptions
        /// <summary>
        /// If false generates new verties for start and end of the body chunk making sharp extrude along path.
        /// If true reuses vertices from previous body chunk end for body chunk start to make smooth edges extrude along path.
        /// </summary>
        public bool smoothPath = false;
        /// <summary>
        /// If true fix corner scale when making 90 degree turns to keep the same width along path.
        /// </summary>
        public bool scaleCorners = false;
        #endregion cornerOptions
        #region dashOptions
        /// <summary>
        /// If true render dashes instead of one continuus extrude.
        /// </summary>
        public bool dashed = false;
        /// <summary>
        /// Space between two dashes.
        /// </summary>
        public float emptyLength = 0.1f;
        /// <summary>
        /// Length of each dash.
        /// </summary>
        public float dashLength = 1f;
        #endregion dashOptions
        #region tipConnectors
        /// <summary>
        /// Meshes that form one mesh that connects arrow body and tail tip.
        /// </summary>
        public List<BodyTipMeshItem> tailMeshConnector = new List<BodyTipMeshItem>();
        /// <summary>
        /// Meshes that form one mesh that connects arrow body and head tip.
        /// </summary>
        public List<BodyTipMeshItem> headMeshConnector = new List<BodyTipMeshItem>();
        #endregion tipConnectors

        /// <summary>
        /// Is body currently rendering outline or empty space.
        /// </summary>
        bool m_dashing = true;
        /// <summary>
        /// Has the m_dashing changed since last chunk.
        /// </summary>
        bool m_switchDash = true;
        /// <summary>
        /// How much dash is left until empty space starts rendering. 
        /// </summary>
        float m_dashChunk = 0;
        /// <summary>
        /// Current color index in path list.
        /// </summary>
        int m_currentColorKey = 0;
        /// <summary>
        /// Start color of body chunk.
        /// </summary>
        Color32 m_colorValue;
        /// <summary>
        /// Current value for width, height and roll.
        /// </summary>
        float[] m_curveCurrentValue = new float[3] { 0, 0, 0 };
        /// <summary>
        /// Curve length scaling for width, height and roll.
        /// </summary>
        float[] m_timeFormat = new float[3] { 1, 1, 1 };
        /// <summary>
        /// Current animation curve key for width, height and roll.
        /// </summary>
        int[] m_currentFunctionKey = new int[3] { 1, 1, 1 };
        /// <summary>
        /// Smallest current chunk for width,height,twist,color,lod and body.
        /// </summary>
        int m_nextLengthIndex = 0;
        /// <summary>
        /// Chunk until end for current curve key or color index for width,height,twist,color,lod and body.
        /// </summary>
        float[] m_nextLength = new float[6] { 0, 0, 0, 0, 0, 0 };

        float m_bodyTraveled;
        bool m_firstBlockRendered = false;
        /// <summary>
        /// Number of outline vertices for each arrow renderer.
        /// </summary>
        List<int> m_vertexCount = new List<int>();
        /// <summary>
        /// Currently rendered number of outline vertices for each arrow renderer.
        /// </summary>
        List<int> m_blockCounters = new List<int>();
        /// <summary>
        /// Las caluclated outline scale.
        /// </summary>
        Vector2 m_lastTurnScale;
        /// <summary>
        /// Last calculated outline rotation.
        /// </summary>
        Quaternion m_lastTurnRotation;
        /// <summary>
        /// Ensures >0 for all lenghts.
        /// </summary>
        public void OnValidate()
        {
            if (lod <= 0) lod = 1;
            if (emptyLength < 0) emptyLength = 0;
            if (dashLength < 0) dashLength = 0;
            if (emptyLength + dashLength < 0.00001f)
            {
                emptyLength = 0.1f;
            }
        }
        public override int MaxMeshIndex()
        {
            int t = 0;
            foreach (MeshItem m in tailMeshConnector)
            {
                if (m.meshIndex > t) t = m.meshIndex;
            }
            foreach (MeshItem m in headMeshConnector)
            {
                if (m.meshIndex > t) t = m.meshIndex;
            }
            foreach (MeshItem m in outline.backFaceMeshes)
            {
                if (m.meshIndex > t) t = m.meshIndex;
            }
            foreach (MeshItem m in outline.frontFaceMeshes)
            {
                if (m.meshIndex > t) t = m.meshIndex;
            }
            foreach (OutlineEdge e in outline.edges)
            {
                if (e.meshIndex > t) t = e.meshIndex;
            }
            return t;
        }
        public override float GetCurrentRoll()
        {
            return m_curveCurrentValue[2];
        }
        public override Color32 GetCurrentColor()
        {
            return m_colorValue; 
        }
        public override void InitializeOutline(List<ArrowRenderer> arrowRenderers, float bodyDisplacement, float colorDisplacement, float bodyLength)
        {
            m_firstBlockRendered = false;
            m_currentFunctionKey[0] = 0;
            m_currentFunctionKey[1] = 0;
            m_currentFunctionKey[2] = 0;
            m_bodyTraveled = 0;

            m_curveCurrentValue[0] = size.x;
            m_curveCurrentValue[1] = size.y;
            m_curveCurrentValue[2] = roll;
            m_colorValue = outline.color;

            for(int i = 0; i < 6; i++)
            {
                m_nextLength[i] = bodyLength * 2;
            }
            m_nextLengthIndex = 5;

            if (colorMode == OutlineBodyRendererColorMode.ColorAlongBody || colorMode == OutlineBodyRendererColorMode.ColorAlongBodyGradient)
            {
                for (int i = 0; i < colors.Count; i++)
                {
                    if (colors[i].colorMode == PathColor.PathColorMode.Percentage)
                    {

                        colors[i].length = colors[i].percentage * bodyLength + 3*SmartArrowUtilities.Utilities.errorRate;
                    }
                    if (colors[i].length < SmartArrowUtilities.Utilities.errorRate)
                    {
                        colors[i].length = 0.1f;
                    }
                }
                m_currentColorKey = 0;
                m_nextLength[3] = colors[m_currentColorKey].length;
                m_nextLengthIndex = 3;
                m_colorValue = colors[m_currentColorKey].startColor;

                if (colorDisplacement < 0)
                {
                    colorDisplacement = -colorDisplacement;
                    while (m_nextLength[3] < colorDisplacement)
                    {
                        colorDisplacement -= m_nextLength[3];
                        m_currentColorKey = (m_currentColorKey + 1) % colors.Count;
                        m_nextLength[3] = colors[m_currentColorKey].length;
                    }
                    m_colorValue = Color32.Lerp(colors[m_currentColorKey].startColor, colors[m_currentColorKey].endColor, colorDisplacement / colors[m_currentColorKey].length);
                    m_nextLength[3] -= colorDisplacement;
                }
                else if (colorDisplacement > 0)
                {
                    m_currentColorKey = (m_currentColorKey - 1 + colors.Count) % colors.Count;
                    m_nextLength[3] = colors[m_currentColorKey].length;
                    while (m_nextLength[3] < colorDisplacement)
                    {
                        colorDisplacement -= m_nextLength[3];
                        m_currentColorKey = (m_currentColorKey - 1 + colors.Count) % colors.Count;
                        m_nextLength[3] = colors[m_currentColorKey].length;
                    }
                    m_colorValue = Color32.Lerp(colors[m_currentColorKey].endColor, colors[m_currentColorKey].startColor, colorDisplacement / colors[m_currentColorKey].length);
                    m_nextLength[3] = colorDisplacement;
                }
            }
            for (int i = 0; i < 3; i++)
            {
                if (curveModes[i] != CurveFunctionMode.Fixed)
                {
                    m_timeFormat[i] = 1;
                    if (curveModes[i] == CurveFunctionMode.Function)
                    {
                        m_timeFormat[i] = curveFunction[i].keys[curveFunction[i].keys.Length - 1].time / bodyLength;
                    }

                    m_curveCurrentValue[i] = curveFunction[i].Evaluate(0 * m_timeFormat[i]);
                    m_nextLength[i] = curveFunction[i].keys[m_currentFunctionKey[i]].time * 1 / m_timeFormat[i];
                    if (m_nextLength[i] < m_nextLength[m_nextLengthIndex]) m_nextLengthIndex = i;
                    m_nextLength[4] = bodyLength / lod;
                    if (m_nextLength[4] < m_nextLength[m_nextLengthIndex]) m_nextLengthIndex = 4;
                }
            }

            if (m_vertexCount == null) m_vertexCount = new List<int>();
            if (m_blockCounters == null) m_blockCounters = new List<int>();
            if (m_vertexCount.Count > arrowRenderers.Count || m_blockCounters.Count > arrowRenderers.Count)
            {
                m_vertexCount.Clear();
                m_blockCounters.Clear();
            }
            while (m_vertexCount.Count < arrowRenderers.Count)
            {
                m_vertexCount.Add(0);
            }
            while (m_blockCounters.Count < arrowRenderers.Count)
            {
                m_blockCounters.Add(0);
            }
            for (int t = 0; t < outline.edges.Count; t++)
            {
                m_vertexCount[outline.edges[t].meshIndex] = 0;
                m_blockCounters[outline.edges[t].meshIndex] = 0;
            }
            for (int t = 0; t < outline.edges.Count; t++)
            {
                for (int j = 0; j < outline.edges[t].points.Count; j++)
                {
                    m_vertexCount[outline.edges[t].meshIndex]++;
                }
            }

            if (!dashed)
            {
                m_dashing = true;
            }
            else
            {
                m_dashing = true;
                float k = dashLength + emptyLength;
                int del = (int)(bodyDisplacement / k);
                if (bodyDisplacement < 0)
                {
                    bodyDisplacement = bodyDisplacement - del * k;
                    if (-bodyDisplacement > emptyLength)
                    {
                        bodyDisplacement += emptyLength;
                        m_dashChunk = dashLength + bodyDisplacement;
                        m_dashing = true;
                    }
                    else
                    {
                        m_dashChunk = emptyLength + bodyDisplacement;
                        m_dashing = false;
                    }

                }
                else
                {
                    bodyDisplacement = bodyDisplacement - del * k;
                    if (bodyDisplacement > emptyLength)
                    {
                        bodyDisplacement -= emptyLength;
                        m_dashChunk = bodyDisplacement;
                        m_dashing = true;
                    }
                    else
                    {
                        m_dashChunk = bodyDisplacement;
                        m_dashing = false;
                    }
                }
                m_switchDash = true;
            }
        }
        public override void GenerateBody(List<ArrowRenderer> arrowRenderers, BodyRendererInputData bodyChunkData, float generateLength)
        {
            Vector2 nextTurnScale = GetTurnScale(bodyChunkData.rotation, bodyChunkData.nextRotation);

            Quaternion nextTurnRotation = Quaternion.Lerp(bodyChunkData.rotation, bodyChunkData.nextRotation, 0.5f);
            if (bodyChunkData.lastRender)
            {
                nextTurnRotation = bodyChunkData.nextRotation;
            }
            if (!dashed)
            {
                AddExtrudedOutline(
                    arrowRenderers,
                    bodyChunkData.startPosition,
                    bodyChunkData.endPosition,
                    m_lastTurnRotation,
                    nextTurnRotation,
                    bodyChunkData.rotation,
                    bodyChunkData.traveledPath,
                    m_lastTurnScale,
                    nextTurnScale,
                    smoothPath && m_firstBlockRendered,
                    bodyChunkData.bodyLength
                );
            }
            else
            {
                Vector3 start = bodyChunkData.startPosition;
                Vector3 dir = (bodyChunkData.endPosition - bodyChunkData.startPosition).normalized;
                Quaternion rot = m_lastTurnRotation;
                Vector2 tempScale = nextTurnScale;

                while (generateLength > m_dashChunk)
                {
                    Vector3 end = start + dir * m_dashChunk;
                    generateLength -= m_dashChunk;
                    if (m_dashing)
                    {
                        if (m_switchDash)
                        {
                            rot = bodyChunkData.rotation;
                            m_lastTurnScale = Vector2.one;
                            RenderBodyBreak(arrowRenderers, start, bodyChunkData.rotation, outline.backFaceMeshes);
                            AddExtrudedOutline(arrowRenderers,
                                start, end, rot,
                                bodyChunkData.rotation, bodyChunkData.rotation,
                                bodyChunkData.traveledPath, tempScale, Vector2.one,
                                false, bodyChunkData.bodyLength);
                        }
                        else
                        {
                            AddExtrudedOutline(arrowRenderers,
                                start,
                                end,
                                rot,
                                bodyChunkData.rotation,
                                bodyChunkData.rotation,
                                bodyChunkData.traveledPath,
                                tempScale,
                                Vector2.one,
                                smoothPath && m_firstBlockRendered,bodyChunkData.bodyLength);
                        }
                        m_lastTurnScale = Vector3.one;
                        RenderBodyBreak(arrowRenderers, end, bodyChunkData.rotation, outline.frontFaceMeshes);
                        rot = bodyChunkData.rotation;
                        tempScale = Vector3.one;
                        m_switchDash = true;
                        m_firstBlockRendered = false;
                        m_dashing = false;
                        m_dashChunk = emptyLength;
                    }
                    else
                    {
                        rot = bodyChunkData.rotation;
                        tempScale = Vector3.one;
                        m_dashing = true;
                        m_firstBlockRendered = false;
                        ConsumeEmptySpace(m_dashChunk, bodyChunkData.bodyLength);
                        m_dashChunk = dashLength;
                    }
                    start = end;
                }
                if (m_dashing)
                {
                    if (m_switchDash)
                    {
                        m_lastTurnScale = Vector3.one;
                        RenderBodyBreak(arrowRenderers, start, bodyChunkData.rotation, outline.backFaceMeshes);
                    }
                    AddExtrudedOutline(arrowRenderers,
                        start,
                        bodyChunkData.endPosition,
                        rot,
                        nextTurnRotation,
                        bodyChunkData.rotation,
                        bodyChunkData.traveledPath,
                        m_lastTurnScale, nextTurnScale,
                        smoothPath && m_firstBlockRendered, bodyChunkData.bodyLength);
                    m_switchDash = false;
                }
                else
                {
                    ConsumeEmptySpace(generateLength, bodyChunkData.bodyLength);
                }
                m_dashChunk -= generateLength;
            }
            m_lastTurnRotation = nextTurnRotation;
            m_lastTurnScale = nextTurnScale;
        }
        public override void GenerateBodyBreak(List<ArrowRenderer> arrowRenderers, BodyRendererInputData bodyChunkData, bool front)
        {
            if (front)
            {
                RenderBodyBreak(arrowRenderers,bodyChunkData.endPosition,m_lastTurnRotation,outline.frontFaceMeshes);
            }
            else
            {
                m_lastTurnScale = GetTurnScale(bodyChunkData.lastRotation, bodyChunkData.rotation);
                m_lastTurnRotation = bodyChunkData.rotation;
                RenderBodyBreak(arrowRenderers,bodyChunkData.startPosition,m_lastTurnRotation,outline.backFaceMeshes);
            }
        }
        public override bool GenerateTailConnection(List<ArrowRenderer> arrowRenderers, BodyRendererInputData bodyChunkData, ArrowTip arrowTip)
        {
            m_lastTurnScale = GetTurnScale(bodyChunkData.lastRotation, bodyChunkData.rotation);
            m_lastTurnRotation = Quaternion.Lerp(bodyChunkData.lastRotation, bodyChunkData.rotation, 0.5f);
            if (arrowTip != null)
            {
                if (m_dashing)
                {
                    RenderTipConnector(arrowRenderers,
                        bodyChunkData.startPosition,
                        m_lastTurnRotation,
                        new Vector2(m_lastTurnScale.x * m_curveCurrentValue[0], m_lastTurnScale.y * m_curveCurrentValue[1]),
                        arrowTip.size,
                        tailMeshConnector);
                    return false;
                }
                return true;
            }
            else
            {
                if (m_dashing)
                {
                    RenderBodyBreak(arrowRenderers,
                    bodyChunkData.startPosition,
                    m_lastTurnRotation,
                    outline.backFaceMeshes);
                }
                return false;
            }
        }
        public override bool GenerateHeadConnection(List<ArrowRenderer> arrowRenderers, BodyRendererInputData bodyChunkData, ArrowTip arrowTip)
        {
            if (arrowTip != null)
            {
                if (m_dashing)
                {
                    RenderTipConnector(arrowRenderers,
                    bodyChunkData.endPosition,
                    bodyChunkData.nextRotation,
                    new Vector3(m_lastTurnScale.x * m_curveCurrentValue[0], m_lastTurnScale.y * m_curveCurrentValue[1],1),
                    new Vector3(arrowTip.size.x,arrowTip.size.y,1),
                    headMeshConnector);
                    return false;
                }
                return true;
            }
            else
            {
                if (m_dashing)
                {
                    RenderBodyBreak(arrowRenderers,bodyChunkData.endPosition, bodyChunkData.nextRotation, outline.frontFaceMeshes);
                }
                return false;
            }
        }
        /// <summary>
        /// Calculates scale for scale corners option.
        /// </summary>
        /// <param name="lastRotation">Rotation of last chunk.</param>
        /// <param name="nextRotation">Rotation of current chunk.</param>
        /// <returns>Calculated scale for outline.</returns>
        Vector2 GetTurnScale(Quaternion lastRotation, Quaternion nextRotation)
        {
            if (scaleCorners)
            {
                Vector2 scaleByAngle = new Vector2();
                scaleByAngle.y = Vector3.Angle(lastRotation * Vector3.up, nextRotation * Vector3.up);
                scaleByAngle.x = Vector3.Angle(lastRotation * Vector3.right, nextRotation * Vector3.right);
                if (scaleByAngle.x <= 90)
                    scaleByAngle.x = Mathf.Lerp(1, 1.414f, scaleByAngle.x / 90);
                else
                    scaleByAngle.x = 1.41f;
                if (scaleByAngle.y <= 90)
                    scaleByAngle.y = Mathf.Lerp(1, 1.414f, scaleByAngle.y / 90);
                else
                    scaleByAngle.y = 1.41f;


                /*
                //float scaleHeightByAngle = Mathf.Abs(Mathf.Rad2Deg * (Mathf.Atan2(2 * a.x * a.w - 2 * a.y * a.z, 1 - 2 * a.x * a.x - 2 * a.z * a.z)- Mathf.Atan2(2 * b.x * b.w - 2 * b.y * b.z, 1 - 2 * b.x * b.x - 2 * b.z * b.z)));
                //float scaleWidthByAngle =Mathf.Abs(Mathf.Rad2Deg * (Mathf.Atan2(2 * a.y * a.w - 2 * a.x * a.z, 1 - 2 * a.y * a.y - 2 * a.z * a.z)- Mathf.Atan2(2 * b.y * b.w - 2 * b.x * b.z, 1 - 2 * b.y * b.y - 2 * b.z * b.z)));
                
                Vector3 dir = Quaternion.LookRotation(nextRotation*Vector3.up,nextRotation*Vector3.back) * Quaternion.Inverse(lastRotation) *  Vector3.forward;
                scaleByAngle.x = Mathf.Atan2(dir.x, dir.z);
                var xzLen = new Vector2(dir.x, dir.z).magnitude;
                scaleByAngle.y = Mathf.Atan2(-dir.y, xzLen);
                scaleByAngle *= Mathf.Rad2Deg;

                scaleByAngle.x = Mathf.Abs(scaleByAngle.x);
                scaleByAngle.y = Mathf.Abs(scaleByAngle.y);
                if (scaleByAngle.x <= 90)
                    scaleByAngle.x = Mathf.Lerp(1, 1.414f, scaleByAngle.x / 90);
                else
                    scaleByAngle.x = 1.41f;

                if (scaleByAngle.y <= 90)
                    scaleByAngle.y = Mathf.Lerp(1, 1.414f, scaleByAngle.y / 90);
                else
                    scaleByAngle.y = 1.41f;









                /*

                
                Quaternion q = b * ;
                Debug.Log("A:"+a * Vector3.forward);
                Debug.Log(a * Vector3.up);
                Debug.Log("B:" + b * Vector3.forward);
                Debug.Log(b * Vector3.up);
                Debug.Log("Q:" + q * Vector3.forward);
                Debug.Log(q * Vector3.up);
                float scaleHeightByAngle = Mathf.Abs(Mathf.Rad2Deg * Mathf.Atan2(2 * q.x * q.w - 2 * q.y * q.z, 1 - 2 * q.x * q.x - 2 * q.z * q.z));
                float scaleWidthByAngle =Mathf.Abs(Mathf.Rad2Deg * Mathf.Atan2(2 * q.y * q.w - 2 * q.x * q.z, 1 - 2 * q.y * q.y - 2 * q.z * q.z));


                
                */
                /*
                 * Current solution
                Vector3 inv = nextRotation * Quaternion.Inverse(lastRotation) * Vector3.forward;
                float scaleWidthByAngle = 1;
                float scaleHeightByAngle = 1;
                if (Mathf.Abs(inv.x)>SmartArrowUtilities.Utilities.errorRate || Mathf.Abs(inv.z) > SmartArrowUtilities.Utilities.errorRate) {
                    scaleWidthByAngle = Vector3.Angle(Vector3.forward, new Vector3(inv.x, 0f, inv.z));
                    if (scaleWidthByAngle <= 90)
                        scaleWidthByAngle = Mathf.Lerp(1, 1.414f, scaleWidthByAngle / 90);
                    else
                        scaleWidthByAngle = 1.41f;
                    scaleHeightByAngle = Vector3.Angle(new Vector3(inv.x, 0, inv.z), inv);
                    if (scaleHeightByAngle <= 90)
                        scaleHeightByAngle = Mathf.Lerp(1, 1.414f, scaleHeightByAngle / 90);
                    else
                        scaleHeightByAngle = 1.41f;
                }
                scaleByAngle = new Vector2(scaleWidthByAngle, scaleHeightByAngle);
                */
                return scaleByAngle;
            }
            return new Vector2(1, 1);
        }
        /// <summary>
        /// Generate extrude vertices from outline.
        /// </summary>
        /// <param name="arrowRenderers">Target arrow renderers.</param>
        /// <param name="position">Outline pivot position.</param>
        /// <param name="rotation">Outline rotation.</param>
        /// <param name="scale">Outline width and height.</param>
        /// <param name="normalsRotation">Outline normals rotaiton.</param>
        /// <param name="uvY">Outline uv y coordinate in the extrude chain.</param>
        void RenderVertices(List<ArrowRenderer> arrowRenderers, Vector3 position, Quaternion rotation, Vector2 scale, Quaternion normalsRotation, float uvY)
        {
            for (int t = 0; t < outline.edges.Count; t++)
            {
                float uvX = 0;
                OutlineEdge e = outline.edges[t];
                if (colorMode == OutlineBodyRendererColorMode.ColorPerEdge)
                {
                    m_colorValue = e.color;
                }
                for (int j = 0; j < e.points.Count; j++)
                {
                    if (colorMode == OutlineBodyRendererColorMode.ColorPerPoint)
                    {
                        m_colorValue = e.points[j].color;
                    }
                    arrowRenderers[e.meshIndex].arrowMesh.AddVertex(position + rotation * Quaternion.AngleAxis(m_curveCurrentValue[2], Vector3.forward) * new Vector3(m_curveCurrentValue[0] * scale.x * e.points[j].position.x, m_curveCurrentValue[1] * scale.y * e.points[j].position.y, 0), new Vector2(uvX, uvY), normalsRotation * Quaternion.AngleAxis(m_curveCurrentValue[2], Vector3.forward) * e.points[j].normal, m_colorValue);
                    uvX += (e.points[j].position - e.points[(j + 1) % e.points.Count].position).magnitude;
                }
            }
        }
        /// <summary>
        /// Generates outline triangles between two extrude position.
        /// </summary>
        /// <param name="arrowRenderers">Target arrow renderers with target arrow meshes.</param>
        void RenderTriangles(List<ArrowRenderer> arrowRenderers)
        {
            for (int t = 0; t < arrowRenderers.Count; t++)
            {
                arrowRenderers[t].arrowMesh.startVertexIndex = arrowRenderers[t].arrowMesh.vertices.Count - 2 * m_vertexCount[t];
            }
            for (int i = 0; i < m_blockCounters.Count; i++)
            {
                m_blockCounters[i] = 0;
            }
            for (int t = 0; t < outline.edges.Count; t++)
            {
                for (int j = 0; j < outline.edges[t].points.Count - 1; j++)
                {
                    arrowRenderers[outline.edges[t].meshIndex].arrowMesh.AddTriangle(m_blockCounters[outline.edges[t].meshIndex] + j, m_blockCounters[outline.edges[t].meshIndex] + j + m_vertexCount[outline.edges[t].meshIndex], m_blockCounters[outline.edges[t].meshIndex] + j + 1);
                    arrowRenderers[outline.edges[t].meshIndex].arrowMesh.AddTriangle(m_blockCounters[outline.edges[t].meshIndex] + j + 1, m_blockCounters[outline.edges[t].meshIndex] + j + m_vertexCount[outline.edges[t].meshIndex], m_blockCounters[outline.edges[t].meshIndex] + j + 1 + m_vertexCount[outline.edges[t].meshIndex]);
                }
                m_blockCounters[outline.edges[t].meshIndex] += outline.edges[t].points.Count;
            }
        }
        /// <summary>
        /// Shifts all functions along body without rendering anything.
        /// </summary>
        /// <param name="length">Body chunk length.</param>
        /// <param name="bodyLength">Total body length.</param>
        void ConsumeEmptySpace(float length,float bodyLength)
        {
            while (m_nextLength[m_nextLengthIndex] < length)
            {
                if (m_nextLengthIndex == 3)
                {
                    length -= m_nextLength[3];
                    m_bodyTraveled += m_nextLength[3];
                    for (int i = 0; i < 3; i++)
                    {
                        if (curveModes[i] != CurveFunctionMode.Fixed)
                        {
                            m_curveCurrentValue[i] = curveFunction[i].Evaluate(m_bodyTraveled * m_timeFormat[i]);
                        }
                        m_nextLength[i] -= m_nextLength[3];
                    }
                    m_currentColorKey = (m_currentColorKey + 1) % colors.Count;
                    m_colorValue = colors[m_currentColorKey].startColor;
                    m_nextLength[3] = colors[m_currentColorKey].length;
                    m_nextLength[4] = bodyLength / lod;
                    for (int i = 0; i < 5; i++)
                    {
                        if (m_nextLength[i] < m_nextLength[m_nextLengthIndex]) m_nextLengthIndex = i;
                    }
                }
                else if (m_nextLengthIndex == 4)
                {
                    length -= m_nextLength[4];
                    m_bodyTraveled += m_nextLength[4];
                    for (int i = 0; i < 3; i++)
                    {
                        if (curveModes[i] != CurveFunctionMode.Fixed)
                        {
                            m_curveCurrentValue[i] = curveFunction[i].Evaluate(m_bodyTraveled * m_timeFormat[i]);
                        }
                        m_nextLength[i] -= m_nextLength[4];
                    }
                    if (colorMode == OutlineBodyRendererColorMode.ColorAlongBody || colorMode == OutlineBodyRendererColorMode.ColorAlongBodyGradient)
                    {
                        m_colorValue = Color32.Lerp(m_colorValue, colors[m_currentColorKey].endColor, m_nextLength[4] / m_nextLength[3]);
                        m_nextLength[3] -= m_nextLength[4];
                    }
                    m_nextLength[4] = bodyLength / lod;
                    for (int i = 0; i < 5; i++)
                    {
                        if (m_nextLength[i] < m_nextLength[m_nextLengthIndex]) m_nextLengthIndex = i;
                    }
                }
                else
                {
                    for (int k = 0; k < 3; k++)
                    {
                        if (m_nextLengthIndex == k)
                        {
                            length -= m_nextLength[k];
                            m_bodyTraveled += m_nextLength[k];
                            for (int i = 0; i < 3; i++)
                            {
                                if (curveModes[i] != CurveFunctionMode.Fixed)
                                {
                                    m_curveCurrentValue[i] = curveFunction[i].Evaluate(m_bodyTraveled * m_timeFormat[i]);
                                }
                                m_nextLength[i] -= m_nextLength[k];
                            }
                            if (colorMode == OutlineBodyRendererColorMode.ColorAlongBody || colorMode == OutlineBodyRendererColorMode.ColorAlongBodyGradient)
                            {
                                m_curveCurrentValue[k] = curveFunction[k].Evaluate(m_bodyTraveled * m_timeFormat[k]);
                                m_colorValue = Color32.Lerp(m_colorValue, colors[m_currentColorKey].endColor, m_nextLength[k] / m_nextLength[3]);
                            }

                            m_currentFunctionKey[k]++;
                            if (m_currentFunctionKey[k] == curveFunction[k].keys.Length) m_currentFunctionKey[k] = 1;
                            m_nextLength[k] = curveFunction[k].keys[m_currentFunctionKey[k]].time - curveFunction[k].keys[m_currentFunctionKey[k] - 1].time;
                            m_nextLength[4] = bodyLength / lod;
                            for (int i = 0; i < 5; i++)
                            {
                                if (m_nextLength[i] < m_nextLength[m_nextLengthIndex]) m_nextLengthIndex = i;
                            }
                            break;
                        }
                    }
                }
                if (m_nextLengthIndex == 5) break;
            }
            if (colorMode == OutlineBodyRendererColorMode.ColorAlongBody || colorMode == OutlineBodyRendererColorMode.ColorAlongBodyGradient)
            {
                m_nextLength[3] -= length;
                m_nextLength[4] = bodyLength / lod;
                m_colorValue = Color32.Lerp(colors[m_currentColorKey].startColor, colors[m_currentColorKey].endColor, 1 - m_nextLength[3] / colors[m_currentColorKey].length);
            }
            m_bodyTraveled += length;
            for (int i = 0; i < 3; i++)
            {
                if (curveModes[i] != CurveFunctionMode.Fixed)
                {
                    m_nextLength[i] -= length;
                    m_curveCurrentValue[i] = curveFunction[i].Evaluate(m_bodyTraveled * m_timeFormat[i]);
                    m_nextLength[4] = bodyLength / lod;
                }
            }
            for (int i = 0; i < 5; i++)
            {
                if (m_nextLength[i] < m_nextLength[m_nextLengthIndex]) m_nextLengthIndex = i;
            }
        }
        /// <summary>
        /// Generates extrude chunk in the extrude chain along path.
        /// </summary>
        /// <param name="arrowRenderers">Target arrow renderers with arrow meshes.</param>
        /// <param name="startPoint">Chunk start position.</param>
        /// <param name="endPoint">Chunk end position.</param>
        /// <param name="startRot">Chunk outlien start rotation.</param>
        /// <param name="endRot">Chunk outline end rotaiton.</param>
        /// <param name="normalRotation">Rotations of normals if chunk is not smooth.</param>
        /// <param name="uVY">Chunk start uv y positon.</param>
        /// <param name="startScale">Chunk outline start scale.</param>
        /// <param name="endScale">Chunk outline end scale.</param>
        /// <param name="smoothPath">Reuse last outline extrude vertices or generate new ones for sharp edges along path.</param>
        /// <param name="bodyLength">Total length of the body.</param>
        void AddExtrudedOutline(List<ArrowRenderer> arrowRenderers, Vector3 startPoint, Vector3 endPoint, Quaternion startRot, Quaternion endRot,Quaternion normalRotation, float uVY, Vector2 startScale, Vector2 endScale, bool smoothPath,float bodyLength)
        {
            if (smoothPath) {
                normalRotation = endRot;
            }
            if (!m_firstBlockRendered || !smoothPath)
            {
                RenderVertices(arrowRenderers, startPoint, startRot, startScale, normalRotation, uVY);
            }
            float chunkLength = (endPoint - startPoint).magnitude;
            Vector3 dir = (endPoint - startPoint).normalized;
            uVY = uVY + (endPoint - startPoint).magnitude;
            while(m_nextLength[m_nextLengthIndex]<chunkLength)
            {
                if (m_nextLengthIndex==3)
                {
                    chunkLength -= m_nextLength[3];
                    m_bodyTraveled+= m_nextLength[3];
                    for (int i = 0; i < 3; i++)
                    {
                        if (curveModes[i] != CurveFunctionMode.Fixed)
                        {
                            m_curveCurrentValue[i] = curveFunction[i].Evaluate(m_bodyTraveled * m_timeFormat[i]);
                        }
                        m_nextLength[i] -= m_nextLength[3];
                    }
                    m_colorValue = colors[m_currentColorKey].endColor;
                    RenderVertices(arrowRenderers,
                        endPoint-dir*chunkLength,
                        normalRotation,
                        Vector2.one,
                        normalRotation,
                        uVY-chunkLength);
                    m_currentColorKey = (m_currentColorKey + 1) % colors.Count;

                    m_colorValue = colors[m_currentColorKey].startColor;
                    RenderTriangles(arrowRenderers);
                    RenderVertices(arrowRenderers,
                        endPoint - dir * chunkLength,
                        normalRotation,
                        Vector2.one,
                        normalRotation,
                        uVY - chunkLength);
                    m_nextLength[3] = colors[m_currentColorKey].length;
                    m_nextLength[4] = bodyLength / lod;
                    for (int i = 0; i < 5; i++)
                    {
                        if (m_nextLength[i] < m_nextLength[m_nextLengthIndex]) m_nextLengthIndex = i;
                    }
                }
                else if (m_nextLengthIndex == 4)
                {
                    chunkLength -= m_nextLength[4];
                    m_bodyTraveled += m_nextLength[4];
                    for (int i = 0; i < 3; i++)
                    {
                        if (curveModes[i] != CurveFunctionMode.Fixed)
                        {
                            m_curveCurrentValue[i] = curveFunction[i].Evaluate(m_bodyTraveled * m_timeFormat[i]);
                        }
                        m_nextLength[i] -= m_nextLength[4];
                    }
                    if (colorMode == OutlineBodyRendererColorMode.ColorAlongBody || colorMode == OutlineBodyRendererColorMode.ColorAlongBodyGradient)
                    {
                        m_colorValue = Color32.Lerp(m_colorValue, colors[m_currentColorKey].endColor, m_nextLength[4] / m_nextLength[3]);
                        m_nextLength[3] -= m_nextLength[4];
                    }
                    RenderVertices(arrowRenderers,
                        endPoint - dir * chunkLength,
                        normalRotation,
                        Vector2.one,
                        normalRotation,
                        uVY - chunkLength);
                    RenderTriangles(arrowRenderers);
                    m_nextLength[4] = bodyLength / lod;
                    for (int i = 0; i < 5; i++)
                    {
                        if (m_nextLength[i] < m_nextLength[m_nextLengthIndex]) m_nextLengthIndex = i;
                    }
                }
                else
                {
                    for (int k = 0; k < 3; k++)
                    {
                        if (m_nextLengthIndex == k)
                        {
                            chunkLength -= m_nextLength[k];
                            m_bodyTraveled += m_nextLength[k];
                            for (int i = 0; i < 3; i++)
                            {
                                if (curveModes[i] != CurveFunctionMode.Fixed)
                                {
                                    m_curveCurrentValue[i] = curveFunction[i].Evaluate(m_bodyTraveled * m_timeFormat[i]);
                                }
                                m_nextLength[i] -= m_nextLength[k];
                            }
                            if (colorMode == OutlineBodyRendererColorMode.ColorAlongBody || colorMode == OutlineBodyRendererColorMode.ColorAlongBodyGradient)
                            {
                                m_curveCurrentValue[k] = curveFunction[k].Evaluate(m_bodyTraveled * m_timeFormat[k]);
                                m_colorValue = Color32.Lerp(m_colorValue, colors[m_currentColorKey].endColor, m_nextLength[k] / m_nextLength[3]);
                            }
                            RenderVertices(arrowRenderers,
                                endPoint - dir * chunkLength,
                                normalRotation,
                                Vector2.one,
                                normalRotation,
                                uVY - chunkLength);
                            m_currentFunctionKey[k]++;
                            if (m_currentFunctionKey[k] == curveFunction[k].keys.Length) m_currentFunctionKey[k] = 1;
                            m_nextLength[k] = curveFunction[k].keys[m_currentFunctionKey[k]].time - curveFunction[k].keys[m_currentFunctionKey[k] - 1].time;
                            m_nextLength[4] = bodyLength / lod;
                            RenderTriangles(arrowRenderers);
                            for (int i = 0; i < 5; i++)
                            {
                                if (m_nextLength[i] < m_nextLength[m_nextLengthIndex]) m_nextLengthIndex = i;
                            }
                            break;
                        }
                    }
                }
                if (m_nextLengthIndex == 5) break;
            }
            if (colorMode == OutlineBodyRendererColorMode.ColorAlongBody || colorMode == OutlineBodyRendererColorMode.ColorAlongBodyGradient) {
                m_nextLength[3] -= chunkLength;
                m_nextLength[4] = bodyLength / lod + 1;
                m_colorValue = Color32.Lerp(colors[m_currentColorKey].startColor, colors[m_currentColorKey].endColor, 1 - m_nextLength[3]/ colors[m_currentColorKey].length);
            }
            m_bodyTraveled += chunkLength;
            for (int i = 0; i < 3; i++)
            {
                if (curveModes[i] != CurveFunctionMode.Fixed)
                {
                    m_nextLength[i] -= chunkLength;
                    m_curveCurrentValue[i] = curveFunction[i].Evaluate(m_bodyTraveled * m_timeFormat[i]);
                    m_nextLength[4] = bodyLength / lod;
                }
            }
            for (int i = 0; i < 5; i++)
            {
                if (m_nextLength[i] < m_nextLength[m_nextLengthIndex]) m_nextLengthIndex = i;
            }
            RenderVertices(arrowRenderers, endPoint, endRot,endScale, normalRotation, uVY);
            RenderTriangles(arrowRenderers);
            m_firstBlockRendered = true;
        }
        /// <summary>
        /// Renders connection between body end and tip start instead of body break and tip back face.
        /// </summary>
        /// <param name="arrowRenderers">Target arrow renderers with target meshes</param>
        /// <param name="position">Mesh start pivot point.</param>
        /// <param name="rotation">Mesh direction rotation.</param>
        /// <param name="innerScale">Scale of body vertices.</param>
        /// <param name="outerScale">Scale of tip vertices.</param>
        /// <param name="tipConnectors">Meshes to render.</param>
        void RenderTipConnector(List<ArrowRenderer> arrowRenderers, Vector3 position, Quaternion rotation, Vector3 innerScale, Vector3 outerScale, List<BodyTipMeshItem> tipConnectors)
        {
            rotation = rotation * Quaternion.AngleAxis(m_curveCurrentValue[2], Vector3.forward);
            for (int t = 0; t < tipConnectors.Count; t++)
            {
                tipConnectors[t].AddToMeshInnerOuter(arrowRenderers[tipConnectors[t].meshIndex].arrowMesh,position,rotation,innerScale,outerScale,m_colorValue, tipConnectors[t].innerVertices);
            }
        }
        /// <summary>
        /// Renders mesh for generate body break and dash breaks.
        /// </summary>
        /// <param name="arrowRenderers">Target arrow renderers with target arrow mesh.</param>
        /// <param name="position">Mesh start pivot point.</param>
        /// <param name="rotation">Mesh direction rotation.</param>
        /// <param name="breakMeshes">Meshes to render.</param>
        void RenderBodyBreak(List<ArrowRenderer> arrowRenderers, Vector3 position, Quaternion rotation, List<MeshItem> breakMeshes)
        {
            rotation = rotation * Quaternion.AngleAxis(m_curveCurrentValue[2], Vector3.forward);
            for (int t = 0; t < breakMeshes.Count; t++)
            {
                breakMeshes[t].AddToMesh(arrowRenderers[breakMeshes[t].meshIndex].arrowMesh, position, rotation, new Vector3(m_lastTurnScale.x * m_curveCurrentValue[0], m_lastTurnScale.y * m_curveCurrentValue[1],1), m_colorValue);
            }
        }
    }

}