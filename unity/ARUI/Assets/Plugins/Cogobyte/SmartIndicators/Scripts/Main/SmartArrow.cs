using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    /// <summary>
    /// Main arrow script that generates mesh vertices and triangles based on arrow path body and tips.
    /// </summary>
    [RequireComponent (typeof(MeshFilter))]
    [RequireComponent(typeof(MeshRenderer))]
    public class SmartArrow : MonoBehaviour
    {
        /// <summary>
        /// All arrow renderers with mesh filters for multiple materials.
        /// </summary>
        public List<ArrowRenderer> arrowRenderers = new List<ArrowRenderer>();
        /// <summary>
        /// Path through witch arrow passes.
        /// </summary>
        public ArrowPath arrowPath;
        /// <summary>
        /// Start position of the arrow within arrow path.
        /// </summary>
        public float StartPercentage = 0;
        /// <summary>
        /// End position of the arrow within arrow path.
        /// </summary>
        public float EndPercentage = 1;
        /// <summary>
        /// Referece to arrow tail definition.
        /// </summary>
        public ArrowTip arrowTail;
        /// <summary>
        /// Reference to arrow head definition.
        /// </summary>
        public ArrowTip arrowHead;
        /// <summary>
        /// Body lengths are moved backwards or forwards based on displacement.
        /// </summary>
        public float displacement = 0;
        /// <summary>
        /// All bodies used to render the arrow.
        /// </summary>
        public List<MultiBodyItem> bodyRenderers = new List<MultiBodyItem>() { };
        /// <summary>
        /// Reference to this arrow transform.
        /// </summary>
        Transform arrowTransform;
        /// <summary>
        /// Index of the current rendered body.
        /// </summary>
        int m_bodyIndex = 0;
        /// <summary>
        /// Remaining space of the current body.
        /// </summary>
        float m_bodyChunkLength = 0;
        /// <summary>
        /// Data object for each rendered body chunk.
        /// </summary>
        BodyRendererInputData m_bodyChunkData = new BodyRendererInputData();
        /// <summary>
        /// Arrow body and tips start and end postions and rotations.
        /// </summary>
        BodyAndTipPathData m_bodyTipData = new BodyAndTipPathData();

        public void Awake()
        {
            foreach(ArrowRenderer r in arrowRenderers)
            {
                r.arrowMesh.mesh = new Mesh();
            }
            Validate();
        }

        /// <summary>
        /// Generates a new arrow renderer with mesh filter for additional materials.
        /// </summary>
        /// <param name="material">Material of the new arrow renderer.</param>
        /// <returns>Game object with mesh filter and mesh renderer.</returns>
        public GameObject AddMeshFilter(Material material)
        {
            GameObject meshFilterGameObject = new GameObject();
            meshFilterGameObject.name = "ArrowMesh" + arrowRenderers.Count;
            meshFilterGameObject.transform.parent = arrowTransform;
            meshFilterGameObject.transform.localPosition = Vector3.zero;
            meshFilterGameObject.transform.localRotation = Quaternion.identity;
            MeshFilter m = meshFilterGameObject.AddComponent<MeshFilter>();
            arrowRenderers.Add(new ArrowRenderer() { meshFilter = m});
            MeshRenderer r = meshFilterGameObject.AddComponent<MeshRenderer>();
            r.sharedMaterial = material;
            return meshFilterGameObject;
        }
        /// <summary>
        /// Ensures all correct values on arror bodies tips and this script.
        /// </summary>
        public void Validate()
        {
            if (arrowTransform == null) arrowTransform = transform;
            if (arrowRenderers == null) arrowRenderers = new List<ArrowRenderer>();
            if (arrowRenderers.Count == 0)
            {
                arrowRenderers.Add(new ArrowRenderer() {meshFilter = GetComponent<MeshFilter>()});
            }
            if (bodyRenderers == null) bodyRenderers = new List<MultiBodyItem>() { };
            if (bodyRenderers.Count == 1)
            {
                bodyRenderers[0].percentage = 1;
                bodyRenderers[0].lengthMode = MultiBodyItem.LengthMode.Fill;
            }
            else
            {
                foreach (MultiBodyItem b in bodyRenderers)
                {
                    if (b.length <= SmartArrowUtilities.Utilities.errorRate)
                    {
                        b.length = 0.1f;
                    }
                }
            }
        }
        public void OnValidate()
        {
            Validate();
        }

        /// <summary>
        /// Updates arrow path and generates arrow vertices and triangles. 
        /// </summary>
        /// <param name="updatePath">Updates path and arrow if true or arrow only if false.</param>
        public void UpdateArrow(bool updatePath = true,bool updateBodyTipData=true)
        {
            Validate();
            for (int i = 0; i < arrowRenderers.Count; i++)
            {
                if (arrowRenderers[i].meshFilter != null)
                {
                    arrowRenderers[i].arrowMesh.ClearMesh();
                }
            }
            if (arrowPath != null)
            {
                float ArrowTailRadius = 0;
                float ArrowHeadRadius = 0;
                float ArrowTailRoll = 0;
                float ArrowHeadRoll = 0;
                bool tailFollowPath = false;
                bool headFollowPath = false;
                bool renderTailBackFace = true;
                bool renderHeadBackFace = true;
                int maxMeshIndex = 0;
                if (arrowTail != null)
                {
                    ArrowTailRadius = arrowTail.GetLength();
                    tailFollowPath = arrowTail.FollowsPath();
                    maxMeshIndex = arrowTail.MaxMeshIndex();
                }
                if (arrowHead != null)
                {
                    ArrowHeadRadius = arrowHead.GetLength();
                    headFollowPath = arrowHead.FollowsPath();
                    int maxMesh = arrowHead.MaxMeshIndex();
                    if(maxMesh>maxMeshIndex) maxMeshIndex=maxMesh;
                }
                if (updatePath)
                {
                    arrowPath.CalculatePath();
                }
                if (updateBodyTipData)
                {
                    arrowPath.GetBodyAndTipData(this, m_bodyTipData);
                }
                if (!m_bodyTipData.noRender)
                {
                    Color32 tailInputColor = Color.white;
                    Color32 headInputColor = Color.white;
                    if (arrowTail != null)
                    {
                        tailInputColor = arrowTail.GetDefaultInputColor();
                    }
                    if (arrowHead != null)
                    {
                        headInputColor = arrowHead.GetDefaultInputColor();
                    }
                    if (!m_bodyTipData.noBody)
                    {
                        if (bodyRenderers.Count > 0)
                        {
                            foreach(MultiBodyItem mB in bodyRenderers)
                            {
                                if (mB.bodyRenderer != null)
                                {
                                    int maxMesh = mB.bodyRenderer.MaxMeshIndex();
                                    if (maxMesh > maxMeshIndex) maxMeshIndex = maxMesh;
                                }
                            }
                            while (maxMeshIndex >= arrowRenderers.Count)
                            {
                                AddMeshFilter(GetComponent<MeshRenderer>().sharedMaterial);
                            }
                            float BodyLength = m_bodyTipData.bodyLength;
                            float vol = BodyLength;
                            int t = 0;
                            foreach (MultiBodyItem mb in bodyRenderers)
                            {
                                if (mb.lengthMode == MultiBodyItem.LengthMode.Percentage)
                                {
                                    if (mb.percentage < SmartArrowUtilities.Utilities.errorRate)
                                    {
                                        mb.length = 0.01f * BodyLength;
                                    }
                                    else
                                    {
                                        mb.length = mb.percentage * BodyLength;
                                    }
                                    vol -= mb.length;
                                }
                                else if (mb.lengthMode == MultiBodyItem.LengthMode.Fixed)
                                {
                                    vol -= mb.length;
                                }
                                else if (mb.lengthMode == MultiBodyItem.LengthMode.Fill)
                                {
                                    t++;
                                }
                            }
                            foreach (MultiBodyItem mb in bodyRenderers)
                            {
                                if (mb.lengthMode == MultiBodyItem.LengthMode.Fill)
                                {
                                    if (t == 1 && bodyRenderers.Count==1)
                                        mb.length = vol + 3*SmartArrowUtilities.Utilities.errorRate;
                                    else
                                        mb.length = vol / t + SmartArrowUtilities.Utilities.errorRate;
                                }
                                if (mb.length < 0.0001f) { mb.length = 0.0001f; }
                            }
                            #region displacment
                            m_bodyIndex = 0;
                            m_bodyChunkLength = bodyRenderers[m_bodyIndex].length;
                            float BodyDisplacement = displacement;
                            if (BodyDisplacement > 0)
                            {
                                while (m_bodyChunkLength < BodyDisplacement)
                                {
                                    BodyDisplacement -= m_bodyChunkLength;
                                    m_bodyIndex = (m_bodyIndex + 1) % bodyRenderers.Count;
                                    m_bodyChunkLength = bodyRenderers[m_bodyIndex].length;
                                }
                            }
                            else if (BodyDisplacement < 0)
                            {
                                m_bodyIndex = (m_bodyIndex - 1 + bodyRenderers.Count) % bodyRenderers.Count;
                                m_bodyChunkLength = bodyRenderers[m_bodyIndex].length;
                                while (m_bodyChunkLength < -BodyDisplacement)
                                {
                                    BodyDisplacement += m_bodyChunkLength;
                                    m_bodyIndex = (m_bodyIndex - 1 + bodyRenderers.Count) % bodyRenderers.Count;
                                    m_bodyChunkLength = bodyRenderers[m_bodyIndex].length;
                                }
                                BodyDisplacement = -BodyDisplacement;
                            }
                            m_bodyChunkLength -= BodyDisplacement;
                            #endregion displacment
                            if (bodyRenderers[m_bodyIndex].bodyRenderer != null)
                            {
                                bodyRenderers[m_bodyIndex].bodyRenderer.InitializeOutline(arrowRenderers,bodyRenderers[m_bodyIndex].displacement, bodyRenderers[m_bodyIndex].colorDisplacement, m_bodyChunkLength);
                                tailInputColor = bodyRenderers[m_bodyIndex].bodyRenderer.GetCurrentColor();
                                ArrowTailRoll = bodyRenderers[m_bodyIndex].bodyRenderer.GetCurrentRoll();
                            }
                            float PathChunkLength = 0;
                            m_bodyChunkData.lastRender = false;
                            m_bodyChunkData.startPosition = m_bodyTipData.bodyStartPoint;
                            m_bodyChunkData.lastRotation = m_bodyChunkData.rotation = m_bodyTipData.bodyStartRotation;
                            m_bodyChunkData.nextRotation = m_bodyTipData.calculatedRotations[m_bodyTipData.bodyStartIndex];
                            m_bodyChunkData.bodyLength = m_bodyTipData.bodyLength;
                            m_bodyChunkData.traveledPath = 0;
                            if (arrowTail != null)
                            {
                                m_bodyChunkData.traveledPath = arrowTail.size.z;
                            }
                            if (bodyRenderers[m_bodyIndex].bodyRenderer != null)
                            {
                                renderTailBackFace = bodyRenderers[m_bodyIndex].bodyRenderer.GenerateTailConnection(arrowRenderers, m_bodyChunkData, arrowTail);
                            }
                            if (m_bodyTipData.bodyStartIndex <= m_bodyTipData.bodyEndIndex)
                            {
                                PathChunkLength = (m_bodyTipData.calculatedPath[m_bodyTipData.bodyStartIndex] - m_bodyChunkData.startPosition).magnitude;
                                GenerateMultiBodySwitch(ref PathChunkLength);
                                m_bodyChunkData.endPosition = m_bodyTipData.calculatedPath[m_bodyTipData.bodyStartIndex];
                                PathChunkLength = (m_bodyChunkData.endPosition - m_bodyChunkData.startPosition).magnitude;
                                if (bodyRenderers[m_bodyIndex].bodyRenderer != null)
                                {
                                    bodyRenderers[m_bodyIndex].bodyRenderer.GenerateBody(arrowRenderers, m_bodyChunkData, PathChunkLength);
                                    headInputColor = bodyRenderers[m_bodyIndex].bodyRenderer.GetCurrentColor();
                                }
                                m_bodyChunkLength -= PathChunkLength;
                                for (int i = m_bodyTipData.bodyStartIndex + 1; i <= m_bodyTipData.bodyEndIndex; i++)
                                {
                                    PathChunkLength = (m_bodyTipData.calculatedPath[i] - m_bodyTipData.calculatedPath[i - 1]).magnitude;
                                    m_bodyChunkData.startPosition = m_bodyTipData.calculatedPath[i - 1];
                                    m_bodyChunkData.rotation = m_bodyTipData.calculatedRotations[i - 1];
                                    m_bodyChunkData.nextRotation = m_bodyTipData.calculatedRotations[i];
                                    GenerateMultiBodySwitch(ref PathChunkLength);
                                    m_bodyChunkData.endPosition = m_bodyTipData.calculatedPath[i];
                                    if (bodyRenderers[m_bodyIndex].bodyRenderer != null)
                                    {
                                        bodyRenderers[m_bodyIndex].bodyRenderer.GenerateBody(arrowRenderers, m_bodyChunkData, PathChunkLength);
                                        headInputColor = bodyRenderers[m_bodyIndex].bodyRenderer.GetCurrentColor();
                                    }
                                    m_bodyChunkLength -= PathChunkLength;
                                    m_bodyChunkData.traveledPath += PathChunkLength;
                                }
                                m_bodyChunkData.startPosition = m_bodyTipData.calculatedPath[m_bodyTipData.bodyEndIndex];
                                m_bodyChunkData.rotation = m_bodyTipData.calculatedRotations[m_bodyTipData.bodyEndIndex + 1];
                                m_bodyChunkData.lastRotation = m_bodyChunkData.nextRotation;
                            }

                            for (int i = 0; i < arrowRenderers.Count; i++)
                            {
                                arrowRenderers[i].lastCachedVertex = arrowRenderers[i].arrowMesh.vertices.Count;
                                arrowRenderers[i].lastCachedTriangle = arrowRenderers[i].arrowMesh.triangles.Count;
                            }
                            m_bodyChunkData.endPosition = m_bodyTipData.bodyEndPoint;
                            PathChunkLength = (m_bodyChunkData.endPosition - m_bodyChunkData.startPosition).magnitude;
                            GenerateMultiBodySwitch(ref PathChunkLength);
                            m_bodyChunkData.endPosition = m_bodyTipData.bodyEndPoint;
                            m_bodyChunkData.lastRotation = m_bodyTipData.bodyEndRotation;
                            m_bodyChunkData.nextRotation = m_bodyTipData.bodyEndRotation;
                            m_bodyChunkData.lastRender = true;
                            if (bodyRenderers[m_bodyIndex].bodyRenderer != null)
                            {
                                bodyRenderers[m_bodyIndex].bodyRenderer.GenerateBody(arrowRenderers, m_bodyChunkData, PathChunkLength);
                                renderHeadBackFace = bodyRenderers[m_bodyIndex].bodyRenderer.GenerateHeadConnection(arrowRenderers, m_bodyChunkData, arrowHead);
                                ArrowHeadRoll = bodyRenderers[m_bodyIndex].bodyRenderer.GetCurrentRoll();
                                headInputColor = bodyRenderers[m_bodyIndex].bodyRenderer.GetCurrentColor();
                            }
                        }
                    }
                    if (arrowTail != null)
                    {
                        while (maxMeshIndex >= arrowRenderers.Count)
                        {
                            AddMeshFilter(GetComponent<MeshRenderer>().sharedMaterial);
                        }
                        arrowTail.GenerateTip(arrowRenderers,  m_bodyTipData, true, renderTailBackFace,tailInputColor,ArrowTailRoll);
                    }
                    if (arrowHead != null)
                    {
                        while (maxMeshIndex >= arrowRenderers.Count)
                        {
                            AddMeshFilter(GetComponent<MeshRenderer>().sharedMaterial);
                        }
                        arrowHead.GenerateTip(arrowRenderers,  m_bodyTipData, false, renderHeadBackFace,headInputColor,ArrowHeadRoll);
                    }
                    if (arrowPath.local)
                    {
                        Vector3 positionOffset = -arrowTransform.position;
                        Quaternion rotationOffset = Quaternion.Inverse(arrowTransform.rotation);
                        foreach (ArrowRenderer m in arrowRenderers)
                        {
                            for (int i = 0; i < m.arrowMesh.vertices.Count; i++)
                            {
                                m.arrowMesh.vertices[i] = positionOffset + rotationOffset * m.arrowMesh.vertices[i];
                                m.arrowMesh.normals[i] = rotationOffset * m.arrowMesh.normals[i];
                            }
                        }
                    }
                    else
                    {
                        Vector3 positionOffset = -arrowTransform.position;
                        Quaternion rotationOffset = Quaternion.Inverse(arrowTransform.rotation);
                        foreach (ArrowRenderer m in arrowRenderers)
                        {
                            for (int i = 0; i < m.arrowMesh.vertices.Count; i++)
                            {
                                m.arrowMesh.vertices[i] = rotationOffset * (positionOffset + m.arrowMesh.vertices[i]);
                                m.arrowMesh.normals[i] = rotationOffset * m.arrowMesh.normals[i];
                            }
                        }
                    }
                }
            }
            for (int i = 0; i < arrowRenderers.Count; i++)
            {
                if (arrowRenderers[i].meshFilter != null)
                {
                    arrowRenderers[i].arrowMesh.PushToMesh(arrowRenderers[i].meshFilter);
                }
            }
        }
        /// <summary>
        /// Switches to next body renderer.
        /// </summary>
        /// <param name="PathChunkLength">Remaining chunk of current body.</param>
        void GenerateMultiBodySwitch(ref float PathChunkLength) {
            while (PathChunkLength > m_bodyChunkLength + SmartArrowUtilities.Utilities.errorRate)
            {
                m_bodyChunkData.endPosition = m_bodyChunkData.startPosition + m_bodyChunkData.rotation * Vector3.forward * m_bodyChunkLength;
                if (bodyRenderers[m_bodyIndex].bodyRenderer != null)
                {
                    bodyRenderers[m_bodyIndex].bodyRenderer.GenerateBody(arrowRenderers, m_bodyChunkData, m_bodyChunkLength);
                    bodyRenderers[m_bodyIndex].bodyRenderer.GenerateBodyBreak(arrowRenderers, m_bodyChunkData, true);
                }
                m_bodyChunkData.traveledPath += m_bodyChunkLength;
                PathChunkLength -= m_bodyChunkLength;
                m_bodyIndex = (m_bodyIndex + 1) % bodyRenderers.Count;
                m_bodyChunkLength = bodyRenderers[m_bodyIndex].length;
                m_bodyChunkData.bodyLength = bodyRenderers[m_bodyIndex].length;
                m_bodyChunkData.startPosition = m_bodyChunkData.endPosition;
                if (bodyRenderers[m_bodyIndex].bodyRenderer != null)
                {
                    bodyRenderers[m_bodyIndex].bodyRenderer.InitializeOutline(arrowRenderers,bodyRenderers[m_bodyIndex].displacement, bodyRenderers[m_bodyIndex].colorDisplacement, m_bodyChunkLength);
                    bodyRenderers[m_bodyIndex].bodyRenderer.GenerateBodyBreak(arrowRenderers, m_bodyChunkData, false);
                }
            }
        }
  
    }
}