using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    /// <summary>
    /// Container for mesh filter and mesh that gets added to the filter to keep them together.
    /// </summary>
    [System.Serializable]
    public class ArrowRenderer
    {
        /// <summary>
        /// A procedural arrow mesh used to hold mesh data and update arrow mesh to unity mesh
        /// </summary>
        public ArrowMesh arrowMesh = new ArrowMesh();
        /// <summary>
        /// Reference to the mesh filter where the mesh will be added on arrow update.
        /// </summary>
        public MeshFilter meshFilter;
        /// <summary>
        /// If mesh is cached on update everything after the last cached vertex gets deleted.
        /// </summary>
        [HideInInspector]
        public int lastCachedVertex = 0;
        /// <summary>
        /// If mesh is cached on update everything after the last cached triangle gets deleted.
        /// </summary>
        [HideInInspector]
        public int lastCachedTriangle = 0;
    }
}