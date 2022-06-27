using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    /// <summary>
    /// Position rotation and length for body renderer chunk.
    /// </summary>
    [System.Serializable]
    public class BodyRendererInputData
    {
        /// <summary>
        /// Chunk start position.
        /// </summary>
        public Vector3 startPosition;
        /// <summary>
        /// Chunk end position.
        /// </summary>
        public Vector3 endPosition;
        /// <summary>
        /// Rotation of previous direction.
        /// </summary>
        public Quaternion lastRotation;
        /// <summary>
        /// Rotation of current direction.
        /// </summary>
        public Quaternion rotation;
        /// <summary>
        /// Rotation of next direction.
        /// </summary>
        public Quaternion nextRotation;
        /// <summary>
        /// Total chunk length.
        /// </summary>
        public float chunkLength;
        /// <summary>
        /// Length of all previous chunks of current body.
        /// </summary>
        public float traveledPath = 0;
        /// <summary>
        /// Total body length.
        /// </summary>
        public float bodyLength = 0;
        /// <summary>
        /// If true it is the last chunk of the body.
        /// </summary>
        public bool lastRender = false;
    }
}