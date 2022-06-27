using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators {
    /// <summary>
    /// Data that describes body and tip lenghts postions and rotations along the path.
    /// </summary>
    public class BodyAndTipPathData
    {
        /// <summary>
        /// Body first point of the path.
        /// </summary>
        public Vector3 bodyStartPoint;
        /// <summary>
        /// First path point index after body start point.
        /// </summary>
        public int bodyStartIndex = 0;
        /// <summary>
        /// Rotation of body first point of the path.
        /// </summary>
        public Quaternion bodyStartRotation = Quaternion.identity;
        /// <summary>
        /// Body last point of the path.
        /// </summary>
        public Vector3 bodyEndPoint;
        /// <summary>
        /// First path point index before the body end point.
        /// </summary>
        public int bodyEndIndex = 0;
        /// <summary>
        /// Rotation of body last point of the path.
        /// </summary>
        public Quaternion bodyEndRotation = Quaternion.identity;
        /// <summary>
        /// List of all points of the path.
        /// </summary>
        public List<Vector3> calculatedPath;
        /// <summary>
        /// Rotations for each points of the path.
        /// </summary>
        public List<Quaternion> calculatedRotations;
        /// <summary>
        /// Start percentage path point after the path percentage application. 
        /// </summary>
        public Vector3 PathStartPoint;
        /// <summary>
        /// First path point index after start percentage point.
        /// </summary>
        public int pathStartIndex = 0;
        /// <summary>
        /// End percentage path point after the path percentage application. 
        /// </summary>
        public Vector3 PathEndPoint;
        /// <summary>
        /// First path point index before end percentage point.
        /// </summary>
        public int pathEndIndex = 0;
        /// <summary>
        /// If true there is no space for body within the path after the tips afe been substracted from the path.
        /// </summary>
        public bool noBody = false;
        /// <summary>
        /// If true there is no space to render anything.
        /// </summary>
        public bool noRender = false;
        /// <summary>
        /// Scale of both tips to fit the path length from 0 to 1.
        /// </summary>
        public float sizeFactor = 1;
        /// <summary>
        /// Total length of arrow body within the path.
        /// </summary>
        public float bodyLength = 0;
    }
}