using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators { 
    /// <summary>
    /// Main abstract class for arrow tips.
    /// Inherit to extend with new tip types.
    /// </summary>
    public abstract class ArrowTip : MonoBehaviour
    {
        /// <summary>
        /// Size of the tip. X is the width scale, y is the height scale and z is the length that takes up arrow path length.
        /// Size of the tip will shrink if there is not enough space for length. 
        /// </summary>
        public Vector3 size = new Vector3(1, 1, 1);
        /// <summary>
        /// Returns the length of the tip used for arrow path calculations for tip start points.
        /// </summary>
        /// <returns>Total length of the tip.</returns>
        public virtual float GetLength()
        {
            return size.z;
        }
        /// <summary>
        /// Returns true if the tip need path points and rotations or false if the tip straight is going straight from point A to point B.
        /// </summary>
        /// <returns>Returns true if the tip need path points and rotations or false if the tip straight is going straight from point A to point B.</returns>
        public virtual bool FollowsPath()
        {
            return false;
        }
        /// <summary>
        /// Returns the maximum mesh filter index that tip vertices will be rendered to.
        /// </summary>
        /// <returns>Maximum mesh filter index</returns>
        public abstract int MaxMeshIndex();
        /// <summary>
        /// Gets a default color when no input color is provided by the body.
        /// </summary>
        /// <returns>Default input color for this tip.</returns>
        public abstract Color32 GetDefaultInputColor();
        /// <summary>
        /// Generates vertices and triangles of the tip to specified arrow renderers.
        /// </summary>
        /// <param name="arrowRenderers">Destination arrow renderers with mesh filter.s</param>
        /// <param name="tipData">Tip data provided by the smart arrow based on arrow path data.</param>
        /// <param name="tail">Is it a tail or a head.</param>
        /// <param name="renderBackFace">Is the space behind tip without a body so it needs to render its own back face mesh.</param>
        /// <param name="inputColor">Color provided by the last body.</param>
        /// <param name="inputRoll">Final roll of the last body.</param>
        public abstract void GenerateTip(List<ArrowRenderer> arrowRenderers, BodyAndTipPathData tipData,bool tail, bool renderBackFace,Color32 inputColor,float inputRoll);
    }
}
