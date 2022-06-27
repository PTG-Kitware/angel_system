using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    /// <summary>
    /// Renders an arrow tip using unity preloaded mesh.
    /// </summary>
    public class MeshTip : ArrowTip
    {
        /// <summary>
        /// Scale along length
        /// </summary>
        [Min(0)]
        public float lengthScale=1;
        /// <summary>
        /// List of meshes to render
        /// </summary>
        public List<MeshItem> meshes = new List<MeshItem>() {};
        /// <summary>
        /// Face that is rendered if there is no body behind the tip. 
        /// </summary>
        public List<MeshItem> backFaceMeshes = new List<MeshItem>() {};
        public override int MaxMeshIndex()
        {
            int maxIndex = 0;
            foreach (MeshItem m in meshes)
            {
                if (m.meshIndex > maxIndex) maxIndex = m.meshIndex;
            }
            foreach (MeshItem m in backFaceMeshes)
            {
                if (m.meshIndex > maxIndex) maxIndex = m.meshIndex;
            }
            return maxIndex;
        }
        public override void GenerateTip(List<ArrowRenderer> arrowRenderers, BodyAndTipPathData tipData, bool tail, bool renderBackFace, Color32 inputColor, float inputRoll)
        {
            Vector3 startPosition = tipData.bodyEndPoint;
            Quaternion tipRotation = tipData.bodyEndRotation;
            if (tail)
            {
                startPosition = tipData.bodyStartPoint;
                tipRotation = tipData.bodyStartRotation;
                //rotate 180
                tipRotation = Quaternion.LookRotation(tipRotation * Vector3.back, tipRotation * Vector3.up);
            }
            Vector3 trueSize = new Vector3(size.x,size.y,lengthScale) * tipData.sizeFactor;
            if (renderBackFace)
            {
                for (int t = 0; t < backFaceMeshes.Count; t++)
                {
                    backFaceMeshes[t].AddToMesh(arrowRenderers[backFaceMeshes[t].meshIndex].arrowMesh, startPosition, tipRotation * Quaternion.AngleAxis(inputRoll, Vector3.forward), trueSize, inputColor);
                }
            }
            for (int t = 0; t < meshes.Count; t++)
            {
                meshes[t].AddToMesh(arrowRenderers[meshes[t].meshIndex].arrowMesh,startPosition,tipRotation * Quaternion.AngleAxis(inputRoll, Vector3.forward), trueSize, inputColor);
            }
        }
        public override Color32 GetDefaultInputColor()
        {
            if (meshes.Count != 0)
            {
                return meshes[0].color;
            }
            if (backFaceMeshes.Count != 0)
            {
                return backFaceMeshes[0].color;
            }
            return Color.white;
        }
    }
}