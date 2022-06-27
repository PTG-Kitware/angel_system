using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    /// <summary>
    /// Abtract class for body renderer types. Generates arrow body vertices and triangles.
    /// </summary>
    [System.Serializable]
    public abstract class BodyRenderer:MonoBehaviour
    {
        /// <summary>
        /// Prepares body first chunk values based on displacments.
        /// </summary>
        /// <param name="arrowRenderers">Vertex counts between two chunks get extracted from arrow renderers.</param>
        /// <param name="bodyDisplacement">How much to move the start of the body forward or backward along the body.</param>
        /// <param name="colorDisplacement">How much to move color forward or backward along the body.</param>
        /// <param name="bodyLength">Total length of the arrow body.</param>
        public abstract void InitializeOutline(List<ArrowRenderer> arrowRenderers, float bodyDisplacement, float colorDisplacement, float bodyLength);
        /// <summary>
        /// Generates a body chunk.
        /// </summary>
        /// <param name="arrowRenderers">Arrow meshes to send generated vertices and triangles to.</param>
        /// <param name="bodyChunkData">Chunk parameters.</param>
        /// <param name="generateLength">Total length of the chunk.</param>
        public abstract void GenerateBody(List<ArrowRenderer> arrowRenderers,BodyRendererInputData bodyChunkData, float generateLength);
        /// <summary>
        /// Generates a mesh at the start or end of the body.
        /// </summary>
        /// <param name="arrowRenderers">Arrow meshes to send generated vertices and triangles to.</param>
        /// <param name="bodyChunkData">Chunk parameters holdin the positions, rotations and scale at the end of the body.</param>
        /// <param name="front">If true generates front break if false generates back break.</param>
        public abstract void GenerateBodyBreak(List<ArrowRenderer> arrowRenderers, BodyRendererInputData bodyChunkData, bool front);
        /// <summary>
        /// Generates a mesh connecing the arrow body and arrow tail.
        /// </summary>
        /// <param name="arrowRenderers">Arrow meshes to send generated vertices and triangles to.</param>
        /// <param name="bodyChunkData">Chunk parameters holdin the positions, rotations and scale at the end of the body.</param>
        /// <param name="arrowTip">Arrow tail script.</param>
        /// <returns></returns>
        public abstract bool GenerateTailConnection(List<ArrowRenderer> arrowRenderers, BodyRendererInputData bodyChunkData, ArrowTip arrowTip);
        /// <summary>
        /// Generates a mesh connecing the arrow body and arrow head.
        /// </summary>
        /// <param name="arrowRenderers">Arrow meshes to send generated vertices and triangles to.</param>
        /// <param name="bodyChunkData">Chunk parameters holdin the positions, rotations and scale at the end of the body.</param>
        /// <param name="arrowTip">Arrow head script.</param>
        /// <returns></returns>
        public abstract bool GenerateHeadConnection(List<ArrowRenderer> arrowRenderers, BodyRendererInputData bodyChunkData, ArrowTip arrowTip);
        /// <summary>
        /// Roll rotation at the current rendered block.
        /// </summary>
        /// <returns>Rotation around z axis in degrees.</returns>
        public abstract float GetCurrentRoll();
        public abstract Color32 GetCurrentColor();
        /// <summary>
        /// Returns the maximum index of mesh filter used for the body.
        /// </summary>
        /// <returns>Maximum index of mesh filter used.</returns>
        public abstract int MaxMeshIndex();
        /// <summary>
        /// Converts gradient to path color list.
        /// </summary>
        /// <param name="colors">Target path color list.</param>
        /// <param name="colorsGradient">Source gradient.</param>
        public static void SetColorsToGradient(List<PathColor> colors, Gradient colorsGradient)
        {
            colors.Clear();
            List<float> timeBreaks = new List<float>();
            for (int i = 0; i < colorsGradient.colorKeys.Length; i++)
            {
                timeBreaks.Add(colorsGradient.colorKeys[i].time);
            }
            for (int i = 0; i < colorsGradient.alphaKeys.Length; i++)
            {
                timeBreaks.Add(colorsGradient.alphaKeys[i].time);
            }
            timeBreaks.Sort();
            colors.Add(new PathColor() { colorMode = PathColor.PathColorMode.Percentage, percentage = 1, startColor = colorsGradient.Evaluate(0), endColor = colorsGradient.Evaluate(1) });
            for (int i = 0; i < timeBreaks.Count; i++)
            {
                if (timeBreaks[i] != 0 && timeBreaks[i] != 1)
                {
                    if (i == 0)
                    {
                        colors[0].percentage = timeBreaks[i];
                        colors[0].endColor = colorsGradient.Evaluate(timeBreaks[i] - SmartArrowUtilities.Utilities.errorRate);
                        colors.Add(new PathColor() { colorMode = PathColor.PathColorMode.Percentage, percentage = 1 - timeBreaks[i] + SmartArrowUtilities.Utilities.errorRate, startColor = colorsGradient.Evaluate(timeBreaks[i] + SmartArrowUtilities.Utilities.errorRate), endColor = colorsGradient.Evaluate(1) });
                    }
                    else
                    {
                        colors[colors.Count - 1].percentage = timeBreaks[i] - timeBreaks[i - 1];
                        colors[colors.Count - 1].endColor = colorsGradient.Evaluate(timeBreaks[i] - SmartArrowUtilities.Utilities.errorRate);
                        colors.Add(new PathColor() { colorMode = PathColor.PathColorMode.Percentage, percentage = 1 - timeBreaks[i] + SmartArrowUtilities.Utilities.errorRate, startColor = colorsGradient.Evaluate(timeBreaks[i] + SmartArrowUtilities.Utilities.errorRate), endColor = colorsGradient.Evaluate(1) });
                    }
                }
            }
        }

    }

}