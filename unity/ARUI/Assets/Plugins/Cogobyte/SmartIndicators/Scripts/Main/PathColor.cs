using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    /// <summary>
    /// Gradient segment used to define a color along a path with customizable length from start color to end color. 
    /// </summary>
    [System.Serializable]
    public class PathColor
    {
        /// <summary>
        /// Percentage mode calcuates the length from percentage of the path. Fixed uses a fixed length value.
        /// </summary>
        public enum PathColorMode{Percentage, Fixed}
        /// <summary>
        /// Current PathColorMode of the path color.
        /// </summary>
        public PathColorMode colorMode = PathColorMode.Percentage;
        /// <summary>
        /// Percentage of the body path that is converted to length property if colorMode is set to Percentage.
        /// </summary>
        [Range(0.0F, 1.0F)]
        public float percentage = 1;
        [Min(0.01f)]
        public float length = 1;
        /// <summary>
        /// Gradient start color at the beggining of the path.
        /// </summary>
        public Color32 startColor = Color.white;
        /// <summary>
        /// Gradient end color at the end of the path.
        /// </summary>
        public Color32 endColor = Color.white;
        public PathColor() {
        }
        public PathColor(PathColor p)
        {
            colorMode = p.colorMode;
            percentage = p.percentage;
            length = p.length;
            startColor = p.startColor;
            endColor = p.endColor;
        }
    }
}