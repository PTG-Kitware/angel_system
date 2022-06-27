using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    [System.Serializable]
    public class MultiBodyItem
    {
        public BodyRenderer bodyRenderer;
        public enum LengthMode { Fixed, Percentage, Fill };
        public LengthMode lengthMode = LengthMode.Fill;
        [Min(0.1f)]
        public float length = 0;
        [Range(0.001F, 1.0F)]
        public float percentage = 1;
        public float displacement = 0;
        public float colorDisplacement = 0;
    }
}