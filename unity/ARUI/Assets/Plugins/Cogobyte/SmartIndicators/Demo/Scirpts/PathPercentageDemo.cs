using Cogobyte.SmartProceduralIndicators;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.Demo.SmartProceduralIndicators
{

    public class PathPercentageDemo : MonoBehaviour
    {
        public SliderHandle startPercentage;
        public SliderHandle bodyPercentage;

        public List<SmartArrow> arrows = new List<SmartArrow>();
        // Start is called before the first frame update
        void Start()
        {

        }

        // Update is called once per frame
        void Update()
        {
            foreach (SmartArrow s in arrows)
            {
                s.StartPercentage = startPercentage.GetSliderValue();
                s.EndPercentage = startPercentage.GetSliderValue() + bodyPercentage.GetSliderValue();
                if (s.EndPercentage > 1) s.EndPercentage = 1;
                s.UpdateArrow();
            }
        }
    }
}