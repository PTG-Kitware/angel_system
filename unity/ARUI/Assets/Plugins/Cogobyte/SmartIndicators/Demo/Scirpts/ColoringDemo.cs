using Cogobyte.SmartProceduralIndicators;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.Demo.SmartProceduralIndicators
{

    public class ColoringDemo : MonoBehaviour
    {
        public SliderHandle sliderR;
        public SliderHandle sliderG;
        public SliderHandle sliderB;
        public SliderHandle sliderLength;

        public SmartArrow outlineArrow;
        public SmartArrow edgeArrow;
        public SmartArrow pointArrow;
        public SmartArrow dashArrow;
        public SmartArrow bodyColorArrow;

        public bool displaceColors = true;
        public bool displaceDash = true;

        public void SwitchDashDisplacemnt()
        {
            displaceDash = !displaceDash;
        }

        public void SwitchColorDisplacemnt()
        {
            displaceColors = !displaceColors;
        }

        // Start is called before the first frame update
        void Start()
        {

        }

        // Update is called once per frame
        void Update()
        {
            Color col = new Color(sliderR.GetSliderValue(), sliderG.GetSliderValue(), sliderB.GetSliderValue());
            ((OutlineBodyRenderer)outlineArrow.bodyRenderers[0].bodyRenderer).outline.color = col;
            ((OutlineBodyRenderer)edgeArrow.bodyRenderers[0].bodyRenderer).outline.edges[0].color = col;
            ((OutlineBodyRenderer)pointArrow.bodyRenderers[0].bodyRenderer).outline.edges[0].points[0].color = col;
            ((OutlineBodyRenderer)bodyColorArrow.bodyRenderers[0].bodyRenderer).colors[1].length = Mathf.Lerp(0.1f, 5, sliderLength.GetSliderValue());
            if (displaceColors)
            {
                dashArrow.bodyRenderers[0].colorDisplacement += Time.deltaTime * 2;
                bodyColorArrow.bodyRenderers[0].colorDisplacement += Time.deltaTime * 2;
            }
            if (displaceDash)
            {
                dashArrow.bodyRenderers[0].displacement += Time.deltaTime * 2;
            }
            dashArrow.UpdateArrow();
            bodyColorArrow.UpdateArrow();
            edgeArrow.UpdateArrow();
            pointArrow.UpdateArrow();
            outlineArrow.UpdateArrow();
        }
    }
}