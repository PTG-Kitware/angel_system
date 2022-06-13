using Cogobyte.SmartProceduralIndicators;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.Demo.SmartProceduralIndicators
{

    public class BodiesDemo : MonoBehaviour
    {
        public SmartArrow arrow;
        public float bodyDisplacement = 0;
        public float dashDisplacement = 0;
        public float shapeDisplacement = 0;
        public bool displaceBody = true;
        public bool displaceDash = true;
        public bool displaceShape = true;
        public SliderHandle bodyLengthHandle;

        public void SwitchBodyDisplacement()
        {
            displaceBody = !displaceBody;
        }
        public void SwitchDashDisplacement()
        {
            displaceDash = !displaceDash;
        }
        public void SwitchShapeDisplacement()
        {
            displaceShape = !displaceShape;
        }

        // Start is called before the first frame update
        void Start()
        {

        }

        // Update is called once per frame
        void Update()
        {
            if (displaceBody)
            {
                bodyDisplacement += Time.deltaTime * 2;
            }
            if (displaceDash)
            {
                dashDisplacement += Time.deltaTime * 2;
            }
            if (displaceShape)
            {
                shapeDisplacement += Time.deltaTime * 2;
            }
            arrow.bodyRenderers[0].length = Mathf.Lerp(0, 20, bodyLengthHandle.GetSliderValue());
            arrow.bodyRenderers[1].displacement = dashDisplacement;
            arrow.bodyRenderers[2].displacement = shapeDisplacement;
            arrow.displacement = bodyDisplacement;
            arrow.UpdateArrow();
        }
    }
}