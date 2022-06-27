using Cogobyte.SmartProceduralIndicators;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.Demo.SmartProceduralIndicators
{

    public class BodyTwistAndScaleDemo : MonoBehaviour
    {
        public SliderHandle twist;
        public SmartArrow twistArrow;
        public SmartArrow animatedTwistArrow;
        public SmartArrow sizeArrow;
        public SmartArrow animatedSizeArrow;

        public List<SmartArrow> arrows = new List<SmartArrow>();
        public Transform sizeHandle;
        float angle = 0;
        // Start is called before the first frame update
        void Start()
        {

        }

        // Update is called once per frame
        void Update()
        {
            angle += Time.deltaTime * 100;
            if (angle > 360) angle = 0;
            ((OutlineBodyRenderer)animatedTwistArrow.bodyRenderers[0].bodyRenderer).curveFunction[2] = AnimationCurve.Linear(0,angle,10, angle+ Mathf.Lerp(0, 720, twist.GetSliderValue()));
            ((OutlineBodyRenderer)twistArrow.bodyRenderers[0].bodyRenderer).curveFunction[2] = AnimationCurve.Linear(0, 0,10, Mathf.Lerp(0, 720, twist.GetSliderValue()));
            Keyframe[] keyframes = ((OutlineBodyRenderer)sizeArrow.bodyRenderers[0].bodyRenderer).curveFunction[0].keys;
            keyframes[1].time = Mathf.Abs(sizeHandle.position.z);
            keyframes[1].value = Mathf.Abs(sizeHandle.position.x);
            ((OutlineBodyRenderer)sizeArrow.bodyRenderers[0].bodyRenderer).curveFunction[0].keys = keyframes;
            ((OutlineBodyRenderer)animatedSizeArrow.bodyRenderers[0].bodyRenderer).curveFunction[0].keys = keyframes;
            keyframes = ((OutlineBodyRenderer)sizeArrow.bodyRenderers[0].bodyRenderer).curveFunction[1].keys;
            keyframes[1].time = Mathf.Abs(sizeHandle.position.z);
            keyframes[1].value = Mathf.Abs(sizeHandle.position.y);
            ((OutlineBodyRenderer)sizeArrow.bodyRenderers[0].bodyRenderer).curveFunction[1].keys = keyframes;
            ((OutlineBodyRenderer)animatedSizeArrow.bodyRenderers[0].bodyRenderer).curveFunction[1].keys = keyframes;
            animatedSizeArrow.UpdateArrow();
            sizeArrow.UpdateArrow();
            twistArrow.UpdateArrow();
            animatedTwistArrow.UpdateArrow();

        }
    }
}