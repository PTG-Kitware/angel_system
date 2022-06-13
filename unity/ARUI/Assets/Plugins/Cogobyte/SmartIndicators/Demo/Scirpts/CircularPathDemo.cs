using Cogobyte.SmartProceduralIndicators;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.Demo.SmartProceduralIndicators
{

    public class CircularPathDemo : MonoBehaviour
    {
        public SmartArrow circleArrow;
        public SmartArrow radiusLine;
        public SmartArrow axisLine;
        public SliderHandle startAngle;
        public SliderHandle endAngle;
        public SliderHandle roll;
        public Transform centerHandle;
        public Transform radiusHandle;
        public Transform axisHandle;
        // Start is called before the first frame update
        void Start()
        {

        }

        // Update is called once per frame
        void Update()
        {
            ((CircleArrowPath)circleArrow.arrowPath).startAngle = Mathf.Lerp(0, 360, startAngle.GetSliderValue());
            ((CircleArrowPath)circleArrow.arrowPath).endAngle = Mathf.Lerp(0, 360, endAngle.GetSliderValue());
            ((CircleArrowPath)circleArrow.arrowPath).upDirectionRollAngle = Mathf.Lerp(0, 360, roll.GetSliderValue());
            ((CircleArrowPath)circleArrow.arrowPath).center = centerHandle.position;
            ((CircleArrowPath)circleArrow.arrowPath).radius = radiusHandle.position - centerHandle.position;
            ((CircleArrowPath)circleArrow.arrowPath).axis = axisHandle.position - centerHandle.position;
            ((PointToPointArrowPath)radiusLine.arrowPath).pointA = centerHandle.position;
            ((PointToPointArrowPath)radiusLine.arrowPath).pointB = radiusHandle.position;
            ((PointToPointArrowPath)axisLine.arrowPath).pointA = centerHandle.position;
            ((PointToPointArrowPath)axisLine.arrowPath).pointB = axisHandle.position;
            radiusLine.UpdateArrow();
            axisLine.UpdateArrow();
            circleArrow.UpdateArrow();
        }
    }
}