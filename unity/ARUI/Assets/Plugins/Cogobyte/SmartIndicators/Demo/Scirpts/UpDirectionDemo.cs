using Cogobyte.SmartProceduralIndicators;
using UnityEngine;

namespace Cogobyte.Demo.SmartProceduralIndicators
{
    public class UpDirectionDemo : MonoBehaviour
    {
        public SmartArrow constantUpArrow;
        public SmartArrow autoUpArrow;
        public SmartArrow spiralArrow;
        public SmartArrow sphereArrow;
        public SliderHandle angleHandle;

        public float lod = 100;
        public float angle = 720;
        public float radius = 3;
        public float height = 10;
        public float minAngle = 100;
        public float maxAngle = 1720;

        // Start is called before the first frame update
        void Start()
        {
        }

        // Update is called once per frame
        void Update()
        {
            angle = Mathf.Lerp(minAngle, maxAngle, angleHandle.GetSliderValue());
            PointListArrowPath p = ((PointListArrowPath)spiralArrow.arrowPath);
            PointListArrowPath p2 = ((PointListArrowPath)sphereArrow.arrowPath);
            PointListArrowPath p3 = ((PointListArrowPath)autoUpArrow.arrowPath);
            PointListArrowPath p4 = ((PointListArrowPath)constantUpArrow.arrowPath);
            p.customPath.Clear();
            p2.customPath.Clear();
            p3.customPath.Clear();
            p4.customPath.Clear();
            for (int i = 0; i < lod; i++)
            {
                Vector3 point = new Vector3(radius * Mathf.Cos(angle * Mathf.Deg2Rad * ((float)i) / lod), Mathf.Lerp(0, height, ((float)i) / lod), radius * Mathf.Sin(angle * Mathf.Deg2Rad * ((float)i) / lod));
                p.customPath.Add(point);
                p2.customPath.Add(point);
                p3.customPath.Add(point);
                p4.customPath.Add(point);
            }
            spiralArrow.UpdateArrow();
            sphereArrow.UpdateArrow();
            autoUpArrow.UpdateArrow();
            constantUpArrow.UpdateArrow();
        }
    }
}