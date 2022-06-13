using Cogobyte.SmartProceduralIndicators;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.Demo.SmartProceduralIndicators
{

    public class PointListDemo : MonoBehaviour
    {
        public SmartArrow pathListArrow;
        public Transform Handle;
        public float moveFactor = 7;

        public void AddForward()
        {
            Handle.position += moveFactor * Vector3.forward;
            ((PointListArrowPath)pathListArrow.arrowPath).customPath.Add(Handle.position);
            pathListArrow.UpdateArrow();
        }

        public void AddBack()
        {
            Handle.position += moveFactor * Vector3.back;
            ((PointListArrowPath)pathListArrow.arrowPath).customPath.Add(Handle.position);
            pathListArrow.UpdateArrow();
        }
        public void AddRight()
        {
            Handle.position += moveFactor * Vector3.right;
            ((PointListArrowPath)pathListArrow.arrowPath).customPath.Add(Handle.position);
            pathListArrow.UpdateArrow();
        }
        public void AddLeft()
        {
            Handle.position += moveFactor * Vector3.left;
            ((PointListArrowPath)pathListArrow.arrowPath).customPath.Add(Handle.position);
            pathListArrow.UpdateArrow();
        }
        public void AddUp()
        {
            Handle.position += moveFactor * Vector3.up;
            ((PointListArrowPath)pathListArrow.arrowPath).customPath.Add(Handle.position);
            pathListArrow.UpdateArrow();
        }
        public void AddDown()
        {
            Handle.position += moveFactor * Vector3.down;
            ((PointListArrowPath)pathListArrow.arrowPath).customPath.Add(Handle.position);
            pathListArrow.UpdateArrow();
        }

        // Start is called before the first frame update
        void Start()
        {

        }

        // Update is called once per frame
        void Update()
        {

        }
    }
}