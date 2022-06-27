using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cogobyte.SmartProceduralIndicators;

namespace Cogobyte.Demo.SmartProceduralIndicators
{

    public class DashDemo : MonoBehaviour
    {
        public Transform dashHandle;
        public Transform emptySpaceHandle;
        public SmartArrow dash;
        public SmartArrow dashAfterEmptySpace;
        public SmartArrow squareDashedArrow;
        public SmartArrow circleDashedArrow;
        public SmartArrow dashLengthEnd;
        public SmartArrow emptyLengthStart;
        public SmartArrow emptyLengthEnd;
        private float dashSpeed = 1;
        public TextMesh speedText;
        // Start is called before the first frame update
        void Start()
        {

        }

        public void IncreaseSpeed()
        {
            dashSpeed++;
            if (dashSpeed > 10) dashSpeed = 10;
            speedText.text = dashSpeed.ToString();
        }

        public void DecreaseSpeed()
        {
            dashSpeed--;
            if (dashSpeed < -10) dashSpeed = -10;
            speedText.text = dashSpeed.ToString();
        }

        // Update is called once per frame
        void Update()
        {
            if (dashHandle.position.z < -10f) dashHandle.position = new Vector3(dashHandle.position.x, dashHandle.position.y, -9.8f);
            if (emptySpaceHandle.position.z < dashHandle.position.z + 0.2f) emptySpaceHandle.position = new Vector3(emptySpaceHandle.position.x, emptySpaceHandle.position.y, dashHandle.position.z + 0.2f);
            if (dashHandle.position.z > 19) dashHandle.position = new Vector3(dashHandle.position.x, dashHandle.position.y, 18.8f);
            if (emptySpaceHandle.position.z > 19) emptySpaceHandle.position = new Vector3(emptySpaceHandle.position.x, emptySpaceHandle.position.y, 19f);


            ((PointToPointArrowPath)dash.arrowPath).pointB.z = dashHandle.position.z + 10;
            ((PointToPointArrowPath)dashAfterEmptySpace.arrowPath).pointA.z = emptySpaceHandle.position.z + 10;

            ((PointListArrowPath)dashLengthEnd.arrowPath).customPath[1] = new Vector3(7, 0, dashHandle.position.z);
            ((PointListArrowPath)dashLengthEnd.arrowPath).customPath[2] = new Vector3(6, 0, dashHandle.position.z);
            ((PointListArrowPath)emptyLengthStart.arrowPath).customPath[0] = new Vector3(3.53f, 0, dashHandle.position.z);
            ((PointListArrowPath)emptyLengthStart.arrowPath).customPath[1] = new Vector3(4, 0, dashHandle.position.z);
            ((PointListArrowPath)emptyLengthEnd.arrowPath).customPath[3] = new Vector3(3.5f, 0, emptySpaceHandle.position.z);
            ((PointListArrowPath)emptyLengthEnd.arrowPath).customPath[4] = new Vector3(4, 0, emptySpaceHandle.position.z);

            dashAfterEmptySpace.UpdateArrow();
            dash.UpdateArrow();
            dashLengthEnd.UpdateArrow();

            emptyLengthStart.UpdateArrow();
            emptyLengthEnd.UpdateArrow();


            ((OutlineBodyRenderer)squareDashedArrow.bodyRenderers[0].bodyRenderer).dashLength = dashHandle.position.z + 10;
            ((OutlineBodyRenderer)squareDashedArrow.bodyRenderers[0].bodyRenderer).emptyLength = emptySpaceHandle.position.z - dashHandle.position.z;
            squareDashedArrow.bodyRenderers[0].displacement += dashSpeed * Time.deltaTime;
            squareDashedArrow.UpdateArrow();
            ((OutlineBodyRenderer)circleDashedArrow.bodyRenderers[0].bodyRenderer).dashLength = dashHandle.position.z + 10;
            ((OutlineBodyRenderer)circleDashedArrow.bodyRenderers[0].bodyRenderer).emptyLength = emptySpaceHandle.position.z - dashHandle.position.z;
            circleDashedArrow.bodyRenderers[0].displacement += dashSpeed * Time.deltaTime;
            circleDashedArrow.UpdateArrow();
        }
    }
}