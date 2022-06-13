using Cogobyte.SmartProceduralIndicators;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.Demo.SmartProceduralIndicators
{

    public class TipDemo : MonoBehaviour
    {
        public SmartArrow outlineTipArrow;
        public SmartArrow vertialOutlineTipArrow;
        public SmartArrow meshTipArrow;
        public List<OutlineTip> outlineTips;
        public List<VerticalOutlineTip> verticalOutlineTips;
        public List<MeshTip> meshTips;
        int currentOutlineTip = 0;
        int currentVerticalOutlineTip = 0;
        int currentMeshOutlineTip = 0;

        public void NextOutlineTip()
        {
            currentOutlineTip = (currentOutlineTip + 1) % outlineTips.Count;
            outlineTipArrow.arrowHead = outlineTips[currentOutlineTip];
            outlineTipArrow.UpdateArrow();
        }

        public void NextVerticalOutlineTip()
        {
            currentVerticalOutlineTip = (currentVerticalOutlineTip + 1) % verticalOutlineTips.Count;
            vertialOutlineTipArrow.arrowHead = verticalOutlineTips[currentVerticalOutlineTip];
            vertialOutlineTipArrow.UpdateArrow();
        }

        public void NextMeshTip()
        {
            currentMeshOutlineTip = (currentMeshOutlineTip + 1) % meshTips.Count;
            meshTipArrow.arrowHead = meshTips[currentMeshOutlineTip];
            meshTipArrow.UpdateArrow();
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