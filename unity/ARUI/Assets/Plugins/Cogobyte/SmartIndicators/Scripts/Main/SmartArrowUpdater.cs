using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    public class SmartArrowUpdater : MonoBehaviour
    {
        public SmartArrow smartArrow;
        // Start is called before the first frame update
        void Start()
        {

        }

        // Update is called once per frame
        void Update()
        {
            smartArrow.UpdateArrow();
        }
    }
}