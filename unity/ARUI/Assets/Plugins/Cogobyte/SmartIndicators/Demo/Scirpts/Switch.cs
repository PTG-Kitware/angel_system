using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

namespace Cogobyte.Demo.SmartProceduralIndicators
{

    public class Switch : MonoBehaviour
    {
        public UnityEvent switchAction;
        void OnMouseDown()
        {
            switchAction.Invoke();
        }
    }

}