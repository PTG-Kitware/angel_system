using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cogobyte.ProceduralLibrary;

namespace Cogobyte.Demo.ProceduralIndicators
{
    //Options of a selectable object
    public class SelectedObjectOptions : MonoBehaviour
    {
        //Outline of the object
        public PathArray myPath;
        //Copy of the outline
        public PathArray destinationPathArray;
        //is the object selected or not
        public bool selected = false;

        void Start()
        {
            //copy the outline so changes can be made
            destinationPathArray = Instantiate(myPath);
        }

        void Update()
        {
            //put the outline center to the position of the object
            myPath.translation = transform.position;
        }
    }
}