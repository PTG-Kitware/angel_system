using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.Demo.ProceduralIndicators
{
    //Switches active arrow examples and texts in FeaturesShowcaseDemo
    public class ShuffleDemos : MonoBehaviour
    {
        public GameObject[] demos;
        public GameObject[] texts;
        int counter = 0;

        public void nextDemo()
        {
            demos[counter].SetActive(false);
            texts[counter].SetActive(false);
            if (counter != demos.Length - 1) { counter++; }
            else { counter = 0; }
            demos[counter].SetActive(true);
            texts[counter].SetActive(true);
        }
    }
}