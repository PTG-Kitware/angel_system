using Cogobyte.SmartProceduralIndicators;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.Demo.SmartProceduralIndicators
{
    public class MultipleMaterialsDemo : MonoBehaviour
    {
        public SmartArrow borderArrow;
        public float borderSize = 0.2f;

        public List<Material> materials;
        public int currentMaterial;
        public SliderHandle borderSlider;
        public MeshRenderer meshRenderer;

        public void Start()
        {
            meshRenderer.material = materials[currentMaterial];
        }

        public void NextMaterial()
        {
            currentMaterial = (currentMaterial + 1) % materials.Count;
            meshRenderer.material = materials[currentMaterial];
        }

        // Update is called once per frame
        void Update()
        {
            borderSize = borderSlider.GetSliderValue() + 0.01f;
            borderArrow.bodyRenderers[0].length = borderSize;
            borderArrow.bodyRenderers[2].length = borderSize;
            Outline o = ((OutlineBodyRenderer)borderArrow.bodyRenderers[1].bodyRenderer).outline;
            o.edges[0].points[1].position.x = -1 + borderSize;
            o.edges[1].points[0].position.x = -1 + borderSize;
            o.edges[1].points[1].position.x = 1 - borderSize;
            o.edges[2].points[0].position.x = 1 - borderSize;
            o.edges[3].points[1].position.y = 1 - borderSize;
            o.edges[4].points[0].position.y = 1 - borderSize;
            o.edges[4].points[1].position.y = -1 + borderSize;
            o.edges[5].points[0].position.y = -1 + borderSize;

            o.edges[6].points[1].position.x = 1 - borderSize;
            o.edges[7].points[0].position.x = 1 - borderSize;
            o.edges[7].points[1].position.x = -1 + borderSize;
            o.edges[8].points[0].position.x = -1 + borderSize;

            o.edges[9].points[1].position.y = -1 + borderSize;
            o.edges[10].points[0].position.y = -1 + borderSize;
            o.edges[10].points[1].position.y = 1 - borderSize;
            o.edges[11].points[0].position.y = 1 - borderSize;
            borderArrow.UpdateArrow();
        }
    }
}