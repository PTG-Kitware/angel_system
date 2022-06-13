using Cogobyte.SmartProceduralIndicators;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.Demo.SmartProceduralIndicators
{

    public class PrefabsDemo : MonoBehaviour
    {
        public List<SmartArrow> arrows;
        public int currentArrow = 0;
        public int lod = 1000;
        public float amplitude = 0.2f;
        public float freq = 2f;
        public float length = 40;
        public float dist = 0;
        public float speed = 2f;
        // Start is called before the first frame update
        void Start()
        {
            arrows[currentArrow].gameObject.SetActive(true);
        }

        public void NextArrow()
        {
            arrows[currentArrow].gameObject.SetActive(false);
            currentArrow = (currentArrow + 1) % arrows.Count;
            arrows[currentArrow].gameObject.SetActive(true);
        }

        // Update is called once per frame
        void Update()
        {
            dist += Time.deltaTime * speed;
            PointListArrowPath p = (PointListArrowPath)arrows[currentArrow].arrowPath;
            p.customPath.Clear();
            for (int i = 0; i <= lod; i++)
            {
                p.customPath.Add(new Vector3(amplitude * Mathf.Sin((((float)i) / lod * length + dist) * freq), 0, ((float)i) / lod) * length);
            }
            arrows[currentArrow].UpdateArrow();
        }
    }
}