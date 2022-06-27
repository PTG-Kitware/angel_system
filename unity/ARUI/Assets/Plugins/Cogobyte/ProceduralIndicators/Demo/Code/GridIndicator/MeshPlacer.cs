using Cogobyte.ProceduralIndicators;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.Demo.ProceduralIndicators
{
    public class MeshPlacer : MonoBehaviour
    {
        public GameObject meshGridIndicator;
        public GameObject GridIndicator;
        // Use this for initialization
        void Start()
        {

        }

        // Update is called once per frame
        void Update()
        {
            RaycastHit hit;
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            int layer_mask = LayerMask.GetMask("Floor");
            //Left mouse to select objects
            if (Physics.Raycast(ray, out hit, 1000.0f, layer_mask))
            {
                meshGridIndicator.transform.position = hit.point + new Vector3(0, 1, 0);
                GridIndicator.transform.position = hit.point + new Vector3(0, 1, 0);
            }
            MeshGridIndicator meshGrid = meshGridIndicator.GetComponent<MeshGridIndicator>();
            float distanceX = 0;
            float distanceY = 0;
            float distanceZ = 0;
            meshGrid.ResetColor();
            Vector3 pos = meshGridIndicator.transform.position + meshGrid.gridOffset;
            for (int i = 0; i < meshGrid.gridX.Length; i++)
            {
                for (int j = 0; j < meshGrid.gridY.Length; j++)
                {
                    for (int k = 0; k < meshGrid.gridZ.Length; k++)
                    {
                        if (j == 0)
                        {
                            Collider[] hitWater = Physics.OverlapBox(pos + new Vector3(distanceX + meshGrid.gridX[i] / 2, 0, distanceZ + meshGrid.gridZ[k] / 2), new Vector3(meshGrid.gridX[i] / 2, 100, meshGrid.gridZ[k] / 2), Quaternion.identity, layer_mask);
                            if (hitWater.Length > 0)
                            {
                                int t = 0;
                                //Check when there is a new collider coming into contact with the box
                                while (t < hitWater.Length)
                                {
                                    if (hitWater[t].name.Equals("Water"))
                                    {
                                        for (int p = 0; p < meshGrid.gridY.Length; p++)
                                        {
                                            meshGrid.SetColor(i, p, k, Color.blue);
                                        }

                                    }
                                    t++;
                                }

                            }
                        }
                        Collider[] hitColliders = Physics.OverlapBox(pos + new Vector3(distanceX + meshGrid.gridX[i] / 2, distanceY + meshGrid.gridY[j] / 2, distanceZ + meshGrid.gridZ[k] / 2), new Vector3(meshGrid.gridX[i] / 2, meshGrid.gridY[j] / 2, meshGrid.gridZ[k] / 2), Quaternion.identity);
                        if (hitColliders.Length > 0)
                        {
                            meshGrid.SetColor(i, j, k, Color.red);
                        }
                        distanceZ += meshGrid.gridZ[k];
                    }
                    distanceY += meshGrid.gridY[j];
                    distanceZ = 0;
                }
                distanceX += meshGrid.gridX[i];
                distanceY = 0;
                distanceZ = 0;
            }



        }
    }
}