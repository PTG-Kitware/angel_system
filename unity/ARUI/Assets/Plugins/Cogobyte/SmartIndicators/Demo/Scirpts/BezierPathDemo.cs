using Cogobyte.SmartProceduralIndicators;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.Demo.SmartProceduralIndicators
{

    public class BezierPathDemo : MonoBehaviour
    {
        public Color32 controlPointLineColor = Color.green;
        public SmartArrow bezierArrow;
        public List<Transform> pointHandle = new List<Transform>();
        public List<Transform> inTangentHandle = new List<Transform>();
        public List<Transform> outTangentHandle = new List<Transform>();
        public GameObject positionHandlePrefab;
        public GameObject simpleLinePrefab;
        public List<SmartArrow> inTangentLine = new List<SmartArrow>();
        public List<SmartArrow> outTangentLine = new List<SmartArrow>();


        public void AddPoint()
        {
            GameObject g = Instantiate(positionHandlePrefab, transform);
            SmartArrow[] smartArrows = g.GetComponentsInChildren<SmartArrow>();
            foreach (SmartArrow s in smartArrows)
            {
                s.UpdateArrow();
            }
            pointHandle.Add(g.transform);
            inTangentLine.Add(Instantiate(simpleLinePrefab, transform).GetComponent<SmartArrow>());
            g = Instantiate(positionHandlePrefab, transform);
            smartArrows = g.GetComponentsInChildren<SmartArrow>();
            foreach (SmartArrow s in smartArrows)
            {
                s.UpdateArrow();
            }
            inTangentHandle.Add(g.transform);

            outTangentLine.Add(Instantiate(simpleLinePrefab, transform).GetComponent<SmartArrow>());
            g = Instantiate(positionHandlePrefab, transform);
            smartArrows = g.GetComponentsInChildren<SmartArrow>();
            foreach (SmartArrow s in smartArrows)
            {
                s.UpdateArrow();
            }
            outTangentHandle.Add(g.transform);
            outTangentLine.Add(Instantiate(simpleLinePrefab, transform).GetComponent<SmartArrow>());

            pointHandle[pointHandle.Count - 1].transform.position = outTangentHandle[outTangentHandle.Count - 2].position;
            inTangentHandle[inTangentHandle.Count - 1].transform.position = pointHandle[pointHandle.Count - 1].transform.position + (pointHandle[pointHandle.Count - 2].transform.position - pointHandle[pointHandle.Count - 1].transform.position).normalized * 3;
            outTangentHandle[outTangentHandle.Count - 1].transform.position = pointHandle[pointHandle.Count - 1].transform.position + (pointHandle[pointHandle.Count - 1].transform.position - pointHandle[pointHandle.Count - 2].transform.position).normalized * 3;
            ((BezierArrowPath)bezierArrow.arrowPath).bezierSpline.AddCurve();

        }

        // Start is called before the first frame update
        void Start()
        {

        }

        // Update is called once per frame
        void Update()
        {
            BezierArrowPath path = ((BezierArrowPath)bezierArrow.arrowPath);
            for (int i = 0; i < path.bezierSpline.points.Count; i++)
            {
                pointHandle[i].position = new Vector3(
                                                        Mathf.Clamp(pointHandle[i].position.x, -30, 30),
                                                        Mathf.Clamp(pointHandle[i].position.y, -30, 30),
                                                        Mathf.Clamp(pointHandle[i].position.z, -30, 30));
                path.bezierSpline.points[i].position = pointHandle[i].position;
                path.bezierSpline.points[i].outTangent = outTangentHandle[i].position;
                path.bezierSpline.points[i].inTangent = inTangentHandle[i].position;
                ((PointToPointArrowPath)inTangentLine[i].arrowPath).pointA = pointHandle[i].position;
                ((PointToPointArrowPath)outTangentLine[i].arrowPath).pointA = pointHandle[i].position;
                ((PointToPointArrowPath)inTangentLine[i].arrowPath).pointB = inTangentHandle[i].position;
                ((PointToPointArrowPath)outTangentLine[i].arrowPath).pointB = outTangentHandle[i].position;
                inTangentLine[i].UpdateArrow();
                outTangentLine[i].UpdateArrow();
            }
            bezierArrow.UpdateArrow();
        }
    }
}