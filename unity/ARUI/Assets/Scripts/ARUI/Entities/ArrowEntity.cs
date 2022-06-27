using UnityEngine;
using Cogobyte.ProceduralIndicators;
using System.Collections.Generic;
using System;

public class ArrowEntity : Entity
{
    private ArrowObject arrow;
    private ArrowPath arrowPath;
    private ArrowTip arrowHead;
    private ArrowTip arrowTail;
    private MeshRenderer meshRenderer;
    public float bodyThickness;
    //Animation variables
    private bool isAnimated;
    private int startIndex;
    private int endIndex;
    private float timer;
    private float pauseTime;
    private int incrSpeed;
    private List<Vector3> originalPath;

    // Start is called before the first frame update
    void Awake()
    {
        transform.parent = transform;
        isAnimated = false;

        // Set default values
        arrow = gameObject.AddComponent<ArrowObject>();
        meshRenderer = gameObject.GetComponent<MeshRenderer>();
        meshRenderer.material = Resources.Load("Materials/DefaultIndicatorsMaterial", typeof(Material)) as Material;
        bodyThickness = 0.005f;

        //Entity Information intitialization
        entityType = Type.Arrow;
        id = gameObject.name;

        // Arrow Path initialization
        arrow.arrowPath = ScriptableObject.CreateInstance<ArrowPath>();
        arrowPath = arrow.arrowPath;
        arrowPath.arrowPathMode = ArrowPath.ArrowPathMode.Extrude;
        arrowPath.widthFunction = AnimationCurve.Constant(0, 1, bodyThickness);
        arrowPath.heightFunction = AnimationCurve.Constant(0, 1, bodyThickness);

        // Arrow Head initialization
        arrow.arrowHead = ScriptableObject.CreateInstance<ArrowTip>();
        arrowHead = arrow.arrowHead;
        arrowPath.arrowHead = arrow.arrowHead;

        // Arrow Tail initialization
        arrow.arrowTail = ScriptableObject.CreateInstance<ArrowTip>();
        arrowTail = arrow.arrowTail;
        arrowPath.arrowTail = arrow.arrowTail;
    }

    public void SetAsPointerArrow(Vector3 center)
    {
        // Arrow Path
        arrowPath.arrowPathType = ArrowPath.ArrowPathType.Function;
        arrowPath.startPoint = center + new Vector3(0, 0.2f, 0);
        arrowPath.endPoint = center + new Vector3(0, 0.1f, 0);
        arrowPath.levelOfDetailAlongPath = 10;
        arrowPath.pathAlongXFunction = AnimationCurve.Constant(0, 1, 0);
        arrowPath.pathAlongYFunction = AnimationCurve.Constant(0, 1, 0);
        arrowPath.pathAlongZFunction = AnimationCurve.Constant(0, 1, 0);
        // Arrow Head
        arrowHead.arrowTipMode = ArrowTip.ArrowTipMode.Extrude;
        float tipSlope = 0.5f;
        arrowHead.widthFunction = AnimationCurve.Linear(0, tipSlope, 1, 0);
        arrowHead.heightFunction = AnimationCurve.Linear(0, tipSlope, 1, 0);
        arrowHead.size = 5 * new Vector3(bodyThickness, bodyThickness, bodyThickness);

        arrow.Init();
    }
    public void SetAsCircularArrow(Vector3 center, Vector3 normal, float angle, float radius)
    {
        transform.position = center;
        transform.up = normal;
        // Arrow Path
        arrowPath.arrowPathType = ArrowPath.ArrowPathType.PointArray;
        float stepSize = 0.01f;
        int firstPoint = 0;
        int lastPoint = (int)(angle * Mathf.Deg2Rad / stepSize);
        arrowPath.editedPath = new List<Vector3>();
        float step = 0;
        for (int i = firstPoint; i < lastPoint; i++)
        {
            arrowPath.editedPath.Add(radius * new Vector3(Mathf.Sin(step), 0, Mathf.Cos(step)));
            step += stepSize;
        }
        // Arrow Head
        arrowHead.arrowTipMode = ArrowTip.ArrowTipMode.Extrude;
        float tipSlope = 0.5f;
        arrowHead.widthFunction = AnimationCurve.Linear(0, tipSlope, 1, 0);
        arrowHead.heightFunction = AnimationCurve.Linear(0, tipSlope, 1, 0);
        arrowHead.size = 0.3f * new Vector3(radius, radius, radius);

        arrow.Init();
    }

    public void SetAsPathArrow(Vector3 start, Vector3 end, float curvatureKey, bool usePointArray)
    {
        // Arrow Path
        arrowPath.startPoint = start;
        arrowPath.endPoint = end;
        arrowPath.levelOfDetailAlongPath = 50;

        if (usePointArray)
        {
            arrowPath.arrowPathType = ArrowPath.ArrowPathType.PointArray;
            arrowPath.editedPath = new List<Vector3>();
            float dist = Vector2.Distance(new Vector2(start.x, start.z), new Vector2(end.x, end.z));
            int totalSteps = 50;
            float divider = 1f / totalSteps;
            float step = 0f;
            for (int i = 0; i < totalSteps; i++)
            {
                step = i == 0 ? divider / 2 : step + divider;
                arrowPath.editedPath.Add(Vector3.Lerp(start, end, step));
            }
            divider = dist / totalSteps;
            step = 0f;
            float constant = Mathf.Pow(3 * 0.5f * dist, 2);
            for (int i = 0; i < totalSteps; i++)
            {
                step += divider;
                arrowPath.editedPath[i] += new Vector3(0, -Mathf.Pow(3 * (step - 0.5f * dist), 2) + constant, 0);
            }
        }
        else
        {
            arrowPath.arrowPathType = ArrowPath.ArrowPathType.Function;
            arrowPath.pathAlongXFunction = AnimationCurve.Constant(0, 1, 0);
            AnimationCurve myCurve = new AnimationCurve();
            myCurve.AddKey(0, 0); // arrowHead
            myCurve.AddKey(0.5f, curvatureKey); // curvature key
            myCurve.AddKey(1, 0);  // arrowTail
            arrowPath.pathAlongYFunction = myCurve;
            arrowPath.pathAlongZFunction = AnimationCurve.Constant(0, 1, 0);
        }

        // Arrow Head
        arrowHead.arrowTipMode = ArrowTip.ArrowTipMode.Extrude;
        float tipSlope = 0.5f;
        arrowHead.widthFunction = AnimationCurve.Linear(0, tipSlope, 1, 0);
        arrowHead.heightFunction = AnimationCurve.Linear(0, tipSlope, 1, 0);
        arrowHead.size = 5 * new Vector3(bodyThickness, bodyThickness, bodyThickness);

        arrow.Init();
    }

    public void SetColor(Color colorName) => meshRenderer.material.SetColor("_Color", colorName);

    // Pause in seconds, speed refers to the number of increments per step
    public void SetAnimation(float pause, int speed = 1)
    {
        if (arrowPath.arrowPathType != ArrowPath.ArrowPathType.PointArray)
        {
            string errMsg = string.Format("{0} does not have point array, therefore cannot be animated", name);
            throw new InvalidOperationException(errMsg);
        }
        originalPath = arrowPath.editedPath;
        startIndex = 0;
        endIndex = 1;
        timer = 0.0f;
        incrSpeed = speed;
        pauseTime = pause;
        isAnimated = true;
    }

    public void animationUpdate()
    {
        if (timer < pauseTime)
        {
            timer += Time.deltaTime;
            return;
        }
        arrowPath.editedPath = new List<Vector3>();
        for (int i = startIndex; i < endIndex; i++)
        {
            arrowPath.editedPath.Add(originalPath[i]);
        }
        Logger.print(arrowPath.editedPath.Count);
        endIndex += incrSpeed;
        if (endIndex >= originalPath.Count)
        {
            endIndex = 1;
            timer = 0.0f;
        }
    }

    // Need to check if update arrow mesh has performance impact
    void Update()
    {
        if (isAnimated) animationUpdate();
        arrow.updateArrowMesh();
    }
}
