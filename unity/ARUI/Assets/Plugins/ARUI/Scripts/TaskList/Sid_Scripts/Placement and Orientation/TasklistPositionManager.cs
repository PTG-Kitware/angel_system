using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Shapes;
using System.Diagnostics;

public class TasklistPositionManager : Singleton<TasklistPositionManager>
{
    // Debug only
    public List<GameObject> objs;

    public GameObject LinePrefab;
    private GameObject _listContainer;

    public float movementSpeed = 1.0f;
    //Offset from camera if no objects exist
    public float xOffset;
    public float zOffset;

    #region Delay Values
    //public float xOffset;
    //public float yOffset;
    //public float zOffset;
    //public float SnapDelay = 2.0f;
    //public float minDistance = 0.5f;
    //Vector3 LastPosition;
    //float CurrDelay;
    #endregion
    public float heightOffset;

    Dictionary<string, GameObject> objsDict = new Dictionary<string, GameObject>();
    Dictionary<string, GameObject> linesDict = new Dictionary<string, GameObject>();
    float LerpTime = 2.0f;

    Vector3 lerpStart;
    Vector3 lerpEnd;
    bool isLerping;

    float timeStarted;

    bool isLooking = false;
    #region Delay Code
    // Start is called before the first frame update
    void Start()
    {
        _listContainer = transform.GetChild(1).gameObject;
        //LastPosition = Camera.main.transform.position;
    }
    #endregion
    //Function to have the task overview snap to the center of all required objects
    //This is done when the recipe goes from one step to another. By default, the overview
    //stays on the center of all required objects
    public void SnapToCentroid()
    {
        Vector3 centroid = Camera.main.transform.position + Camera.main.transform.forward * zOffset + Camera.main.transform.right * xOffset;
        Vector3 finalPos = new Vector3(centroid.x, Camera.main.transform.position.y + heightOffset, centroid.z);
        BeginLerp(this.transform.position, finalPos);
    }

    // Update is called once per frame
    void Update()
    {
        if (MultiTaskList.Instance == null)
            return;

        MultiTaskList.Instance.SetLineStart(_listContainer.transform.position);
        Vector3 centroid = new Vector3(0, 0, 0);
        foreach (KeyValuePair<string, GameObject> pair in objsDict)
        {
            centroid += pair.Value.transform.position;
        }
        centroid = centroid / objsDict.Count;
        MultiTaskList.Instance.SetLineEnd(centroid);
    }

    //Source -> https://www.blueraja.com/blog/404/how-to-use-unity-3ds-linear-interpolation-vector3-lerp-correctly
    //Code to lerp from one position to another
    void BeginLerp(Vector3 startPos, Vector3 endPos)
    {
        timeStarted = Time.time;
        isLerping = true;
        lerpStart = startPos;
        lerpEnd = endPos;
    }

    void FixedUpdate()
    {
        if (isLerping)
        {
            float timeSinceStarted = Time.time - timeStarted;
            float percentComplete = timeSinceStarted / LerpTime;
            transform.position = Vector3.Lerp(lerpStart, lerpEnd, percentComplete);
            if (percentComplete >= 1.0f)
            {
                isLerping = false;
            }
        }
    }

    /// <summary>
    /// If the user is looking at a task overview object then set isLooking to true
    /// </summary>
    /// <param name="val"></param>
    public void SetIsLooking(bool val)
    {
        isLooking = val;
    }



}