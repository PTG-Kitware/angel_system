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
        _listContainer = transform.GetComponentInChildren<FollowCameraCanvas>().gameObject;
        //LastPosition = Camera.main.transform.position;
    }
    #endregion
    //Function to have the task overview snap to the center of all required objects
    //This is done when the recipe goes from one step to another. By default, the overview
    //stays on the center of all required objects
    public void SnapToCentroid()
    {
        #region Snapping to center of required items
            /*        Vector3 centroid = new Vector3(0, 0, 0);
                    if (objsDict.Count > 0)
                    {
                        foreach (KeyValuePair<string, GameObject> pair in objsDict)
                        {
                            centroid += pair.Value.transform.position;
                        }
                        centroid = centroid / objsDict.Count;
                        this.GetComponent<MultipleListsContainer>().SetLineEnd(centroid);
                    }*/
            #endregion
        Vector3 centroid = Camera.main.transform.position + Camera.main.transform.forward * zOffset + Camera.main.transform.right * xOffset;
        //UnityEngine.Debug.Log(centroid);
        //UnityEngine.Debug.Log(Camera.main.transform.position.y);
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
        //If user is looking at task overview
        if (isLooking)
        {
            int currIndex = MultiTaskList.Instance.CurrentIndex;
            //If they are looking at the current recipe, show lines pointing to required items
            //CurrDelay = 0.0f;
            if (currIndex == 0)
            {
                foreach (KeyValuePair<string, GameObject> pair in objsDict)
                {
                    UpdateLines(pair.Key, pair.Value);
                }
            }
            else
            {
                DeactivateLines();
            }
            #region old code
            /*            this.GetComponent<MultipleListsContainer>().OverviewLine.gameObject.SetActive(false);
                        int currIndex = this.GetComponent<MultipleListsContainer>().currIndex;
                        if (currIndex == 0)
                        {
                            foreach (KeyValuePair<string, GameObject> pair in objsDict)
                            {
                                UpdateLines(pair.Key, pair.Value);
                            }
                        } else
                        {
                            DeactivateLines();
                        }
                        Vector3 finalPos = Camera.main.transform.position + Camera.main.transform.forward * zOffset + Camera.main.transform.right * xOffset + Camera.main.transform.up * yOffset;
                        this.transform.position = Vector3.Lerp(transform.position, finalPos, Time.deltaTime * movementSpeed);*/
            #endregion
        }
        //If user is not looking at task overview
        else
        {
            DeactivateLines();
            #region Delay code
            /*            CurrDelay += Time.deltaTime;
                        if (CurrDelay >= SnapDelay)
                        {
                            DeactivateLines();
                            float currDistance = Vector3.Distance(Camera.main.transform.position, this.transform.position);
                            if (currDistance > minDistance)
                            {
                                //Once the user moves a certain distance away from the task overview
                                //start lerping the object position to the camera position
                                //with an offset in the z-axis by zOffset and an offset in y-axis by yOffset
                                Vector3 finalPos = Camera.main.transform.position + Camera.main.transform.forward * zOffset + Camera.main.transform.right * xOffset + Camera.main.transform.up * yOffset;
                                BeginLerp(this.transform.position, finalPos);
                            }
                            CurrDelay = 0.0f;
                        }*/
            #endregion
            #region old code
            /*            this.GetComponent<MultipleListsContainer>().OverviewLine.gameObject.SetActive(true);
                        DeactivateLines();
                        Vector3 centroid = new Vector3(0, 0, 0);
                        foreach (KeyValuePair<string, GameObject> pair in objsDict)
                        {
                            centroid += pair.Value.transform.position;
                        }
                        centroid = centroid / objsDict.Count;
                        this.GetComponent<MultipleListsContainer>().SetLineEnd(centroid);
                        Vector3 finalPos = new Vector3(centroid.x, Camera.main.transform.position.y + heightOffset, centroid.z);
                        this.transform.position = Vector3.Lerp(transform.position, finalPos, Time.deltaTime * movementSpeed);*/
            #endregion
        }

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
    /// Remove a specific object (based on key given)
    /// </summary>
    /// <param name="key"></param>
    public void RemoveObj(string key)
    {
        objs.Remove(objsDict[key]);
        objsDict.Remove(key);
        Destroy(linesDict[key]);
        linesDict.Remove(key);
    }
    /// <summary>
    /// Clear all required objects
    /// </summary>
    public void ClearObjs()
    {
        objsDict.Clear();
        objs.Clear();
        foreach (KeyValuePair<string, GameObject> line in linesDict)
        {
            Destroy(line.Value);
        }
        linesDict.Clear();
    }

    ////Add new gameobject as a required task object
    //public void AddObj(string key)
    //{
    //    //TODO: Replace with script for searching for object
    //    GameObject obj = GameObject.Find(key);
    //    if (obj != null)
    //    {
    //        objsDict.Add(key, obj);
    //        objs.Add(obj);
    //        GameObject pointerObj = Instantiate(LinePrefab);
    //        pointerObj.name = key;
    //        linesDict.Add(key, pointerObj);
    //        Line pointer = pointerObj.GetComponent<Line>();
    //        pointer.Start = transform.position;
    //        pointer.End = obj.transform.position;
    //        //SnapToCentroid();
    //        DeactivateLines();
    //    }
    //}

    //Set all lines inactive (once user is not looking at current task
    public void DeactivateLines()
    {
        foreach(KeyValuePair<string, GameObject> pair in linesDict)
        {
            pair.Value.SetActive(false);
        }
    }
    /// <summary>
    /// Update all the lines that point to task objects 
    /// </summary>
    /// <param name="key"></param>
    /// <param name="obj"></param>
    public void UpdateLines(string key, GameObject obj)
    {
        if (key != "MainCam")
        {
            linesDict[key].SetActive(true);
            Line pointer = linesDict[key].GetComponent<Line>();
            //pointer.Start = transform.position;
            pointer.End = obj.transform.position;
        }
    }
    
    /// <summary>
    /// Set the start location of the line pointing at required objects
    /// </summary>
    /// <param name="key"></param>
    /// <param name="StartPos"></param>
    public void SetLineStart(string key, Vector3 StartPos)
    {
        //Check if key exists in dictionary first!!
        if (linesDict.ContainsKey(key))
        {
            Line pointer = linesDict[key].GetComponent<Line>();
            pointer.Start = StartPos;
        } 
        //else
        //{
        //    AddObj(key);
        //    if(linesDict.ContainsKey(key))
        //    {
        //        Line pointer = linesDict[key].GetComponent<Line>();
        //        pointer.Start = StartPos;
        //    }
        //}
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