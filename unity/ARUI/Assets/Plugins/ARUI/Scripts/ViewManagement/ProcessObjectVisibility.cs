using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Xml;
using IntervalTree;
using UnityEngine;
using Random = UnityEngine.Random;

public class ProcessObjectVisibility : Singleton<ProcessObjectVisibility>
{
    //** Rendered Z-Buffer
    private Texture2D _imageTex;
    private Camera _zBufferCam;

    private int scale = 1; //ratio between rendered image in AR and the zBuffer, set at run-time

    private Dictionary<CVDetectedObj, List<Rect>> AABBsPerNonControllable;

    private List<CVDetectedObj> visibleNonControllables = new List<CVDetectedObj>();
    private Dictionary<Color, CVDetectedObj> colorToNonControllable;
    private Dictionary<int, CVDetectedObj> indexToNonControllable;
    private Dictionary<Color, int> colorIDs;

    private bool processing = false;
    public bool IsProcessing { get { return processing; } }

    private List<Color> assignedColors = new List<Color>();

    public void Start()
    {
        _zBufferCam = Camera.main.transform.GetChild(0).transform.GetComponent<Camera>();
        scale = AngelARUI.Instance.ARCamera.pixelWidth / _zBufferCam.targetTexture.width;
        _imageTex = new Texture2D(_zBufferCam.targetTexture.width, _zBufferCam.targetTexture.height, TextureFormat.ARGB32, false);

        Debug.Log("ZBuffer Initialized " + _zBufferCam.targetTexture.width + "," + _zBufferCam.targetTexture.height);

        StartCoroutine(DoCapture());
    }

    IEnumerator DoCapture()
    {
        Debug.Log("ZBufferCam: " + _zBufferCam.pixelWidth + " " + _zBufferCam.pixelHeight);
        Debug.Log("Main: " + Camera.main.pixelWidth + " " + Camera.main.pixelHeight);

        int samplecount = 30;
        int counter = 0;
        float[] sum = new float[2] { 0, 0 };
        while (true)
        {
            yield return new WaitForEndOfFrame();

            while (visibleNonControllables.Count <= 0 || !AngelARUI.Instance.IsVMActiv)
            {
                yield return new WaitForEndOfFrame();
            }

            int zerocounter = 0;
            foreach(var controllable in visibleNonControllables)
            {
                if (controllable.AABB.width==0 || controllable.AABB.height==0)
                    zerocounter++;
            }

            if (zerocounter == visibleNonControllables.Count)
            {
                AABBsPerNonControllable = new Dictionary<CVDetectedObj, List<Rect>>();
                indexToNonControllable = new Dictionary<int, CVDetectedObj> { };
                colorToNonControllable = new Dictionary<Color, CVDetectedObj>();
                colorIDs = new Dictionary<Color, int>();
                continue;
            }
                
            if (counter == samplecount)
            {
                //AngelARUI.Instance.LogDebugMessage("Visibility Timer: copy " + (sum[0] / samplecount) + "sec, sweep " + (sum[1] / samplecount) + "sec", true);
                sum = new float[2] { 0, 0 };
                counter = 0;
            }
            counter++;

            float copytimer = Time.realtimeSinceStartup;
            //Get ZBuffer picture
            RenderTexture.active = _zBufferCam.targetTexture;
            _imageTex.ReadPixels(new Rect(0, 0, _zBufferCam.targetTexture.width, _zBufferCam.targetTexture.height), 0, 0);
            _imageTex.Apply();
            RenderTexture.active = null;

            copytimer = Time.realtimeSinceStartup - copytimer;

            yield return new WaitForEndOfFrame();

            indexToNonControllable = new Dictionary<int, CVDetectedObj> { };
            colorToNonControllable = new Dictionary<Color, CVDetectedObj>();
            colorIDs = new Dictionary<Color, int>();

            int index = 0;

            foreach (var item in visibleNonControllables)
            {
                if (item.AABB.Equals(Vector3.zero)) continue;

                colorToNonControllable.Add(item.Color, item);
                indexToNonControllable.Add(index, item);
                colorIDs.Add(item.Color, index);
                index++;
            }

            //print for testing
            //Utils.SaveCapture(imageTex,"test");

            processing = true;
            float sweepTimer = Time.realtimeSinceStartup;

            StartCoroutine(ComputeRectsForEachObjectInZBuffer());

            while (processing)
            {
                yield return new WaitForEndOfFrame();
            }

            sweepTimer = Time.realtimeSinceStartup-sweepTimer;
            sum[0] += copytimer;
            sum[1] += sweepTimer;

        }
    }

    IEnumerator ComputeRectsForEachObjectInZBuffer()
    {
        // Declare variables
        Dictionary<int, List<ViewSpaceRectangle>> objectVPRep = new Dictionary<int, List<ViewSpaceRectangle>>();
        Dictionary<int, IntervalTree<int, ViewSpaceRectangle>> objectXIT = new Dictionary<int, IntervalTree<int, ViewSpaceRectangle>>(); //objectXIT is an array of 1D interval trees
        foreach (var item in indexToNonControllable.Keys)
        {
            objectVPRep.Add(item, new List<ViewSpaceRectangle>());
            objectXIT.Add(item,new IntervalTree<int, ViewSpaceRectangle>());
        }

        int startX = -1;
        int previousObjectValue= -1;
        int currentObjectValue = -1;

        // Loop through the pixels in the buffer
        for (int y = 0; y < _imageTex.height; y+= ARUISettings.SMPixelSkip)
        {
            previousObjectValue = -1;
            int x = 0;
            for (x = 0; x < (_imageTex.width); x += ARUISettings.SMPixelSkip)
            {
                currentObjectValue = GetObjectID(x, y);

                if (previousObjectValue == -1)  // No object in previous pixel 
                {
                    startX = x;
                    previousObjectValue = currentObjectValue;
                }
                else if (previousObjectValue != currentObjectValue)// Change detected, combine region to all adjacent spaces for object
                {
                    ProcessRectangle(objectVPRep[previousObjectValue], objectXIT[previousObjectValue], startX, x - 1, y);
                    startX = x;
                    previousObjectValue = currentObjectValue;
                    //yield return new WaitForEndOfFrame();
                }
               
            }

            // For the last pixel in each row
            if (previousObjectValue != -1 && startX != x)
            {
                ProcessRectangle(objectVPRep[previousObjectValue], objectXIT[previousObjectValue], startX, x - 1, y);
                //yield return new WaitForEndOfFrame();
            }

        }

        AABBsPerNonControllable = new Dictionary<CVDetectedObj, List<Rect>>();

        // Add all resulting spaces in interval tree to objectVPRep array for each object
        for (int j = 0; j < indexToNonControllable.Keys.Count; j++)
        {
            CVDetectedObj currentNonVM = indexToNonControllable[j];
            foreach (var rect in objectXIT[j].Values)
                objectVPRep[j].Add(rect);

            AABBsPerNonControllable.Add(currentNonVM, new List<Rect>());
            
            foreach (var item in objectVPRep[j])
            {
                ViewSpaceRectangle scaled = new ViewSpaceRectangle(item.StartX * scale, item.StartY * scale, item.EndX * scale, item.EndY * scale);
                int h = scaled.EndY - scaled.StartY;
                int w = scaled.EndX - scaled.StartX;

                AABBsPerNonControllable[currentNonVM].Add(new Rect(scaled.StartX, scaled.StartY, w ,h ));
            }

        }

        processing = false;

        yield return new WaitForEndOfFrame();
    }

    private void ProcessRectangle(List<ViewSpaceRectangle> objectVP, IntervalTree<int, ViewSpaceRectangle> objectXIT, int startX, int lastX, int placeY)
    {
        // Declare variables
        ViewSpaceRectangle newRect = new ViewSpaceRectangle(startX, placeY, lastX, placeY);
        List<ViewSpaceRectangle> allNewPotentialRects = new List<ViewSpaceRectangle>();
        allNewPotentialRects.Add(newRect);

        //neighbors in previous row
        IEnumerable<ViewSpaceRectangle> allOverlappingInXRects = objectXIT.Query(startX, lastX);

        // Loop through all overlapping rectangles in the X axis
        foreach (ViewSpaceRectangle R in allOverlappingInXRects)
        {
            if (newRect.IsAdjacentinY(R, ARUISettings.SMPixelSkip))
            {
                // Rectangles are adjacent in the Y axis
                ViewSpaceRectangle combinedRect = newRect.ConsensusInY(R);
                if (R.IsEnclosedBy(combinedRect))
                    objectXIT.Remove(R);

                List<ViewSpaceRectangle> temp = new List<ViewSpaceRectangle>(allNewPotentialRects);
                // Remove any rectangles in allNewPotentialRects that are enclosed by combinedRect
                foreach (var rect in allNewPotentialRects)
                {
                    if (rect.IsEnclosedBy(combinedRect))
                        temp.Remove(rect);
                }
                allNewPotentialRects = temp;

                // Add combinedRect to allNewPotentialRects if it is not enclosed by any rectangles in allNewPotentialRects
                if (!allNewPotentialRects.Exists(rect => combinedRect.IsEnclosedBy(rect)))
                    allNewPotentialRects.Add(combinedRect);

                if (R.EndX < lastX)
                {
                    objectVP.Add(R);
                    objectXIT.Remove(R);
                }
            }
            else
            {
                objectVP.Add(R);
                objectXIT.Remove(R);
            }

        }

        // Add all new potential rectangles to objectXIT
        foreach (ViewSpaceRectangle rect in allNewPotentialRects)
            objectXIT.Add(rect.StartX, rect.EndX, rect);
    }

    public Color RegisterNonControllable(CVDetectedObj vmc)
    {
        Color currentC = Color.white;
        if (vmc.Color.Equals(Color.clear))
        {
            currentC = new Color(Random.Range(0f, 1f), Random.Range(0f, 1f), Random.Range(0f, 1f), 1.0f);
            while (assignedColors.Contains(currentC))
                currentC = new Color(Random.Range(0f, 1f), Random.Range(0f, 1f), Random.Range(0f, 1f), 1.0f);
        } else
        {
            currentC = vmc.Color;
        }

        assignedColors.Add(currentC);

        if (!visibleNonControllables.Contains(vmc))
        {
            AngelARUI.Instance.LogDebugMessage("Registered: " + vmc.gameObject.name, true);
            visibleNonControllables.Add(vmc);
        }
         
        return currentC;
    }

    public void DeregisterNonControllable(CVDetectedObj vmc)
    {
        if (!visibleNonControllables.Contains(vmc)) return;

        AngelARUI.Instance.LogDebugMessage("Deregistered: " + vmc.gameObject.name, true);

        visibleNonControllables.Remove(vmc);
        assignedColors.Remove(vmc.Color);
    }

    private void PositionInView()
    {
        const float DISTANCE_FROM_CAM = 50;
        Vector2 padding = new Vector2(0.01f, 0.1f); //Distance we want to keep from the viewport borders.

        Bounds bounds = GetComponent<MeshFilter>().mesh.bounds;    //Get the bounds of the model - these are in local space of the model, axis aligned.
                                                                   //Calculate the max width the object is allowed to have in world space, based on the padding we decided.
        float maxWidth = Vector3.Distance(Camera.main.ViewportToWorldPoint(new Vector3(padding.x, 0.5f, DISTANCE_FROM_CAM)),
            Camera.main.ViewportToWorldPoint(new Vector3(1f - padding.x, 0.5f, DISTANCE_FROM_CAM)));
        //Calculate the scale based on width only - you will have to check if the model is tall instead of wide and check against the aspect of the camera, and act accordingly.
        float scale = (maxWidth / bounds.size.x);
        //Apply the scale to the model.
        transform.localScale = Vector3.one * scale;

        //Position the model at the desired distance.
        Vector3 desiredPosition = DISTANCE_FROM_CAM * Camera.main.transform.forward + Camera.main.transform.position;
        //The max width we calculated is for the entirety of the model in the viewport, so we need to position it so the front of the model is at the desired distance, not the center.
        //You will also have to keep rotation of the camera and the model in mind.
        transform.position = desiredPosition + new Vector3(0, 0, bounds.extents.z * scale);
    }


    private void OnGUI()
    {
        if (!AngelARUI.Instance.PrintVMDebug) return;

        GUIStyle tintableText = new GUIStyle(GUI.skin.box);
        tintableText.normal.background = Texture2D.whiteTexture; // must be white to tint properly
        tintableText.normal.textColor = Color.white; // whatever you want

        if (AABBsPerNonControllable != null && AABBsPerNonControllable.Count > 0)
        {
            foreach (var nonvm in AABBsPerNonControllable.Keys)
            {
                foreach (var rect in AABBsPerNonControllable[nonvm])
                {
                    int screenX = (int)(rect.x);
                    int screenY = (int)(rect.y);
                    int screenW = (int)(rect.width);
                    int screenH = (int)(rect.height);

                    //GUI.backgroundColor = new Color(Random.Range(0f, 1f), Random.Range(0f, 1f), Random.Range(0f, 1f), 0.2f);
                    GUI.backgroundColor = new Color(nonvm.Color.r, nonvm.Color.g, nonvm.Color.b,0.3f);
                    GUI.Box(new Rect(screenX, screenY, screenW, screenH)
                        , "Rect : (" + rect.x
                                                    + "," + rect.y
                                                    + "," + rect.width
                                                    + "," + rect.height, tintableText);
                }
            }
        }

    }

    public Dictionary<VMNonControllable, List<Rect>> GetAllRects() {
        Dictionary<VMNonControllable, List<Rect>> rects = new Dictionary<VMNonControllable, List<Rect>>();
        if (AABBsPerNonControllable == null) return rects;

        foreach (var item in AABBsPerNonControllable.Keys)
        {
            if (!item.IsDestroyed)
                rects.Add((VMNonControllable)item, AABBsPerNonControllable[item]);
        }

        return rects;
    }

    /// <summary>
    /// 00 is LL
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    private int GetObjectID(int x, int y)
    {
        //00 is LL
        int Ycorrected = _imageTex.height - y;
        Color value = _imageTex.GetPixel(x, Ycorrected);

        if (value.a != 0 && value!= Color.black)
        {
            foreach (var item in assignedColors)
            {
                if (Utils.IsSameColor(value, item, 0.02f))
                {
                    value = item;
                    break;
                }
            }
        }
        
        if (colorIDs.ContainsKey(value))
            return colorIDs[value];
        else
            return -1;
        
    }

}
