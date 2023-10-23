using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using Random = UnityEngine.Random;

/// <summary>
/// GUI space has 0,0 at top left. Screen space is 0,0 at bottom left.
/// Spacemanager and zBuffer coordinate system = GUI
/// </summary>
public class ViewManagement : Singleton<ViewManagement>
{
    private bool _init = false;
    private bool _smIsAlive = false;                             /// < true if the current data is valid, false if it is processing in the current frame

    private Dictionary<VMControllable, Rect> vmToRect;          /// < AABB: minx, miny, maxX, maxY - SCREEN SPACE
    private List<VMNonControllable> _allCoarseNC;                /// < AABB: minx, miny, maxX, maxY - GUI coordinate system
    private Dictionary<VMNonControllable, List<Rect>> allProcessedNC;

    private List<Rect> _allEmptyRect;                             /// < AABB: minx, miny, maxX, maxY - GUI coordinate system
    private int _objectsInViewSpace = 0;

    private void Start() => StartCoroutine(RunViewManagement());

    private IEnumerator RunViewManagement()
    {
        Debug.Log("View Management Initialized, screen: " + AngelARUI.Instance.ARCamera.pixelWidth + "," + AngelARUI.Instance.ARCamera.pixelHeight);

        _init = true;
        _allCoarseNC = new List<VMNonControllable>();

        while (true)
        {
            _debugCappedAABB = null;
            _allEmptyRect = null;

            int counter = 0;
            while (counter == 0)
            {
                yield return new WaitForEndOfFrame();
                if (ProcessObjectVisibility.Instance!=null)
                {
                    allProcessedNC = ProcessObjectVisibility.Instance.GetAllRects();
                    if (allProcessedNC.Count > 0) { }
                        counter = 1;
                }
                
                if (_allCoarseNC.Count>0)
                    counter = 1;
            }

            SpaceManagement.Instance.CreateIntervaltree(0, AngelARUI.Instance.ARCamera.pixelWidth, AngelARUI.Instance.ARCamera.pixelHeight);
            SpaceManagement.Instance.CreateIntervaltree(1, AngelARUI.Instance.ARCamera.pixelWidth, AngelARUI.Instance.ARCamera.pixelHeight);

            _objectsInViewSpace = 0;

            AddAllObjectsToViewSpace();

            _smIsAlive = true;

            vmToRect = GetBestLayout();

            if (_objectsInViewSpace != 0)
                _allEmptyRect = SpaceManagement.Instance.GetAllEmptyRect(0);

            yield return new WaitForSeconds(0.05f);

            //Delete trees of previous interval tree
            SpaceManagement.Instance.DeleteTree(0);
            SpaceManagement.Instance.DeleteTree(1);

            _smIsAlive = false;

            yield return new WaitForSeconds(0.05f);
        }

    }

    #region View Management 

    /// <summary>
    /// Add all non controllable objects currently visible to the space manager as full space
    /// </summary>
    private void AddAllObjectsToViewSpace()
    {
        _debugCappedAABB = new List<int[]>();

        foreach (var vmnc in _allCoarseNC)
        {
            if (vmnc != null)
            {
                Rect item = vmnc.AABB;
                if (item.width > ARUISettings.VMPixelIteration && item.height > ARUISettings.VMPixelIteration &&
                    Vector3.Magnitude(vmnc.transform.position - AngelARUI.Instance.ARCamera.transform.position) > 0.2f)
                {
                    int[] AABB = Utils.GetCappedGUI(item);

                    if ((AABB[2] - AABB[0]) > ARUISettings.VMPixelIteration && (AABB[3] - AABB[1]) > ARUISettings.VMPixelIteration)
                    {
                        _objectsInViewSpace++;
                        SpaceManagement.Instance.AddFullRectToTree(0, AABB);
                        SpaceManagement.Instance.AddRectToTree(1, AABB);

                        _debugCappedAABB.Add(AABB);
                    }
                }
            }
        }

        foreach (var vmnc in allProcessedNC.Keys)
        {
            if (vmnc != null)
            {
                foreach (var item in allProcessedNC[vmnc])
                {
                    if (Vector3.Magnitude(vmnc.transform.position - AngelARUI.Instance.ARCamera.transform.position) > 0.2f)
                    {
                        int[] AABB = Utils.GetCappedGUI(item);

                        if ((AABB[2] - AABB[0]) > ARUISettings.VMPixelIteration && (AABB[3] - AABB[1]) > ARUISettings.VMPixelIteration)
                        {
                            _objectsInViewSpace++;
                            SpaceManagement.Instance.AddFullRectToTree(0, AABB);
                            SpaceManagement.Instance.AddRectToTree(1, AABB);

                            _debugCappedAABB.Add(AABB);
                        }
                    }
                }
            }
        }

    }

    /// <summary>
    /// Get best potential new position for every controllable objects
    /// </summary>
    /// <returns></returns>
    /// 
    private Dictionary<VMControllable, Rect> GetBestLayout()
    {
        List<VMControllable> all = new List<VMControllable>();
        VMControllable[] allOther = FindObjectsOfType<VMControllable>();
        foreach (var vmc in allOther)
        {
            all.Add(vmc);
        }

        Dictionary<VMControllable, Rect> bestLayout = new Dictionary<VMControllable, Rect>();

        foreach (VMControllable obj in all)
        {
            int[] cappedRect = Utils.GetCappedGUI(obj.AABB);

            //don't add if zero area
            if (((cappedRect[2] - cappedRect[0]) <= 0 || (cappedRect[3] - cappedRect[1]) <= 0))
            {
                bestLayout.Add(obj, Rect.zero);
                continue;
            }

            //check overlap
            bool overlap = false;
            List<Rect> overlapFull = SpaceManagement.Instance.GetOverlap(1, cappedRect);
            if (overlapFull.Count > 1)
                overlap = true;

            if (overlap )
            {
                Vector3 posScreen;
                posScreen = GetClosestEmptyPos(
                    AngelARUI.Instance.ARCamera.WorldToScreenPoint(obj.transform.position),
                    obj.AABB, ARUISettings.Padding);
               
                Rect newPosRect = new Rect(
                    posScreen.x - obj.AABB.width / 2, posScreen.y - obj.AABB.height / 2,
                    obj.AABB.width, obj.AABB.height);

                int[] AABB = Utils.ScreenToGUI(newPosRect);

                if ((AABB[2] - AABB[0]) > ARUISettings.VMPixelIteration && (AABB[3] - AABB[1]) > ARUISettings.VMPixelIteration)
                {
                    SpaceManagement.Instance.AddFullRectToTree(0, AABB);
                    SpaceManagement.Instance.AddRectToTree(1, AABB);

                    bestLayout.Add(obj, newPosRect);
                    continue;
                }
            }
            else
            {
                SpaceManagement.Instance.AddFullRectToTree(0, cappedRect);
                SpaceManagement.Instance.AddRectToTree(1, cappedRect);
            }

            bestLayout.Add(obj, Rect.zero);
        }

        return bestLayout;
    }


    /// <summary>
    /// Get position in closest empty rectangle based 
    /// </summary>
    /// <param name="prevPosInScreenSpace">previous position in screen space of objRectGUI</param>
    /// <param name="objRectGUI">rectangle of the current rect</param>
    /// <param name="padding">added to bounds of objRectGui</param>
    /// <returns></returns>
    private Vector3 GetClosestEmptyPos(Vector3 prevPosInScreenSpace, Rect objRectGUI, int padding)
    {
        int[] cappedRectScreen = Utils.GetCappedScreen(objRectGUI);
        Rect closestEmptyScreen = SpaceManagement.Instance.GetClosestEmtpy(0, cappedRectScreen);

        return GetClosestPointInRectScreen(
            new int[2] { (int)prevPosInScreenSpace.x, (int)prevPosInScreenSpace.y },
            cappedRectScreen, closestEmptyScreen, padding);
    }

    /// <summary>
    /// Get position in closest empty rectangle based 
    /// </summary>
    /// <param name="prevPosInScreenSpace">previous position in screen space of objRectGUI</param>
    /// <param name="objRectGUI">rectangle of the current rect</param>
    /// <param name="padding">added to bounds of objRectGui</param>
    /// <returns></returns>
    private Vector3 GetClosestEmptyNoOverlapPos(Vector3 prevPosInScreenSpace, Vector3 targetPosInScreenPos,
        Rect objRectGUI, int padding)
    {
        int[] cappedRectScreen = Utils.GetCappedScreen(objRectGUI);
        Rect closestEmptyScreen = SpaceManagement.Instance.GetClosestEmtpy(0, targetPosInScreenPos);

        return GetClosestPointInRectScreen(
            new int[2] { (int)prevPosInScreenSpace.x, (int)prevPosInScreenSpace.y },
            cappedRectScreen, closestEmptyScreen, padding);
    }

    #endregion

    /// <summary>
    /// Add non controllable tracking
    /// </summary>
    /// <param name="vmc"></param>
    public void RegisterNonControllable(VMNonControllable vmc)
    {
        if (_smIsAlive == false && _allCoarseNC != null && !_allCoarseNC.Contains(vmc) && !(vmc is CVDetectedObj))
            _allCoarseNC.Add(vmc);

    }

    /// <summary>
    /// Remove non controallble tracking
    /// </summary>
    /// <param name="vmc"></param>
    public void DeRegisterNonControllable(VMNonControllable vmc)
    {
        if (_allCoarseNC != null)
            _allCoarseNC.Remove(vmc);
    }

    /// <summary>
    /// TODO
    /// </summary>
    /// <param name="prevPointScreen">previous position in screen space of objRectGUI</param>
    /// <param name="fullRectScreen"></param>
    /// <param name="closestEmptyRectScreen"></param>
    /// <param name="padding"></param>
    /// <returns></returns>
    private Vector3 GetClosestPointInRectScreen(int[] prevPointScreen, int[] fullRectScreen, Rect closestEmptyRectScreen, int padding)
    {
        int newX = prevPointScreen[0];
        int newY = prevPointScreen[1];

        int height = (int)(fullRectScreen[3] - fullRectScreen[1]);
        int width = (int)(fullRectScreen[2] - fullRectScreen[0]);

        if (fullRectScreen[1] < closestEmptyRectScreen.y)
        { //check if controllable coming from x bottom
            newY = (int)closestEmptyRectScreen.y + padding + (height / 2);
        }
        else if ((closestEmptyRectScreen.y + closestEmptyRectScreen.height) < fullRectScreen[1] + height)
        { //check if controllable coming from x top
            int maxY = (int)(closestEmptyRectScreen.y + closestEmptyRectScreen.height);
            newY = maxY - padding - (height / 2);
        }

        if (fullRectScreen[0] < closestEmptyRectScreen.x)
        { //check if controllable coming from x left
            newX = (int)closestEmptyRectScreen.x + padding + (width / 2);
        }
        else if ((closestEmptyRectScreen.x + closestEmptyRectScreen.width) < fullRectScreen[0] + width)
        { //check if controllable coming from x right
            int maxX = (int)(closestEmptyRectScreen.x + closestEmptyRectScreen.width);
            newX = maxX - padding - (width / 2);
        }

        return new Vector3(newX, newY, 0);
    }

    /// <summary>
    /// TODO
    /// </summary>
    /// <param name="prevPointScreen">previous position in screen space of objRectGUI</param>
    /// <param name="fullRectScreen"></param>
    /// <param name="closestEmptyRectScreen"></param>
    /// <param name="padding"></param>
    /// <returns></returns>
    private Vector3 GetClosestPointNoOverlap(int[] prevPointScreen, int[] controllableRect, Rect closestEmptyRectScreen, int padding)
    {
        int newX = prevPointScreen[0];
        int newY = prevPointScreen[1];

        int height = (int)(controllableRect[3] - controllableRect[1]);
        int width = (int)(controllableRect[2] - controllableRect[0]);

        if (closestEmptyRectScreen.y < newY)
        {
            int maxY = (int)(closestEmptyRectScreen.y + closestEmptyRectScreen.height);
            newY = maxY - padding - (height / 2);
        }

         return new Vector3(
            closestEmptyRectScreen.y + (closestEmptyRectScreen.width/ 2),
            newY, 0);
    }


    /// <summary>
    /// Returns best choice of empty rectangle for the given controllable, if no best rectangle is availalbe, return zero
    /// </summary>
    /// <param name="vmC"></param>
    /// <returns></returns>
    public Rect GetBestEmptyRect(VMControllable vmC)
    {
        if (vmToRect != null && vmToRect.ContainsKey(vmC))
            return vmToRect[vmC];
        else
            return Rect.zero;
    }


    #region FOR DEBUGGING ONLY
    ///** For Debugging 
    private List<int[]> _debugCappedAABB;                         /// < AABB: minx, miny, maxX, maxY - GUI coordinate system

#if (UNITY_EDITOR)

    public bool printVM = false;
    public void Update()
    {
        if (Input.GetKeyUp(KeyCode.L))
            printVM = !printVM;
    }

    //[GUITarget(1)]

    void OnGUI()
    {
        if (!AngelARUI.Instance.PrintVMDebug) return;
        if (!_init || _allEmptyRect == null || _allEmptyRect.Count == 1 || _objectsInViewSpace == 0 || !AngelARUI.Instance.IsVMActiv) return;

        float scale = 1f;
        GUIStyle tintableText = new GUIStyle(GUI.skin.box);
        tintableText.normal.background = Texture2D.whiteTexture; // must be white to tint properly
        tintableText.normal.textColor = Color.white; // whatever you want

        //GUI.backgroundColor = new Color(255, 255, 255, 0.7f);
        //GUI.Box(new Rect(orbrect.x * scale, orbrect.y * scale, (orbrect.width - orbrect.x) * scale, (orbrect.height - orbrect.y) * scale), "Orb", tintableText);

        ////**** Draw all bright rects
        //if (allCappedBrightAABBs != null && allCappedBrightAABBs.Count > 0)
        //{
        //    Debug.Log("Count bright: " + allCappedBrightAABBs.Count);
        //    foreach (var item in allCappedBrightAABBs)
        //    {
        //        GUI.backgroundColor = new Color(255, 0, 0, 0.8f);
        //        GUI.Box(new Rect(item[0], item[1], item[2] - item[0], item[3] - item[1]), "Bright AABB : (" + item[0]
        //                                        + "," + item[1]
        //                                        + "," + (item[2] - item[0])
        //                                        + "," + (item[3] - item[1]), tintableText);
        //    }
        //}

        if (printVM)
        {
            //****Draw all empty recs
            for (int i = 0; i < _allEmptyRect.Count; i++)
            {
                GUI.backgroundColor = new Color(Random.Range(0f, 1f), Random.Range(0f, 1f), Random.Range(0f, 1f), 0.2f);
                Rect scaledRect = new Rect(_allEmptyRect[i].x * scale, _allEmptyRect[i].y * scale, _allEmptyRect[i].width * scale, _allEmptyRect[i].height * scale);
                GUI.Box(scaledRect, "Bounding Box : (" + _allEmptyRect[i].x
                                                    + "," + _allEmptyRect[i].y
                                                    + "," + _allEmptyRect[i].width
                                                    + "," + _allEmptyRect[i].height, tintableText);
            }
        }

        ////**** Draw all full recs
        //if (allCappedAABBs != null && allCappedAABBs.Count > 0)
        //{
        //    foreach (var item in allCappedAABBs)
        //    {
        //        GUI.backgroundColor = new Color(255, 0, 0, 0.8f);
        //        Rect scaledRect = new Rect(item[0] * scale, item[1] * scale, (item[2] - item[0]) * scale, (item[3] - item[1]) * scale);
        //        GUI.Box(scaledRect, "AABB : (" + item[0]
        //                                        + "," + item[1]
        //                                        + "," + (item[2] - item[0])
        //                                        + "," + (item[3] - item[1]), tintableText);
        //    }
        //}
    }


#endif
    #endregion
}