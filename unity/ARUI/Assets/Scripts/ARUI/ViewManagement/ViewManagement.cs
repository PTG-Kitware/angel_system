using DilmerGames.Core.Singletons;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;

/// <summary>
/// GUI space has 0,0 at top left. Screen space is 0,0 at bottom left.
/// Spacemanager and zBuffer coordinate system = GUI
/// </summary>
public class ViewManagement : Singleton<ViewManagement>
{
    private bool init = false;
    private bool smIsAlive = false;                             /// < true if the current data is valid, false if it is processing in the current frame
    private int padding = 20;                                   /// < buffer in pixels

    private Dictionary<VMControllable, Rect> vmToRect;          /// < AABB: minx, miny, maxX, maxY - SCREEN SPACE
    private List<VMObject> allAABBs;                            /// < AABB: minx, miny, maxX, maxY - GUI coordinate system
    private List<VMNonControllable> allNonControllableAABB;                  /// < AABB: minx, miny, maxX, maxY - GUI coordinate system

    private List<Rect> allEmptyRect;                            /// < AABB: minx, miny, maxX, maxY - GUI coordinate system
    private int objectsInViewSpace = 0;

    ///** For Debugging 
    private List<int[]> debugCappedAABB;                         /// < AABB: minx, miny, maxX, maxY - GUI coordinate system

    private void Start() => StartCoroutine(RunViewManagement());

    private IEnumerator RunViewManagement()
    {
        yield return new WaitForSeconds(3f);

        Debug.Log("View Management Initialized, screen: " + AngelARUI.Instance.ARCamera.pixelWidth + "," + AngelARUI.Instance.ARCamera.pixelHeight);

        init = true;
        allNonControllableAABB = new List<VMNonControllable>();

        yield return new WaitForSeconds(2f);

        while (true)
        {
            debugCappedAABB = null;
            allEmptyRect = null;    

            while (allNonControllableAABB.Count==0)
                yield return new WaitForEndOfFrame();

            SpaceManagement.Instance.CreateIntervaltree(0, AngelARUI.Instance.ARCamera.pixelWidth, AngelARUI.Instance.ARCamera.pixelHeight);
            SpaceManagement.Instance.CreateIntervaltree(1, AngelARUI.Instance.ARCamera.pixelWidth, AngelARUI.Instance.ARCamera.pixelHeight);

            objectsInViewSpace = 0;

            AddAllObjectsToViewSpace();

            smIsAlive = true;

            vmToRect = GetBestLayout();

            if (objectsInViewSpace != 0)
                allEmptyRect = SpaceManagement.Instance.GetAllEmptyRect(0);

            yield return new WaitForSeconds(0.1f);

            //Delete trees of previous interval tree
            SpaceManagement.Instance.DeleteTree(0);
            SpaceManagement.Instance.DeleteTree(1);

            smIsAlive = false;

            yield return new WaitForSeconds(0.1f);

        }

    }

    #region View Management 

    /// <summary>
    /// Add all non controllable objects currently visible to the space manager as full space
    /// </summary>
    private void AddAllObjectsToViewSpace()
    {
        debugCappedAABB = new List<int[]>();

        foreach (var vmc in allNonControllableAABB)
        {
            Rect item = vmc.AABB;
            if (//item!=null && Vector3.Magnitude(item.transform.position - transform.position) > 0.3f &&
                item.width!=0 && item.height!=0)
            {
                int[] AABB = Utils.GetCappedGUI(item);

                if (!(AABB[2] < 0 || AABB[3] < 0))
                {
                    objectsInViewSpace++;
                    SpaceManagement.Instance.AddFullRectToTree(0, AABB);
                    SpaceManagement.Instance.AddRectToTree(1, AABB);

                    debugCappedAABB.Add(AABB);
                }
            }
        }

    }

    /// <summary>
    /// Get best potential new position for every controllable objects
    /// </summary>
    /// <returns></returns>
    private Dictionary<VMControllable, Rect> GetBestLayout()
    {
        VMControllable[] all = FindObjectsOfType<VMControllable>();
        Dictionary<VMControllable, Rect> bestLayout = new Dictionary<VMControllable, Rect>();

        int vmcInView = all.Length;
        foreach (VMControllable obj in all)
        {
            int[] cappedRect = Utils.GetCappedGUI(obj.AABB);

            //Get closest
            if ((cappedRect[2] < 0 || cappedRect[3] < 0))
            {
                bestLayout.Add(obj, Rect.zero);
                continue;
            }

            List<Rect> overlapFull = SpaceManagement.Instance.GetOverlap(1, cappedRect);

            if (overlapFull.Count > 1)
            {
                Vector3 pos = GetClosestEmptyPos(AngelARUI.Instance.ARCamera.WorldToScreenPoint(obj.transform.position),
                obj.AABB, padding);

                Rect newPosRect = new Rect(
                    pos.x - obj.AABB.width / 2,
                    pos.y - obj.AABB.height / 2,
                    obj.AABB.width,
                    obj.AABB.height);

                int[] AABB = Utils.GetCappedGUI(newPosRect);

                if (!(AABB[2] < 0 || AABB[3] < 0))
                {
                    if (vmcInView > 1)
                    {
                        SpaceManagement.Instance.AddFullRectToTree(0, AABB);
                        SpaceManagement.Instance.AddRectToTree(1, AABB);
                    }

                    bestLayout.Add(obj, newPosRect);
                    continue;
                }
            }
            else
            {
                if (vmcInView > 1)
                {
                    SpaceManagement.Instance.AddFullRectToTree(0, cappedRect);
                    SpaceManagement.Instance.AddRectToTree(1, cappedRect);
                }
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
        Rect closestEmptyGUI = SpaceManagement.Instance.GetClosestEmtpy(0, cappedRectScreen);

        return GetClosestPointInRectScreen(
            new int[2] { (int)prevPosInScreenSpace.x, (int)prevPosInScreenSpace.y },
            cappedRectScreen, closestEmptyGUI, padding);
    }

    /// <summary>
    /// Add non controllable tracking
    /// </summary>
    /// <param name="vmc"></param>
    public void RegisterNonControllable(VMNonControllable vmc)
    {
        if (smIsAlive == false && allNonControllableAABB != null && !allNonControllableAABB.Contains(vmc))
            allNonControllableAABB.Add(vmc);
    }

    /// <summary>
    /// Remove non controallble tracking
    /// </summary>
    /// <param name="vmc"></param>
    public void DeRegisterNonControllable(VMNonControllable vmc)
    {
        if (allNonControllableAABB != null)
            allNonControllableAABB.Remove(vmc);
    }

    #endregion

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

    private void OnDestroy()
    {
        if (SpaceManagement.Instance != null)
        {
            SpaceManagement.Instance.DeleteTree(0);
            SpaceManagement.Instance.DeleteTree(1);
        }
    }

    #region Debugging
#if (UNITY_EDITOR)
    public bool printVMDebug = true;

    void OnGUI()
    {
        if (!init || allEmptyRect == null || allEmptyRect.Count == 1 || objectsInViewSpace == 0 || !AngelARUI.Instance.IsVMActiv || !printVMDebug) return;

        float scale = 1f;
        GUIStyle tintableText = new GUIStyle(GUI.skin.box);
        tintableText.normal.background = Texture2D.whiteTexture; // must be white to tint properly
        tintableText.normal.textColor = Color.white; // whatever you want

        foreach (var vmc in vmToRect.Keys)
        {
            Rect rect = vmc.AABB;
            int[] item = Utils.GetCappedGUI(rect);
            GUI.backgroundColor = new Color(255, 255, 255, 0.4f);
            Rect scaledRect = new Rect(item[0] * scale, item[1] * scale, (item[2] - item[0]) * scale, (item[3] - item[1]) * scale);
            GUI.Box(scaledRect, "AABB : (" + item[0]
                                            + "," + item[1]
                                            + "," + (item[2] - item[0])
                                            + "," + (item[3] - item[1]), tintableText);
        }

        //****Draw all empty recs
        for (int i = 0; i < allEmptyRect.Count; i++)
        {
            GUI.backgroundColor = new Color(Random.Range(0f, 1f), Random.Range(0f, 1f), Random.Range(0f, 1f), 0.2f);
            Rect scaledRect = new Rect(allEmptyRect[i].x * scale, allEmptyRect[i].y * scale, allEmptyRect[i].width * scale, allEmptyRect[i].height * scale);
            GUI.Box(scaledRect, "Bounding Box : (" + allEmptyRect[i].x
                                                + "," + allEmptyRect[i].y
                                                + "," + allEmptyRect[i].width
                                                + "," + allEmptyRect[i].height, tintableText);
        }

        //**** Draw all full recs
        if (debugCappedAABB != null && debugCappedAABB.Count > 0)
        {
            foreach (var item in debugCappedAABB)
            {
                GUI.backgroundColor = new Color(255, 0, 0, 0.8f);
                Rect scaledRect = new Rect(item[0] * scale, item[1] * scale, (item[2] - item[0]) * scale, (item[3] - item[1]) * scale);
                GUI.Box(scaledRect, "AABB : (" + item[0]
                                                + "," + item[1]
                                                + "," + (item[2] - item[0])
                                                + "," + (item[3] - item[1]), tintableText);
            }
        }
    }

#endif
    #endregion
}