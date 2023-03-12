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
    private bool _init = false;
    private bool _smIsAlive = false;                             /// < true if the current data is valid, false if it is processing in the current frame
    private int _padding = 20;                                   /// < buffer in pixels

    private Dictionary<VMControllable, Rect> _vmToRect;          /// < AABB: minx, miny, maxX, maxY - SCREEN SPACE
    private List<VMNonControllable> _allNonControllableAABB;      /// < AABB: minx, miny, maxX, maxY - GUI coordinate system

    private List<Rect> _allEmptyRect;                             /// < AABB: minx, miny, maxX, maxY - GUI coordinate system
    private int _objectsInViewSpace = 0;
    private int _minPixelSize = 1;

    ///** For Debugging 
    private List<int[]> _debugCappedAABB;                         /// < AABB: minx, miny, maxX, maxY - GUI coordinate system


    private void Start() => StartCoroutine(RunViewManagement());

    private IEnumerator RunViewManagement()
    {
        Debug.Log("View Management Initialized, screen: " + AngelARUI.Instance.ARCamera.pixelWidth + "," + AngelARUI.Instance.ARCamera.pixelHeight);

        _init = true;
        _allNonControllableAABB = new List<VMNonControllable>();

        while (true)
        {
            _debugCappedAABB = null;
            _allEmptyRect = null;    

            while (_allNonControllableAABB.Count==0)
                yield return new WaitForEndOfFrame();

            SpaceManagement.Instance.CreateIntervaltree(0, AngelARUI.Instance.ARCamera.pixelWidth, AngelARUI.Instance.ARCamera.pixelHeight);
            SpaceManagement.Instance.CreateIntervaltree(1, AngelARUI.Instance.ARCamera.pixelWidth, AngelARUI.Instance.ARCamera.pixelHeight);

            _objectsInViewSpace = 0;

            AddAllObjectsToViewSpace();

            _smIsAlive = true;

            _vmToRect = GetBestLayout();

            if (_objectsInViewSpace != 0)
                _allEmptyRect = SpaceManagement.Instance.GetAllEmptyRect(0);

            yield return new WaitForSeconds(0.1f);

            //Delete trees of previous interval tree
            SpaceManagement.Instance.DeleteTree(0);
            SpaceManagement.Instance.DeleteTree(1);

            _smIsAlive = false;

            yield return new WaitForSeconds(0.1f);

        }

    }

    #region View Management 

    /// <summary>
    /// Add all non controllable objects currently visible to the space manager as full space
    /// </summary>
    private void AddAllObjectsToViewSpace()
    {
        _debugCappedAABB = new List<int[]>();

        foreach (var vmc in _allNonControllableAABB)
        {
            Rect item = vmc.AABB;
            if (item.width> _minPixelSize && item.height> _minPixelSize) { 
            
                int[] AABB = Utils.GetCappedGUI(item);

                if (AABB[2] > _minPixelSize && AABB[3] > _minPixelSize)
                {
                    _objectsInViewSpace++;
                    SpaceManagement.Instance.AddFullRectToTree(0, AABB);
                    SpaceManagement.Instance.AddRectToTree(1, AABB);

                    _debugCappedAABB.Add(AABB);
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
            if ((cappedRect[2] <= 0 || cappedRect[3] <= 0))
            {
                bestLayout.Add(obj, Rect.zero);
                continue;
            }

            List<Rect> overlapFull = SpaceManagement.Instance.GetOverlap(1, cappedRect);

            if (overlapFull.Count > 1)
            {
                Vector3 pos = GetClosestEmptyPos(AngelARUI.Instance.ARCamera.WorldToScreenPoint(obj.transform.position),
                obj.AABB, _padding);

                Rect newPosRect = new Rect(
                    pos.x - obj.AABB.width / 2,
                    pos.y - obj.AABB.height / 2,
                    obj.AABB.width,
                    obj.AABB.height);

                int[] AABB = Utils.GetCappedGUI(newPosRect);

                if (AABB[2] > _minPixelSize && AABB[3] > _minPixelSize)
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
        if (_smIsAlive == false && _allNonControllableAABB != null && !_allNonControllableAABB.Contains(vmc))
            _allNonControllableAABB.Add(vmc);
    }

    /// <summary>
    /// Remove non controallble tracking
    /// </summary>
    /// <param name="vmc"></param>
    public void DeRegisterNonControllable(VMNonControllable vmc)
    {
        if (_allNonControllableAABB != null)
            _allNonControllableAABB.Remove(vmc);
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
        if (_vmToRect != null && _vmToRect.ContainsKey(vmC))
            return _vmToRect[vmC];
        else
            return Rect.zero;
    }

    #region Debugging

#if (UNITY_EDITOR)
    public bool printVMDebug = true;

    void OnGUI()
    {
        if (!_init || _allEmptyRect == null || _allEmptyRect.Count == 1 || _objectsInViewSpace == 0 || !AngelARUI.Instance.IsVMActiv || !printVMDebug) return;

        float scale = 1f;
        GUIStyle tintableText = new GUIStyle(GUI.skin.box);
        tintableText.normal.background = Texture2D.whiteTexture; // must be white to tint properly
        tintableText.normal.textColor = Color.white; // whatever you want

        foreach (var vmc in _vmToRect.Keys)
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
        for (int i = 0; i < _allEmptyRect.Count; i++)
        {
            GUI.backgroundColor = new Color(Random.Range(0f, 1f), Random.Range(0f, 1f), Random.Range(0f, 1f), 0.2f);
            Rect scaledRect = new Rect(_allEmptyRect[i].x * scale, _allEmptyRect[i].y * scale, _allEmptyRect[i].width * scale, _allEmptyRect[i].height * scale);
            GUI.Box(scaledRect, "Bounding Box : (" + _allEmptyRect[i].x
                                                + "," + _allEmptyRect[i].y
                                                + "," + _allEmptyRect[i].width
                                                + "," + _allEmptyRect[i].height, tintableText);
        }

        //**** Draw all full recs
        if (_debugCappedAABB != null && _debugCappedAABB.Count > 0)
        {
            foreach (var item in _debugCappedAABB)
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