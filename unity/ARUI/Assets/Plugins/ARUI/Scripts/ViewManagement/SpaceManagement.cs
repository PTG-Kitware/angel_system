using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEngine;

public class ViewRect
{
    public Rect rect;
    public float area;
    public float dist;
}

/// <summary>
/// Space manager follows GUI coordinates (top left is 0 0)
/// </summary>
public class SpaceManagement : Singleton<SpaceManagement>
{
    [DllImport("VMmain")]
    private static extern void create_itervalltree_2i(ulong treeID, ulong width, ulong height);

    [DllImport("VMmain")]
    private static extern void add_fullrect_2i_in_tree(ulong treeID, ulong minx, ulong miny, ulong maxx, ulong maxy);

    [DllImport("VMmain")]
    private static extern void add_rect_2i(ulong treeID, ulong minx, ulong miny, ulong maxx, ulong maxy);

    [DllImport("VMmain")]
    [return: MarshalAs(UnmanagedType.BStr)]
    private static extern string tree_to_string(ulong treeID);

    [DllImport("VMmain")]
    [return: MarshalAs(UnmanagedType.BStr)]
    private static extern string get_all_rect(ulong treeID);

    [DllImport("VMmain")]
    private static extern void release_tree(ulong treeID);

    [DllImport("VMmain")]
    private static extern string combine_in_y(ulong treeID, ulong minx, ulong miny, ulong maxx, ulong maxy, ulong minx2, ulong miny2, ulong maxx2, ulong maxy2);

    [DllImport("VMmain")]
    [return: MarshalAs(UnmanagedType.BStr)]
    private static extern string get_all_overlapping_rect(ulong treeID, ulong minx, ulong miny, ulong maxx, ulong maxy);

    #region Spacemanagement

    /// <summary>
    /// Check if DLL for spacemanagement is loaded.
    /// </summary>
    /// <returns>true if dll is loaded, else false.</returns>
    public bool CheckIfDllLoaded()
    {
        bool loaded = true;
        try
        {
            CreateIntervaltree(99, 12, 12);
        }
        catch (DllNotFoundException ex)
        {
            loaded = false;
            AngelARUI.Instance.LogDebugMessage("VM DLL not found: \n" + ex.Message, true);
        }

        return loaded;
    }

    /// <summary>
    /// Create a 2D interval tree with the given id, width and height.
    /// </summary>
    /// <param name="id">ID of spacetree</param>
    /// <param name="width">in pixels</param>
    /// <param name="height">in pixel</param>
    public void CreateIntervaltree(int id, int width, int height) => create_itervalltree_2i(Convert.ToUInt64(id), Convert.ToUInt64(width), Convert.ToUInt64(height));

    /// <summary>
    /// Add given rectangle to space tree,  
    /// </summary>
    /// <param name="id">ID of space tree rect should be added</param>
    /// <param name="rect">xmin, ymin, xmax, ymax, GUI coordinate system</param>
    public void AddRectToTree(int id, int[] rect) => add_rect_2i(Convert.ToUInt64(id), Convert.ToUInt64(rect[0]), Convert.ToUInt64(rect[1]), Convert.ToUInt64(rect[2]), Convert.ToUInt64(rect[3]));

    /// <summary>
    /// Add full rectangle to space tree,  
    /// </summary>
    /// <param name="id">ID of space tree rect should be added</param>
    /// <param name="rect">xmin, ymin, xmax, ymax, GUI coordinate system</param>
    public void AddFullRectToTree(int id, int[] rect) => add_fullrect_2i_in_tree(Convert.ToUInt64(id), Convert.ToUInt64(rect[0]), Convert.ToUInt64(rect[1]), Convert.ToUInt64(rect[2]), Convert.ToUInt64(rect[3]));

    /// <summary>
    /// Get list of full rectangles that overlap with given rectangle
    /// </summary>
    /// <param name="id">ID of space tree rect should be added</param>
    /// <param name="rect">xmin, ymin, xmax, ymax, GUI coordinate system</param>
    /// <returns></returns>
    public List<Rect> GetOverlap(int id, int[] rect)
    {
        string rects = get_all_overlapping_rect(Convert.ToUInt64(id), Convert.ToUInt64(rect[0]), Convert.ToUInt64(rect[1]), Convert.ToUInt64(rect[2]), Convert.ToUInt64(rect[3]));
        return ParseRectangles(rects);
    }

    /// <summary>
    /// Get closest empty rectangle given rect
    /// </summary>
    /// <param name="ID">ID of 2D spacetree</param>
    /// <param name="rect">Screen coordinate system = origin is bottom left, xmin, ymin, xmax, ymax</param>
    /// <returns>Rect of closest empty one in screen space</returns>
    public Rect GetClosestEmtpy(int ID, int[] rect) => ParseRectangles_Closest(get_all_rect(Convert.ToUInt64(ID)), rect);

    /// <summary>
    /// Get closest empty rectangle given rect
    /// </summary>
    /// <param name="ID">ID of 2D spacetree</param>
    /// <param name="rect">Screen coordinate system = origin is bottom left, xmin, ymin, xmax, ymax</param>
    /// <returns>Rect of closest empty one in screen space</returns>
    public Rect GetClosestEmtpy(int ID, Vector3 point) => ParseRectangles_Closest(get_all_rect(Convert.ToUInt64(ID)), point);

    /// <summary>
    /// Get a list of all empty rectangles
    /// </summary>
    /// <param name="ID">ID of 2D spacetree</param>
    /// <returns>Rectangles in GUI space - 0,0 at top left</returns>
    public List<Rect> GetAllEmptyRect(int ID) => ParseRectangles(get_all_rect(Convert.ToUInt64(ID)));
    #endregion

    #region Rectangles Parsing

    /// <summary>
    /// Returns list of rectangles from stream
    /// </summary>
    /// <param name="rectangleStream">rectangles in GUI space in string</param>
    /// <returns>List of empty rectangles</returns>
    private List<Rect> ParseRectangles(string rectangleStream)
    {
        string[] inst = rectangleStream.Split('(');
        List<Rect> allRects = new List<Rect>();

        //string debugoutput = "";
        //debugoutput += "Screen Size: x" + AngelARUI.Instance.ARCamera.pixelHeight + "y" + AngelARUI.Instance.ARCamera.pixelWidth;

        //first string in the array is the prefix
        for (int i = 1; i < inst.Length; i++)
        {
            string clean = inst[i].Replace("(", "").Replace(")", "").Replace(" ", "");
            string[] current = clean.Split(',');
            //Debug.Log(clean);
            float xmin = Convert.ToSingle(current[0]);
            float xmax = Convert.ToSingle(current[2]);
            float ymin = Convert.ToSingle(current[1]);
            float ymax = Convert.ToSingle(current[3]);

            Rect res = new Rect(xmin, ymin, xmax - xmin, ymax - ymin);
            allRects.Add(res);

            //debugoutput += " (" + xmin + "," + ymin + "," + xmax + "," + ymax + ") [" + (xmax - xmin) * (ymax - ymin) + "]";
        }

        //if (inst.Length>1)
        //    Debug.Log(debugoutput);

        return allRects;
    }

    /// <summary>
    /// Returns rectangle that is the closest to the given rectangle rect
    /// </summary>
    /// <param name="rectangleStream">rectangles in GUI space in string</param>
    /// <param name="rect">Screen coordinate system = origin is bottom left, xmin, ymin, xmax, ymax</param>
    /// <returns></returns>
    private Rect ParseRectangles_Closest(string rectangleStream, int[] rect)
    {
        string[] inst = rectangleStream.Split('(');

        //string debugoutput = "";
        //debugoutput += "Screen Size: x" + AngelARUI.Instance.ARCamera.pixelHeight + "y" + AngelARUI.Instance.ARCamera.pixelWidth + " available: "+ (inst.Length-1).ToString();
        //debugoutput += "Orb Rect: x" + rect[0] + "y" + rect[1] + " xMAX: " + (rect[2]) + " yMAX: " + (rect[3]);

        Rect closestRectX = new Rect(0, 0, 0, 0);
        Rect closestRectY = new Rect(0, 0, 0, 0);

        float minDistX = 999999;
        float minDistY = 999999;

        for (int i = 1; i < inst.Length; i++)
        {
            string clean = inst[i].Replace("(", "").Replace(")", "").Replace(" ", "");
            string[] current = clean.Split(',');
            //debugoutput += "Current: x" + current[0] + "y" + current[1];

            int xmin = Int32.Parse(current[0]);
            int width = (Int32.Parse(current[2]) - xmin);
            if (width < (rect[2] - rect[0]))
                continue;

            int ymin = Int32.Parse(current[1]);
            int height = (Int32.Parse(current[3]) - ymin);
            if (height < (rect[3] - rect[1]))
                continue;

            //debugoutput += "- " + height + ", " + width;

            if (width == AngelARUI.Instance.ARCamera.pixelWidth && height == AngelARUI.Instance.ARCamera.pixelHeight)
                continue;

            //Translate from VM Coordinates into screen coordinates
            Rect emptyRect = new Rect(xmin, AngelARUI.Instance.ARCamera.pixelHeight - ymin - height, width, height);
            float resMaxX = (emptyRect.x + width);
            float resMaxY = (emptyRect.y + height);

            //case: overlap between edge of orb and edge of current empty rect.
            if (rect[0] < emptyRect.x && emptyRect.x < rect[2] && !(rect[3] < emptyRect.y || resMaxY < rect[1])) //check if overlap edge left 
                return emptyRect;
            else if (rect[1] < emptyRect.y && emptyRect.y < rect[3] && !(rect[2] < emptyRect.x || resMaxX < rect[0])) //check if overlap edge bottom
                return emptyRect;
            else if (rect[0] < resMaxX && resMaxX < rect[2] && !(rect[3] < emptyRect.y || resMaxY < rect[1]))  //check if overlap edge right
                return emptyRect;
            else if (rect[1] < resMaxY && resMaxY < rect[3] && !(rect[2] < emptyRect.x || resMaxX < rect[0]))  //check if overlap edge top
                return emptyRect;

            //case: no overlap 
            if (emptyRect.x < rect[0] && (rect[0] - emptyRect.x) < minDistX)
            {  //there is a rectangle to the left of the orb, shorter distance
                minDistX = (rect[0] - emptyRect.x);
                closestRectX = emptyRect;
            }
            else if (rect[2] < resMaxY && (resMaxY - rect[2]) < minDistX)
            {   //orb top edge is above the bottom edge of the empty rect
                minDistX = (resMaxY - rect[2]);
                closestRectX = emptyRect;
            }

            if (emptyRect.y < rect[1] && (rect[1] - emptyRect.y) < minDistY)
            {
                minDistY = (rect[1] - emptyRect.y);
                closestRectY = emptyRect;
            }
            else if (rect[3] < resMaxY && (resMaxY - rect[3]) < minDistY)
            {
                minDistY = (resMaxY - rect[3]);
                closestRectY = emptyRect;
            }
        }

        if (minDistX < minDistY)
            return closestRectX;
        else
            return closestRectY;
    }

    /// <summary>
    /// Returns rectangle that is the closest to the given rectangle rect
    /// </summary>
    /// <param name="rectangleStream">rectangles in GUI space in string</param>
    /// <param name="rect">Screen coordinate system = origin is bottom left, xmin, ymin, xmax, ymax</param>
    /// <returns></returns>
    private Rect ParseRectangles_Closest(string rectangleStream, Vector2 screenSpacePos)
    {
        string[] inst = rectangleStream.Split('(');

        //string debugoutput = "";
        //debugoutput += "Screen Size: x" + AngelARUI.Instance.ARCamera.pixelHeight + "y" + AngelARUI.Instance.ARCamera.pixelWidth + " available: "+ (inst.Length-1).ToString();
        //debugoutput += "Orb Rect: x" + rect[0] + "y" + rect[1] + " xMAX: " + (rect[2]) + " yMAX: " + (rect[3]);

        Rect closestRectX = new Rect(0, 0, 0, 0);
        Rect closestRectY = new Rect(0, 0, 0, 0);

        float minDistX = 999999;
        float minDistY = 999999;

        for (int i = 1; i < inst.Length; i++)
        {
            string clean = inst[i].Replace("(", "").Replace(")", "").Replace(" ", "");
            string[] current = clean.Split(',');
            //debugoutput += "Current: x" + current[0] + "y" + current[1];

            int xmin = Int32.Parse(current[0]);
            int width = (Int32.Parse(current[2]) - xmin);

            int ymin = Int32.Parse(current[1]);
            int height = (Int32.Parse(current[3]) - ymin);

            //debugoutput += "- " + height + ", " + width;

            if (width == AngelARUI.Instance.ARCamera.pixelWidth && height == AngelARUI.Instance.ARCamera.pixelHeight)
                continue;

            //Translate from VM Coordinates into screen coordinates
            Rect emptyRect = new Rect(xmin, AngelARUI.Instance.ARCamera.pixelHeight - ymin - height, width, height);
            float resMaxX = (emptyRect.x + width);
            float resMaxY = (emptyRect.y + height);

            //case: no overlap 
            if (Mathf.Abs(emptyRect.x - screenSpacePos.x) < minDistX)
            {  //there is a rectangle to the left of the orb, shorter distance
                minDistX = Mathf.Abs(emptyRect.x - screenSpacePos.x);
                closestRectX = emptyRect;
            }
            else if (Mathf.Abs( (emptyRect.x+width) - screenSpacePos.x) < minDistX)
            {   //orb top edge is above the bottom edge of the empty rect
                minDistX = Mathf.Abs((emptyRect.x + width) - screenSpacePos.x);
                closestRectX = emptyRect;
            }

            if (Mathf.Abs(emptyRect.y - screenSpacePos.y) < minDistY)
            {
                minDistY = Mathf.Abs(emptyRect.y - screenSpacePos.y);
                closestRectY = emptyRect;
            }
            else if (Mathf.Abs((emptyRect.y + height) - screenSpacePos.y) < minDistY)
            {
                minDistY = Mathf.Abs((emptyRect.y + height) - screenSpacePos.y);
                closestRectY = emptyRect;
            }
        }

        if (minDistX < minDistY)
            return closestRectX;
        else
            return closestRectY;
    }

    /// <summary>
    /// Returns list of rectangles from string. The list is sorted by area. Largest rectangles come first. 
    /// GUI coordinate system (0,0 is top left)
    /// </summary>
    /// <param name="rectangleStream"></param>
    /// <returns>List of rectangles, sorted in descending order by area</returns>
    private List<Rect> ParseRectangles_SortedArea(string rectangleStream)
    {
        string[] inst = rectangleStream.Split('(');
        List<ViewRect> areas = new List<ViewRect>();

        //string debugoutput = "";
        //debugoutput += "Screen Size: x" + AngelARUI.Instance.ARCamera.pixelHeight + "y" + AngelARUI.Instance.ARCamera.pixelWidth;

        for (int i = 1; i < inst.Length; i++)
        {
            string clean = inst[i].Replace("(", "").Replace(")", "").Replace(" ", "");
            string[] current = clean.Split(',');
            //Debug.Log(clean);
            float xmin = Convert.ToSingle(current[0]);
            float ymin = Convert.ToSingle(current[1]);
            float xmax = Convert.ToSingle(current[2]);
            float ymax = Convert.ToSingle(current[3]);
            Rect res = new Rect(xmin, ymin, xmax - xmin, ymax - ymin);

            ViewRect currentRect = new ViewRect();
            currentRect.area = res.width * res.height;
            currentRect.rect = res;
            areas.Add(currentRect);

            //debugoutput += " ("+ xmin +","+ ymin + "," + xmax + "," + ymax +") ["+ (xmax-xmin) * (ymax-ymin) +"]";
        }

        //if (inst.Length>1)
        //Debug.Log(debugoutput);

        areas = areas.OrderByDescending(o => o.area).ToList();
        return areas.Select(x => x.rect).ToList(); ;
    }

    #endregion

    #region Clean-up
    /// <summary>
    /// Free space and delete previous rangetrees
    /// </summary>
    /// <param name="ID">ID of 2D spacetree to be deleted</param>
    public void DeleteTree(int ID)
    {
        try
        {
            release_tree(Convert.ToUInt64(ID));
        }
        catch (Exception ex) { }
    }

    #endregion
}
