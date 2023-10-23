using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using UnityEngine;

public static class ARUISettings 
{
    //**** Orb
    public static readonly float OrbMinDistToUser = 0.6f;    /// <in meters
    public static readonly float OrbMaxDistToUser = 1.1f;    /// <in meters

    // Orb radial behavior - higher values means the element will stay further away from the center of view
    public static readonly float OrbMinViewDegrees = 10f;        ///<The element will stay at least this far away from the center of view
    public static readonly float OrbMaxViewDegSticky = 21f;
    public static readonly float OrbMaxViewDegRegular = 13f;
    public static readonly float OrbMaxViewDegCenter = 10f;

    public static readonly float OrbMoveLerpRegular = 0.7f;
    public static readonly float OrbOutOfFOVThresV = 21;      /// <in radial deg

    public static readonly Color OrbMessageBGColor = new Color(0.06f, 0.06f, 0.06f, 0.5f);
    public static readonly int OrbMessageMaxCharCountPerLine = 70;
    public static readonly int OrbNoteMaxCharCountPerLine = 50;

    //**** Task List
    public static readonly Color TaskFutureColor = Color.gray;
    public static readonly Color TaskCurrentColor = Color.white;
    public static readonly Color TaskDoneColor = new Color(0.30f, 0.60f, 0.30f);

    public static readonly int TaskMaxNumTasksOnList = 7;     /// Must be >=1 and an odd number

    public static readonly float TasksMinDistToUser = 0.6f;   /// <in meters
    public static readonly float TasksMaxDistToUser = 1f;     /// <in meters

    //**** Confirmation Dialogue
    public static readonly float DialogueTimeInSeconds = 8f;  /// <How much time the user has to decide (excluding the time the use is loking at the ok button

    //**** Dwell Button
    public static bool EyeDwellAllowed = true;
    public static readonly float EyeDwellTime = 6f;      /// <How long the user has to look at the button to trigger the event
    public static readonly Color BtnBaseColor = new Color(0.5377358f, 0.5377358f, 0.5377358f, 0.24f);
    public static readonly Color BtnActiveColor = new Color(0.7f, 0.7f, 0.8f, 0.4f);
    public static readonly Color BtnLoadingDiscColor = new Color(0.8f, 0.8f, 0.8f);

    //**** View Management
    public static readonly int Padding = 20;             /// < buffer in pixels
    public static readonly int VMPixelIteration = 1;
    public static readonly int SMPixelSkip = 2;          /// < Adjust the resolution of the SM algorithm

}
