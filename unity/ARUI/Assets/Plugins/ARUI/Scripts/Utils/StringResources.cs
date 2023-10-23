using System;
using System.Collections.Generic;
using UnityEngine;

public static class StringResources
{
    public static Dictionary<string, int> LayerToLayerInt;
    public static int LayerToInt(string layer) => LayerToLayerInt[layer];

    //Sounds
    public static string ConfirmationSound_path = "Sounds/MRTK_ButtonPress";
    public static string NotificationSound_path = "Sounds/MRTK_Notification";
    public static string NextTaskSound_path = "Sounds/MRTK_Voice_Confirmation";
    public static string MoveStart_path = "Sounds/MRTK_Move_Start";
    public static string MoveEnd_path = "Sounds/MRTK_Move_End";
    public static string WarningSound_path = "Sounds/warning";
    public static string SelectSound_path = "Sounds/MRTK_Select_Secondary";

    //Prefabs
    public static string POIHalo_path = "Prefabs/Halo3D";
    public static string Orb_path = "Prefabs/Orb";
    public static string EyeTarget_path = "Prefabs/EyeTarget";
    public static string ConfNotification_path = "Prefabs/ConfirmationNotification";
    public static string HandPoseManager_path = "Prefabs/HandPoseManager";
    public static string Sid_Tasklist_path = "Prefabs/Sid_Tasklist/Task_Overview";
    public static string Sid_TaskListComponent_path = "Prefabs/Sid_Tasklist/Flashcards/TaskListComponent";
    public static string Sid_TaskOverview_Container_path = "Prefabs/Sid_Tasklist/Containers/Secondary_TaskOverview_Container";
    public static string Sid_MainTaskOverview_Container_path = "Prefabs/Sid_Tasklist/Containers/Main_TaskOverview_Container";

    //Textures
    public static string zBufferTexture_path = "Textures/zBuffer";
    public static string zBufferMat_path = "Materials/zBuffer";

    //Used Layers
    public static string UI_layer = "UI";
    public static string VM_layer = "VM";
    public static string zBuffer_layer = "zBuffer";
    public static string Hand_layer = "Hand";
    public static string spatialAwareness_layer = "Spatial Awareness";

    // GO names
    public static string tasklist_name = "TaskOverview";
    public static string orb_name = "OrbAssistant";
    public static string eyeGazeManager_name = "EyeGazeManager";
    public static string dataManager_name = "DataManager";
    public static string audioManager_name = "AudioManager";
    public static string confirmationWindow_name = "ConfirmatioWindow";
    public static string HandPoseManager_name = "HandPoseManager";
}