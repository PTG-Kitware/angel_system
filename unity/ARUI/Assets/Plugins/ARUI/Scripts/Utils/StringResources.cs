using System;
using System.Collections.Generic;
using UnityEngine;

public static class StringResources
{
    public static Dictionary<string, int> LayerToLayerInt;
    public static int LayerToInt(string layer) => LayerToLayerInt[layer];

    //Sounds
    public static string BtnConfirmationSound_path = "Sounds/MRTK_ButtonPress";
    public static string NotificationSound_path = "Sounds/MRTK_Notification";
    public static string NextTaskSound_path = "Sounds/MRTK_Voice_Confirmation";
    public static string MoveStart_path = "Sounds/MRTK_Move_Start";
    public static string MoveEnd_path = "Sounds/MRTK_Move_End";
    public static string WarningSound_path = "Sounds/warning";
    public static string SelectSound_path = "Sounds/MRTK_Select_Secondary";
    public static string VoiceConfirmation_path = "Sounds/MRTK_Select_Main";
    public static string ActionConfirmation_path = "Sounds/Confirmation";

    //Prefabs
    public static string POIHalo_path = "Prefabs/Halo3D";
    public static string Orb_path = "Prefabs/Orb/Orb";
    public static string EyeTarget_path = "Prefabs/EyeTarget";
    public static string ConfNotificationOrb_path = "Prefabs/Orb/OrbConfirmationNotification";
    public static string MultiSelectNotificationOrb_path = "Prefabs/Orb/OrbSelectNotification";
    public static string YesNoNotificationOrb_path = "Prefabs/Orb/OrbYesNoNotification";
    public static string HandPoseManager_path = "Prefabs/HandPoseManager";
    public static string Sid_Tasklist_path = "Prefabs/Sid_Tasklist/Task_Overview";
    public static string TaskOverview_template_path = "Prefabs/Sid_Tasklist/TaskOverview_template";

    public static string dialogue_path = "Prefabs/Dialogue";

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
    public static string HandPoseManager_name = "HandPoseManager";
    public static string NotificationManager_name = "NotificationManager";
}