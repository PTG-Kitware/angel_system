# ARUI - 3D UI for angel system

Env: 2020.3.25f and MRTK 2.7.3

## How to use ARUI:

1) Create an empty GameObject in the highest hierarchy level at pos (0,0,0) and scale (1,1,1)
2) Add the AngelARUI script to it. This script generates all necessary UI elements at runtime
3) Call the AngelARUI methods from another script

## Example Scene

The Unity scene 'SampleScene' in folder 'Plugins/ARUI/Scenes' shows how one can use the AngelARUI script. Other than the MRTK toolkit, there are two important components in this scene: An object with the 'AngelARUI' script attached and a script 'ExampleScript' that calls the functions of AngelARUI (at 'DummyTestDataGenerator').

## Scene Setup
If you build your own scene using the AngelARUI script, there are a few things to note:
1) The ARUI uses layers to detect various collisions (e.g. between eye gaze and UI elements). It is important that the layer "UI" exists in the
Unity project and the layer index should be 5. Please reserve the layer for the ARUI and do not set your own objects with that layer
2) Make sure that the MRTK toolkit behavior in your scene is correctly set: (or use the AngelARUI settings file - AngelMRTKSettingsProfile)
    1) Tab 'Input' -> Pointers -> Eye-tracking has to be enabled
    2) Tab 'Input' -> Pointers -> Pointing Raycast Layer Masks -> There should be a layer called "UI"
    3) Tab 'Input' -> Pointers -> Assign a pointer to the 'ShellHandRayPointer_ARUI' prefab in Resource/ARUI/Prefabs/ ( {'articulated hand', 'generic openVR'} and 'any')
    4) Tab 'Input' -> Speech -> add keyword 'stop' (the parameters at the keyword can be left None or null)
    5) Tab 'Input' -> Articulated Hand Tracking -> Assign the prefab 'Resources/ARUI/Prefabs/HandJoint/' to 'handjoint' model (important for view management)
    
3) In the hierarchy, at the main camera, search for the "GazeProvider" script and select Raycast Layer Masks -> "UI" (if not already selected)

## Functions, Testing and Debugging
The ARUI can be customized as follows:
* 'AngelARUI.Instance.ShowDebugEyeGazeTarget(..)' enable/disable an eye-gaze debug cube if the user is looking at a component in the ARUI. The default is false.
* 'AngelARUI.Instance.ShowDebugMessagesInLogger(..)' enable/disable debugging messages in the logger window (see example scene), Default is true
* 'AngelARUI.Instance.SetViewManagement(..)' enable/disable view management. Default is true
* 'AngelARUI.Instance.MuteAudio(..)' mute or unmute audio instructions
* File 'ARUISettings.cs' contains some design variables (use with caution)

All features with the exception of TTS (audio instructions) should work with holograhic remoting.

### Set task graph:
The main part of the UI is the orb; the orb tells the user what task the user is currently on. In addition to the orb, there is also a task list in the scene. (Can be toggled using eye-gaze dwelling at the button above the orb)

For now, the ARUI supports a two-level task graph. To set the task graph, call 'AngelARUI.Instance.SetTasks(tasks);' where 'tasks' could look like this:
```
    string[,] tasks =
    {
        {"0", "Text example MainTask 1"},
        {"1", "Text example Subtask 1 of MainTask 1"},
        {"1", "Text example Subtask 2 of MainTask 1"},
        {"1", "Text example Subtask 2 of MainTask 1"},
        {"1", "Text example Subtask 2 of MainTask 1"},
        {"0", "Text example MainTask 2"},
        {"0", "Text example MainTask 3"},
        {"1", "Text example Subtask 1 of MainTask 3"},
        {"1", "Text example Subtask 2 of MainTask 3"},
    };
 ```

The first column indicates if the row is a main task (0) or subtask (1) of the last main task. The second column provides the text of the task.
To set the current task, the user has to do, call: 'AngelARUI.Instance.SetCurrentTaskID(index);' The integer value 'index' presents the row index in the 'tasks' array. (eg. Index 4 would be {"1", "Text example Subtask 2 of MainTask 1"}). If the index does not match the given task graph, the function call is ignored. Please note that a main task can not be set as a current task. If the function is called with an index of a main task, the next subtask will be set (e.g. index 0 would be {"1", "Text example Subtask 1 of MainTask 1"}). Also, for now, we assume that the tasks must be executed sequentially. If index 4 is set, then all tasks before 4 are assumed as done, and all tasks after 4 are assumed as not done.

Examples can be found in "TapTestData.cs". The tasklist can be toggled via script using AngelARUI.Instance.ToggleTasklist();, but mostly the user is in charge.

Overall, if you call 'AngelARUI.Instance.SetCurrentTaskID(index);' the orb message will change, the task list will refresh and the user will hear the instructions (only in build).

### Notifications (beta)
At the moment, the ARUI supports skip notifications and a confirmation dialogue. For the skip notification, The notification message can be changed, before building and running the project, in the AngelARUI behavior script in the Editor. As soon as a skip notification is called, it is removed if a new task was set (through SetCurrentTaskID(..) or by calling 'AngelARUI.Instance.ShowSkipNotification(false);'

#### Confirmation Dialogue 
Here is an example of how to call the confirmation dialogue (found in ExampleScript.cs). For now, the purpose of the confirmation dialogue is to ask the user for permission to execute an action if the NLP node of the system detected a certain user intent (e.g., the user wanted to go to the next task)
```
int next = 2;
//1) Set message (e.g., "Did you mean '{user intent}'?"
InterpretedAudioUserIntentMsg intentMsg = new InterpretedAudioUserIntentMsg();
intentMsg.user_intent = "Go to the next task";

//2) Set event that should be triggered if user confirms
AngelARUI.Instance.SetUserIntentCallback((intent) => { AngelARUI.Instance.SetCurrentTaskID(next); });

//4) Show dialogue to user
AngelARUI.Instance.TryGetUserFeedbackOnUserIntent(intentMsg);
```

## Build, Deploy and Run
### Build and Deploy
Before you build the project, make sure that the following layers are defined: 'zBuffer' (24), 'Hand' (25), 'VM' (26) and 'UI' (5). The layer 'Spatial Awareness' (31) is used by the ARUI as well, but usually created if MRTK is imported. 

The building process is the same as a regular HL2 application, except that before you build the app package in VS, the VMMain.dll (Plugins\WSAPlayer\ARM64) has to be added to the projects. (Right click on the UWP Project in the explorer in VS, 'Add' -> 'External File' -> VMMain.dll. Set content to "True". 

After deployment, when you run the app for the first time, make sure to give permission to the eye-tracking and it is crucial that the calibration is done properly.

# UI and Interactions
* The UI uses eye gaze as input. The user can enable and disable the task list by looking at the button next to the white orb. The position of the orb and the tasklist can be adjusted using the tap gesture (near interactions) or the ray cast (far interactions).
* Audio task instructions can be stopped (just once) with keyword 'stop'. 
* The confirmation button on the confirmation dialogue can be triggered using eye-gaze or touching (index finger)

## Limitations
- Eye tracking might not be reliable if the user wears glasses.
- At start-up, it might take a few seconds until the eye gaze rays is reliable
- If it is recognized by the system that a new user uses the application, the eye tracking calibration might start. This is good since eye tracking is not reliable if not correctly calibrated to the current user.
- TextToSpeech only works in build
- If eye calibration is not ideal, one has to manually go to the hololens2 settings and rerun the eye calibration

## 3rd Party Libraries and Assets
3D Model for testing - https://grabcad.com/library/honda-gx-160
MRTK 2.7.3 - https://github.com/microsoft/MixedRealityToolkit-Unity/releases/tag/v2.7.3
Shapes - https://assetstore.unity.com/packages/tools/particles-effects/shapes-173167
Flat Icons - https://assetstore.unity.com/packages/2d/gui/icons/ux-flat-icons-free-202525
Simple Hand Pose Detector - https://github.com/RobJellinghaus/MRTK_HL2_HandPose/tree/main

## Changelog

10/5/23:
* Show Next/Previous task at orb
* Orb scales with distance to user, so the content is still legible if orb is further away
* Warning and error notification + growing disc to get user's attention 
* Fix vs. Follow Mode: If orb is dragged and hand is closed, the orb is fixed to a 3D position, Drag and pinch to undo

6/1/23: 
* Adding fall back options if eye gaze does not work: Enable/Disable attention-based task overview and allows users to 'touch' the task list button, in addition to dwelling

5/31/23: 
* Adding Space Managment, a more accurate representation of full space for the view management algorithm
* Adding 'RegisterDetectedObject(..)' and 'DeRegisterDetectedObject(..)' so a 3D mesh can be added to the view management.
* Small improvements confirmation dialogue
* Notification indicator for task messages
* Redesign orb face ('eyes', 'mouth')

3/11/23: 
* Improvement confirmation dialogue (top of FOV, instead of the bottom, added audio feedback and touch input)
* Added view management (for orb (controllable), tasklist, confirmation dialogue and hands (all non-controllables). The objective of view management is to avoid decreasing the legibility of virtual or real objects in the scene. Controllable will move away from non-controllable obejcts  (e.g., the orb should not overlap with hands if the user is busy working on a task)
* Code documentation 
* Minor improvements (task list fixed with transparent items, fixed orb message not shown when looking at task list)
* Added 'stop' keyword that immediately stops the audio task instructions
* Audio task instructions can be muted

2/19/23: 
* Fixed issue with task id. If the taskID given in SetTaskID(..) is the same as the current one, the orb will not react anymore.
* Added confirmation dialogue
* Added option to mute text to speech for task instructions

10/30/22: 
* Added dragging signifier to the task list
* Added Skip notification (message + warning sound)
* Added textToSpeech option
* fixes with eye collisions with spatial mesh
* fixes with task progress and 'all done' handling
* fixed 'jitter' of orb in some situations by adding lazy orb reactions in the following solver
* Added halo, so user can find task list more easily
* Disabled auto repositioning of task list (but allowing manual adjustments)

10/20/22: 
* Adding direct manipulation to task list
* If the tasklistID is greater than the number if tasks, the user sees the message "All done". Alternatively, the recipe can be set as 
  done by calling AngelARUI.Instance.SetAllTasksDone();
* The orb moves out of the way if the user reads the task list
* The orb shows an indicator on the progress of the recipe
* The tasklist button is moved to the top 
* Delayed task message activation to avoid accidental trigger
* Both the task list and the orb can be moved using far (with ray) and near interactions with either hand
