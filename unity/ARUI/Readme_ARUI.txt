How to use ARUI in your scene:

* Create empty GameObject in the highest hierarchy level
* Add the AngelARUI script to it. (The script generates all necessary UI elements at run time)

**** Examples

*** Set task graph:
For now the ARUI supports a two level task graph:
To set the task graph call:

AngelARUI.Instance.SetTasks(tasks);

where 'tasks' could look like this:
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

The first column indicates if the row is a maintask (0) or subtask (1) of the last main task.
The second column provides the text of the task.

To set the current task the user has to do, call:
    AngelARUI.Instance.SetCurrentTaskID(index);

The integer value 'index' presents the row index in the 'tasks' array. (eg. index 4 would be {"1", "Text example Subtask 2 of MainTask 1"})
If the index does not match with the given task graph, the function call is ignored.

Examples can be found in "TapTestData.cs".

The tasklist can be toggled using AngelARUI.Instance.ToggleTasklist();

**** MRTK settings
For eye-tracking to work, the user has to give permission to the eye-tracking. Also, eye-tracking has to be enabled in the MRTK toolkit.
If eye tracking does not work through the spatial map created by Hololens:
Goto: 

Local InputSystemProfile -> Pointers -> Pointing Raycast Layer Masks -> Disable every option, except the layer "UI"
In the AngelMRTK profiles, the pointers (hand, eye, rays) collide with all objects marked as layer "UI"
In the hierarchy, at the main camera, search for the "GazeProvider" script and select Raycast Layer Masks -> "UI"

**** UI Functions
The UI uses eye gaze as input. The user can enable and disable the task list by looking at the button next to the white orb.
The position of the orb and the tasklist can be adjusted using the tap gesture or the raycast (far)

**** Debugging
If an instance of the Logger is in the scene, the ARUI prints debug message to the Unity console and the logger window.
One can disable the ARUI debug messages in the unity hierarchy by setting the "ShowARUIDebugMessages" to false.

**** Limitations
- The eye tracking might not be reliable if the user wears glasses.
- At start-up, it might take few seconds until the eye gaze rays is reliable
- If it is recognized by the system that a new user uses the application, the eye tracking calibration might start. This is good, since
  eye tracking is not reliable if not correctly calibrated to the current user.


**** Changelog
10/20/22: 
* Adding direct manipulation to task list
* If the tasklistID is greater than the number if tasks, the user sees a message "All done". Alternatively, the recipe can be set as 
  done by calling AngelARUI.Instance.SetAllTasksDone();
* The orb moves out of the way if the user reads the task list
* The orb shows an inidicator on the progress of the recipe
* the tasklist button is moved to the top 
* Delayed task message activation to avoid accidental trigger
* Both, the task list and the orb can be moved using far (with ray) and near interactions with either hand


