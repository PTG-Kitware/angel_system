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

To set the current task the use has to do, call:
AngelARUI.Instance.SetCurrentTaskID(index);

the integer value 'index' presents the row index in the 'tasks' array. (eg. index 4 would be {"1", "Text example Subtask 2 of MainTask 1"})
if the index does not match with the given task graph, the function call is ignored.

Examples can be found in "TapTestData.cs"

**** MRTK settings
For eye-tracking to work, the user has to give permission to the eye-tracking. Also, eye-tracking has to be enabled in the MRTK toolkit.
If eye tracking does not work through the spatial map created by Hololens:
Goto: Local InputSystemProfile -> Pointers -> Pointing Raycast Layer Masks -> Disable every option, except the layer "UI"

**** UI Functions
The UI uses eye-gaze as input. The user can enable and disable the task list by looking at the button next to the white orb.

**** Debugging
If an instance of the Logger is in the scene, the ARUI prints debug message to the Unity console and the logger window.
One can disable the ARUI debug messages in the unity hierarchy by setting the "ShowARUIDebugMessages" to false.

**** Limitations
The eye tracking might not be reliable if the user wears glasses.
