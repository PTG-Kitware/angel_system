using UnityEngine;

public enum ListPosition
{
    Top = 0,
    Bottom = 1,
    Middle =2,
}

public class TaskListElement : MonoBehaviour
{
    public int id;
    private bool isDone = false;

    private Color inactiveColor = Color.gray;
    private Color activeColor = Color.white;
    private Color doneColor = new Color(0.30f, 0.60f, 0.30f);

    private TMPro.TextMeshProUGUI textCanvas;
    private Shapes.Rectangle checkBox;
    private Shapes.Cone checkBoxCurrent;
    private Shapes.Line subTaskIndicator;

    private int taskLevel = 0;

    private float currentAlpha = 1f;
    // Left, Top, Right, Bottom
    private Vector4 prefabMargin;
    private Vector4 subTaskMargin = new Vector4(0.01f, 0, 0, 0);
    private float topBottomMargin = 0.03f;

    private string postMessage = "";
    private string taskMessage = "";

    /// <summary>
    /// Get all reference from the task list element prefab
    /// </summary>
    private void InitIfNeeded()
    {
        if (checkBox == null)
        {
            checkBox = GetComponentInChildren<Shapes.Rectangle>();
            if (checkBox == null) Debug.Log("Script could not be found: Shapes.Rectangle at " + gameObject.name);
            
            checkBoxCurrent = GetComponentInChildren<Shapes.Cone>();
            if (checkBoxCurrent == null) Debug.Log("Script could not be found: Shapes.Cone at " + gameObject.name);

            subTaskIndicator = transform.GetComponentInChildren<Shapes.Line>();
            if (subTaskIndicator == null) Debug.Log("Script could not be found: Shapes.Line at " + gameObject.name);

            textCanvas = GetComponent<TMPro.TextMeshProUGUI>();
            if (textCanvas == null) Debug.Log("Script could not be found: TMPro.TextMeshProUGUI at " + gameObject.name);

            prefabMargin = new Vector4(textCanvas.margin.x, textCanvas.margin.y, textCanvas.margin.z, textCanvas.margin.w);
        }
    }

    /// <summary>
    /// Reset all values
    /// </summary>
    public void Reset(bool visible)
    {
        if (visible)
            currentAlpha = 1f;
        textCanvas.text = taskMessage;
        checkBox.gameObject.SetActive(false);
        checkBoxCurrent.gameObject.SetActive(false);
        UpdateColor(inactiveColor);
    }

    /// <summary>
    /// Set text, ID and level in task hierarchy (0 for main task, 1 for subtask)
    /// </summary>
    /// <param name="taskID">id in the task list (starts with 0)</param>
    /// <param name="text">text of task message</param>
    /// <param name="taskLevel">0 or 1</param>
    public void InitText(int taskID, string text, int taskLevel)
    {
        InitIfNeeded();

        textCanvas.text = text;
        this.taskLevel = taskLevel;
        taskMessage = text;
        id = taskID;
        
        checkBox.gameObject.SetActive(false);
        checkBoxCurrent.gameObject.SetActive(false);

        UpdateColor(inactiveColor);

        if (taskLevel == 0)
        {
            textCanvas.margin = prefabMargin;
            subTaskIndicator.gameObject.SetActive(false);
            //textCanvas.fontStyle = TMPro.FontStyles.UpperCase;
        }
        else
        {
            textCanvas.margin = prefabMargin + subTaskMargin;
            subTaskIndicator.gameObject.SetActive(false);
        }
    }
    
    /// <summary>
    /// Set the this task as done
    /// </summary>
    /// <param name="isDone"></param>
    public void SetIsDone(bool isDone)
    {
        InitIfNeeded();

        checkBox.gameObject.SetActive(true);
        checkBoxCurrent.gameObject.SetActive(false);

        this.isDone = isDone;

        //define color and alpha of element based on user attention and task state
        if (isDone)
        {
            UpdateColor(doneColor);
            checkBox.Type = Shapes.Rectangle.RectangleType.HardSolid;
        }
        else
        {
            UpdateColor(inactiveColor);
            checkBox.Type = Shapes.Rectangle.RectangleType.HardBorder;
        }
            
        this.postMessage = "";
        if (taskLevel == 0)
            textCanvas.text = taskMessage;

    }

    /// <summary>
    /// Set this task as the one the user has to do
    /// </summary>
    /// <param name="postMessage"></param>
    public void SetAsCurrent(string postMessage)
    {
        InitIfNeeded();

        checkBox.gameObject.SetActive(false);
        checkBoxCurrent.gameObject.SetActive(true);

        UpdateColor(activeColor);

        this.postMessage = postMessage;
        if (taskLevel==0 && postMessage.Length>0)
            textCanvas.text = taskMessage + " - " +postMessage;

    }

    /// <summary>
    /// Update the color of the task message and icon depending on it's state
    /// </summary>
    /// <param name="newColor"></param>
    private void UpdateColor(Color newColor)
    {
        textCanvas.color = new Color(newColor.r, newColor.g, newColor.b, currentAlpha);
        checkBoxCurrent.Color = new Color(newColor.r, newColor.g, newColor.b, currentAlpha);
        checkBox.Color = new Color(newColor.r, newColor.g, newColor.b, currentAlpha);
        subTaskIndicator.Color = new Color(newColor.r, newColor.g, newColor.b, currentAlpha);
    }

    /// <summary>
    ///  Set the alpha value of the task text, used for fading
    /// </summary>
    /// <param name="alpha"></param>
    public void SetAlpha(float alpha)
    {
        textCanvas.color = new Color(textCanvas.color.r, textCanvas.color.g, textCanvas.color.b, alpha);
        checkBoxCurrent.Color = new Color(checkBoxCurrent.Color.r, checkBoxCurrent.Color.g, checkBoxCurrent.Color.b, alpha);
        subTaskIndicator.Color = new Color(subTaskIndicator.Color.r, subTaskIndicator.Color.g, subTaskIndicator.Color.b, alpha);
        checkBox.Color = new Color(checkBox.Color.r, checkBox.Color.g, checkBox.Color.b, alpha);

        currentAlpha = alpha;
    }


}
