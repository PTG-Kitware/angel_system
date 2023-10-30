using Shapes;
using System.Globalization;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class ManageStepFlashcardSolo : MonoBehaviour
{
    private TextMeshProUGUI _taskText;
    public TextMeshProUGUI TaskText
    {
        get { return _taskText; }   
    }

    private Step _currStep;
    public Rectangle _backgroundGrid = null;
    private RectTransform _rect;
    private Image _icon;

    public void InitializeComponents()
    {
        _rect = GetComponent<RectTransform>();   
        _icon = gameObject.GetComponentInChildren<Image>();
        _taskText = gameObject.GetComponent<TextMeshProUGUI>();

        _backgroundGrid = GetComponentInChildren<Rectangle>();
    }

    public void Update()
    {
        if (_taskText == null)
            InitializeComponents();

        if (_backgroundGrid)
        {
            _backgroundGrid.gameObject.SetActive(_icon.gameObject.activeSelf);
            _backgroundGrid.Height = _rect.rect.height;
        }
    }

    /// <summary>
    /// Set the step message of the task to the given task
    /// </summary>
    /// <param name="newStep"></param>
    public void SetFlashcard(Step newStep)
    {
        if (_taskText == null)
            InitializeComponents();

        _currStep = newStep;
        _taskText.SetText(_currStep.StepDesc);

        foreach (string item in _currStep.RequiredItems)
            UnderlineTasksObjects(item);

        if (_icon != null)
            _icon.gameObject.SetActive(true);
    }

    /// <summary>
    /// Set the step message of the task to the given task + progress indicator + task name
    /// </summary>
    /// <param name="newStep"></param>
    /// <param name="current"></param>
    /// <param name="allSteps"></param>
    public void SetFlashcard(Step newStep, int current, int allSteps)
    {
        if (_taskText == null)
            InitializeComponents();

        _currStep = newStep;
        _taskText.SetText("("+ current + "/"+ allSteps+") : "+_currStep.StepDesc);

        foreach(string item in _currStep.RequiredItems) {
            UnderlineTasksObjects(item);
        }

        if (_icon != null)
            _icon.gameObject.SetActive(true);
    }

    /// <summary>
    /// Set the task as done by changing the text to the given message and blending out the status icon
    /// </summary>
    /// <param name="message"></param>
    public void SetAsDone(string message)
    {
        if (_taskText == null)
            InitializeComponents();

        _taskText.SetText(message);

        if (_icon != null)
            _icon.gameObject.SetActive(false);
    }

    /// <summary>
    /// Parse through the text and underline task objects.
    /// </summary>
    /// <param name="substring"></param>
    private void UnderlineTasksObjects(string substring)
    {
        string currTextLower = TaskText.text.ToLower(new CultureInfo("en-US", false));
        string currText = TaskText.text;
        int index = currTextLower.IndexOf(substring);
        if (index >= 0)
        {
            currText = currText.Insert(index, "<u><b>");
            currText = currText.Insert(index + substring.Length + 6, "</u></b>");
        }
        TaskText.SetText(currText);
    }
}
