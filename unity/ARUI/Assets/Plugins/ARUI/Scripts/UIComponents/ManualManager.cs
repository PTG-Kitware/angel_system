//using System;
//using System.Collections.Generic;
//using UnityEngine;

//public class ManualManager : Singleton<ManualManager>
//{
//    private bool _manualInitialized = false;

//    private bool _menuActive = false;
//    public bool MenuActive
//    {
//        get => _menuActive;
//    }

//    private Dictionary<string, DwellButton> allTaskBtns;
//    private GameObject _okayButton;

//    // Start is called before the first frame update
//    private void Start()
//    {
//        allTaskBtns = new Dictionary<string, DwellButton>();
//        for (int i = 0; i<5; i++)
//        {
//            DwellButton btn = transform.GetChild(i).gameObject.AddComponent<DwellButton>();
//            btn.InitializeButton(EyeTarget.menuBtn, () => Debug.Log("MenuBtn pressed"), null, false, DwellButtonType.Toggle, true);
//            allTaskBtns.Add(btn.gameObject.name.Substring(0, btn.gameObject.name.LastIndexOf('_')), btn);
//        }

//        DwellButton okayBtn = transform.GetChild(5).gameObject.AddComponent<DwellButton>();
//        _okayButton = okayBtn.gameObject;
//        okayBtn.InitializeButton(EyeTarget.menuBtn, () => SubmitTaskSelection(), null, true, DwellButtonType.Select, true);

//        _okayButton.gameObject.SetActive(false);
//        foreach (DwellButton btn in allTaskBtns.Values)
//        {
//            btn.Toggled = false;
//            btn.gameObject.SetActive(false);
//        }
//    }

//    /// <summary>
//    /// Triggered if user dwells on the okay button.
//    /// </summary>
//    private void SubmitTaskSelection()
//    {
//        List<string> allToggled = new List<string>();
//        foreach (DwellButton btn in allTaskBtns.Values)
//        {
//            if (btn.Toggled)
//                allToggled.Add(btn.gameObject.name.Substring(0, btn.gameObject.name.LastIndexOf('_')));
//        }

//        if (allToggled.Count > 0)
//        {
//            DataProvider.Instance.SetSelectedTasksFromManual(allToggled);
//            _menuActive = false;

//            for (int i = 0; i < 6; i++)
//                transform.GetChild(i).gameObject.SetActive(false);
//        }
//        else
//            Debug.Log("Nothing selected");
//    }

//    private void Update()
//    {
//        if (!_menuActive || !_manualInitialized) return;

//        UpdateOkayBtnVisibility();
//    }

//    /// <summary>
//    /// Show okay button if at least one menu btn is toggled, else do not show. 
//    /// </summary>
//    private void UpdateOkayBtnVisibility()
//    {
//        int toggledCount = 0;
//        foreach (DwellButton btn in allTaskBtns.Values)
//        {
//            if (btn.Toggled)
//                toggledCount++;
//        }

//        if (toggledCount == 0 && _okayButton.gameObject.activeSelf)
//            _okayButton.gameObject.SetActive(false);

//        else if (toggledCount > 0 && !_okayButton.gameObject.activeSelf)
//            _okayButton.gameObject.SetActive(true);
//    }

//    #region Getter and Setter

//    /// <summary>
//    /// Initialize the manual menu by providing a list of strings that represent the taskIDs
//    /// Can only be called once, nothing happens if manual is already set.
//    /// </summary>
//    /// <param name="manual"></param>
//    public void SetManual(List<string> manual)
//    {
//        if (_manualInitialized) return;

//        _manualInitialized = true;

//        foreach (string btnNames in allTaskBtns.Keys)
//            allTaskBtns[btnNames].IsDisabled = !manual.Contains(btnNames);

//        DataProvider.Instance.InitManual(manual);
//    }

//    /// <summary>
//    /// Enable or disable the manual menu.
//    /// Nothing happens if the manual is not initialized.
//    /// </summary>
//    /// <param name="isActive"></param>
//    public void SetMenuActive(bool isActive)
//    {
//        if (!_manualInitialized) return;

//        _menuActive = isActive;
//        _okayButton.gameObject.SetActive(isActive);

//        foreach (DwellButton btn in allTaskBtns.Values)
//        {
//            if (!btn.IsDisabled)
//            {
//                string taskID = btn.gameObject.name.Substring(0, btn.gameObject.name.LastIndexOf('_'));
//                btn.Toggled = DataProvider.Instance.CurrentSelectedTasks.ContainsKey(taskID) && isActive;
//                btn.gameObject.SetActive(isActive);
//            }
//        }
//    }

//    #endregion
//}
