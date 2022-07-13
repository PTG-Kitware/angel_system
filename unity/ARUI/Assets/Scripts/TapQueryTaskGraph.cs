using DilmerGames.Core.Singletons;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.UI;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Angel;


public class TapQueryTaskGraph : MonoBehaviour, IMixedRealityInputActionHandler
{
    private Logger _logger = null;

    ROSConnection ros;
    public string querytaskgraphTopicName = "query_task_graph";

    private Timer timerTest;

    private bool actionInProgress = false;

    private int currentTask = 0;

    private bool taskGraphInitialized = false;
    private int loopIdx = 0;

    /// <summary>
    /// Lazy acquire the logger object and return the reference to it.
    /// </summary>
    /// <returns>Logger instance reference.</returns>
    private ref Logger logger()
    {
        if (this._logger == null)
        {
            // TODO: Error handling for null loggerObject?
            this._logger = GameObject.Find("Logger").GetComponent<Logger>();
        }
        return ref this._logger;
    }

    public void Start()
    {
        // Create the QueryTaskGraph subscriber
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterRosService<QueryTaskGraphRequest, QueryTaskGraphResponse>(querytaskgraphTopicName);
    }


    /// <summary>
    /// Listen to Keyevents for debugging (only in the editor)
    /// </summary>
    public void Update()
    {
        Logger log = logger();
        loopIdx++;
        if (Input.GetKeyUp(KeyCode.RightArrow))
        {
            currentTask++;
            AngelARUI.Instance.SetCurrentTaskID(currentTask);
        }
        else if (Input.GetKeyUp(KeyCode.LeftArrow))
        {
            currentTask--;
            AngelARUI.Instance.SetCurrentTaskID(currentTask);
        }

        if (Input.GetKeyUp(KeyCode.Space))
        {
            AngelARUI.Instance.ToggleTasklist();
        }

        // Check for a task graph every 5 seconds
        // TODO: probably a better way to do this
        if (taskGraphInitialized == false && (loopIdx % 300 == 0))
        {
            // Send message to ROS
            QueryTaskGraphRequest queryTaskGraphRequest = new QueryTaskGraphRequest();
            ros.SendServiceMessage<QueryTaskGraphResponse>(querytaskgraphTopicName, queryTaskGraphRequest, QueryTaskGraphCallback);
        }

    }

    void QueryTaskGraphCallback(QueryTaskGraphResponse msg)
    {
        Logger log = logger();

        List<uint> edges = new List<uint>();
        edges.AddRange(msg.task_graph.node_edges);
        List<List<string>> tasks = new List<List<string>>();

        int taskIdx = 0;
        uint curTask = edges[taskIdx];

        List<string> elem = new List<string> { "0", msg.task_graph.task_nodes[curTask].name };
        tasks.Add(elem);

        uint nextTask = edges[taskIdx + 1];

        // remove pair form list
        edges.RemoveAt(taskIdx + 1);
        edges.RemoveAt(taskIdx);

        int num_pairs = edges.Count / 2;
        for (int i = 0; i < num_pairs; i++)
        {
            curTask = nextTask;
            taskIdx = edges.IndexOf(curTask);

            elem = new List<string> { "0", msg.task_graph.task_nodes[curTask].name };
            tasks.Add(elem);

            nextTask = edges[taskIdx + 1];

            // remove pair form list
            edges.RemoveAt(taskIdx + 1);
            edges.RemoveAt(taskIdx);
        }

        // Add the last task
        elem = new List<string> { "0", msg.task_graph.task_nodes[nextTask].name };
        tasks.Add(elem);

        // Send tasks to ARUI
        string[,] final_tasks = new string[tasks.Count, 2];
        for (int i = 0; i < tasks.Count; i++)
        {
            final_tasks[i, 0] = tasks[i][0];
            final_tasks[i, 1] = tasks[i][1];
        }

        AngelARUI.Instance.SetTasks(final_tasks);

        taskGraphInitialized = true;
    }

    private IEnumerator AddIfHit(BaseInputEventData eventData)
    {
        var result = eventData.InputSource.Pointers[0].Result;
        if (result != null)
        {
            var hitPosition = result.Details.Point;

            if (result.CurrentPointerTarget?.layer == 31)
            {
                AngelARUI.Instance.UpdateDatabase("id", UpdateType.add, hitPosition, Utils.ConvertClassNumToStr((uint)UnityEngine.Random.Range(0, 99)));
            }
        }
        currentTask++;
        AngelARUI.Instance.SetCurrentTaskID(currentTask);

        yield return new WaitForSeconds(1f);
        actionInProgress = false;
    }

    #region tap input registration
    private void OnEnable()
    {
        CoreServices.InputSystem?.RegisterHandler<IMixedRealityInputActionHandler>(this);
        AngelARUI.Instance.PringDebugMessage("Generate Test Data using Tap gesture", true);
    }


    private void OnDisable()
    {
        CoreServices.InputSystem?.UnregisterHandler<IMixedRealityInputActionHandler>(this);
    }

    public void OnActionEnded(BaseInputEventData eventData)
    {
        if (eventData != null && eventData.InputSource.SourceType.Equals(InputSourceType.Hand) && !actionInProgress)
        {
            actionInProgress = true;
            StartCoroutine(AddIfHit(eventData));
        }
    }

    public void OnActionStarted(BaseInputEventData eventData) { }


    #endregion

}
