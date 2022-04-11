using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.NetworkInformation;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

/// <summary>
/// Represents the current state of a task.
/// </summary>
public class TaskUpdateMessage
{
    public string _taskName;
    public uint _numSteps;
    public List<string> _steps;
    public string _currStep;
    public string _prevStep;
    public string _currActivity;
    public string _nextActivity;
    public bool _updated = true;

    public TaskUpdateMessage(string taskName, uint numSteps, List<string> steps, string currStep,
                             string prevStep, string currActivity, string nextActivity)
    {
        _taskName = taskName;
        _numSteps = numSteps;
        _steps = steps;
        _currStep = currStep;
        _prevStep = prevStep;
        _currActivity = currActivity;
        _nextActivity = nextActivity;
    }
}


/// <summary>
/// Class responsible for keeping track of the user's task progress and updating
/// the task AR display.
/// </summary>
public class TaskManager : MonoBehaviour
{
    private System.Net.Sockets.TcpClient _tcpClient;
    private System.Net.Sockets.TcpListener _tcpServer;
    private NetworkStream _tcpStream;
    private string _TcpServerIPAddr = "";
    private const int _TaskUpdateTcpPort = 11011;

    private Logger _logger = null;
    private TaskLogger _taskLogger = null;

    private string _debugString = "";

    private TaskUpdateMessage _taskUpdateMessage;

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

    /// <summary>
    /// Lazy acquire the task logger object and return the reference to it.
    /// </summary>
    /// <returns>Logger instance reference.</returns>
    private ref TaskLogger taskLogger()
    {
        if (this._taskLogger == null)
        {
            this._taskLogger = GameObject.Find("TaskLogger").GetComponent<TaskLogger>();
        }
        return ref this._taskLogger;
    }

    // Start is called before the first frame update
    protected void Start()
    {
        Logger log = logger();
        TaskLogger taskLog = taskLogger();

        _taskUpdateMessage = new TaskUpdateMessage("Waiting for task", 0, new List<string>(),
                                                   "N/A", "N/A", "N/A", "N/A");

        log.LogInfo(_taskUpdateMessage._taskName);
        taskLog.UpdateTaskDisplay(_taskUpdateMessage);

        try
        {
            _TcpServerIPAddr = PTGUtilities.getIPv4AddressString();
        }
        catch (InvalidIPConfiguration e)
        {
            log.LogInfo(e.ToString());
            return;
        }

        Thread t = new Thread(SetupTaskManagerServer);
        t.Start();
    }

    // Update is called once per frame
    void Update()
    {
        if (_debugString != "")
        {
            this.logger().LogInfo(_debugString);
            _debugString = "";
        }

        if (_taskUpdateMessage._updated)
        {
            this.taskLogger().UpdateTaskDisplay(_taskUpdateMessage);
            _taskUpdateMessage._updated = false;
        }
    }

    /// <summary>
    /// Starts the TCP server for this object, waits for a client to connect, and then
    /// starts listening for task updates.
    /// </summary>
    void SetupTaskManagerServer()
    {
        IPAddress localAddr = IPAddress.Parse(_TcpServerIPAddr);

        _tcpServer = new TcpListener(localAddr, _TaskUpdateTcpPort);

        // Start listening for client requests.
        _tcpServer.Start();

        // Perform a blocking call to accept requests.
        _tcpClient = _tcpServer.AcceptTcpClient();
        _tcpStream = _tcpClient.GetStream();

        ListenForTaskUpdates();
    }


    /// <summary>
    /// Continuously reads task update messages from the TCP socket.
    /// Updates the task logger display when a new message is received.
    /// </summary>
    void ListenForTaskUpdates()
    {
        while (true)
        {
            // Check if there is data to read and read it if there is
            bool dataAvailable = false;
            if (_tcpStream != null)
            {
                dataAvailable = _tcpStream.DataAvailable;
            }

            if (dataAvailable)
            {
                byte[] readBuffer = new byte[1024];
                int bytesRead = 0;

                do
                {
                    try
                    {
                        bytesRead = _tcpStream.Read(readBuffer, 0, readBuffer.Length);
                        dataAvailable = _tcpStream.DataAvailable;
                    }
                    catch (Exception e)
                    {
                        bytesRead = 0;
                        break;
                    }
                }
                while (dataAvailable);

                int bufferIndex = 0;
                while (bufferIndex != bytesRead)
                {
                    // PTG header:
                    //   -- 32-bit sync = 4 bytes
                    //   -- 32-bit ros msg length = 4 bytes
                    // ROS2 message:
                    //  header
                    //   -- 32 bit seconds = 4 bytes
                    //   -- 32 bit nanoseconds = 4 bytes
                    //   -- frame id string
                    //  task_name string
                    //  num_steps = 4 bytes
                    //  steps string list
                    //  current_step string
                    //  previous_step string
                    //  current_activity string
                    //  next_activity string

                    // verify sync
                    byte[] syncBytes = new byte[4];
                    Array.Copy(readBuffer, bufferIndex, syncBytes, 0, 4);
                    uint sync = System.BitConverter.ToUInt32(syncBytes, 0);
                    if (sync != 0x1ACFFC1D)
                    {
                        _debugString += "Invalid sync! Exiting...";
                        break;
                    }
                    bufferIndex += 4;

                    // get message length
                    byte[] lengthBytes = new byte[4];
                    Array.Copy(readBuffer, bufferIndex, syncBytes, 0, 4);
                    uint length = System.BitConverter.ToUInt32(syncBytes, 0);
                    //_debugString += "message length = " + length.ToString();
                    bufferIndex += 4;

                    // skip detection stamp time
                    bufferIndex += 8;

                    // skip frame ID string
                    int nullIndex = GetNullCharIndex(readBuffer, bufferIndex, bytesRead);
                    int sLen = nullIndex - bufferIndex;

                    bufferIndex = nullIndex + 1;

                    // get task_name string
                    nullIndex = GetNullCharIndex(readBuffer, bufferIndex, bytesRead);
                    sLen = nullIndex - bufferIndex;
                    string taskName = System.Text.Encoding.UTF8.GetString(readBuffer, bufferIndex, sLen);
                    //_debugString += "task name = " + taskName;
                    bufferIndex = nullIndex + 1;

                    _taskUpdateMessage._taskName = taskName;

                    // get number of steps
                    byte[] numStepsBytes = new byte[4];
                    Array.Copy(readBuffer, bufferIndex, numStepsBytes, 0, 4);
                    uint numSteps = System.BitConverter.ToUInt32(numStepsBytes, 0);
                    //_debugString += "num steps = " + numSteps.ToString();
                    bufferIndex += 4;

                    _taskUpdateMessage._numSteps = numSteps;

                    // get steps
                    List<string> steps = new List<string>();
                    for (int i = 0; i < numSteps; i++)
                    {
                        nullIndex = GetNullCharIndex(readBuffer, bufferIndex, bytesRead);

                        sLen = nullIndex - bufferIndex;
                        string step = System.Text.Encoding.UTF8.GetString(readBuffer, bufferIndex, sLen);
                        //_debugString += "step = " + step + "\n";
                        steps.Add(step);

                        bufferIndex = nullIndex + 1;
                    }
                    _taskUpdateMessage._steps = steps;

                    // get current step
                    nullIndex = GetNullCharIndex(readBuffer, bufferIndex, bytesRead);
                    sLen = nullIndex - bufferIndex;
                    string currStep = System.Text.Encoding.UTF8.GetString(readBuffer, bufferIndex, sLen);
                    //_debugString += "current step = " + currStep;
                    bufferIndex = nullIndex + 1;
                    _taskUpdateMessage._currStep = currStep;

                    // get previous step
                    nullIndex = GetNullCharIndex(readBuffer, bufferIndex, bytesRead);
                    sLen = nullIndex - bufferIndex;
                    string prevStep = System.Text.Encoding.UTF8.GetString(readBuffer, bufferIndex, sLen);
                    //_debugString += "previous step = " + prevStep;
                    bufferIndex = nullIndex + 1;
                    _taskUpdateMessage._prevStep = prevStep;

                    // get current activity
                    nullIndex = GetNullCharIndex(readBuffer, bufferIndex, bytesRead);
                    sLen = nullIndex - bufferIndex;
                    string currActivity = System.Text.Encoding.UTF8.GetString(readBuffer, bufferIndex, sLen);
                    //_debugString += "current activity = " + currActivity;
                    bufferIndex = nullIndex + 1;
                    _taskUpdateMessage._currActivity = currActivity;

                    // get next activity
                    nullIndex = GetNullCharIndex(readBuffer, bufferIndex, bytesRead);
                    sLen = nullIndex - bufferIndex;
                    string nextActivity = System.Text.Encoding.UTF8.GetString(readBuffer, bufferIndex, sLen);
                    //_debugString += "next activity = " + nextActivity;
                    bufferIndex = nullIndex + 1;
                    _taskUpdateMessage._nextActivity = nextActivity;

                    // signal to update the task logger display
                    _taskUpdateMessage._updated = true;
                }

            }
        }
    }


    /// <summary>
    /// Returns the index of the first null character (0) in the given byte array.
    /// </summary>
    /// <returns>null char index.</returns>
    private static int GetNullCharIndex(byte[] array, int index, int length)
    {
        int nullIndex = -1;
        for (int k = index; k < length; k++)
        {
            if (array[k] == 0)
            {
                nullIndex = k;
                break;
            }
        }

        return nullIndex;
    }

}
