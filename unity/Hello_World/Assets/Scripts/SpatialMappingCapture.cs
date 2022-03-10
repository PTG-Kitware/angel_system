using Microsoft.MixedReality.SceneUnderstanding;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.SpatialAwareness;
using Microsoft.MixedReality.Toolkit.Examples.Demos;
using Microsoft.MixedReality.Toolkit.Experimental.SpatialAwareness;
using Microsoft.MixedReality.Toolkit.UI;
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


public class DetectionSet
{
    public List<Vector3> _leftPoints;
    public List<Vector3> _topPoints;
    public List<Vector3> _rightPoints;
    public List<Vector3> _bottomPoints;
    public List<string> _labels;
    public bool _drawn;

    public DetectionSet(List<Vector3> leftPoints, List<Vector3> topPoints,
                        List<Vector3> rightPoints, List<Vector3> bottomPoints, List<string> labels)
    {
        _leftPoints = leftPoints;
        _topPoints = topPoints;
        _rightPoints = rightPoints;
        _bottomPoints = bottomPoints;
        _labels = labels;
        _drawn = false;
    }

}

public class SpatialMappingCapture : MonoBehaviour, IMixedRealitySpatialAwarenessObservationHandler<SpatialAwarenessMeshObject>
{
    private System.Net.Sockets.TcpClient _tcpClient;
    private System.Net.Sockets.TcpListener _tcpServer;
    private NetworkStream _tcpStream;
    private string _TcpServerIPAddr = "";
    private const int _SMTcpPort = 11010;

    private Logger _logger = null;
    private string _debugString = "";
    private bool _observersResumed = false;

    private List<DetectionSet> _latestDetections = new List<DetectionSet>();
    private List<GameObject> _latestDetectionObjects = new List<GameObject>();

    private static Mutex _mut = new Mutex();

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

    // Start is called before the first frame update
    protected void Start()
    {
        Logger log = logger();

        // Suspend Mesh Observation from all Observers until we connect to the TCP socket
        CoreServices.SpatialAwarenessSystem.SuspendObservers();

        try
        {
            _TcpServerIPAddr = PTGUtilities.getIPv4AddressString();
        }
        catch (InvalidIPConfiguration e)
        {
            log.LogInfo(e.ToString());
            return;
        }

        Thread tSpatialMappingCapture = new Thread(SetupSpatialMappingCapture);
        tSpatialMappingCapture.Start();
    }

    // Update is called once per frame
    void Update()
    {
        if (_debugString != "")
        {
            this.logger().LogInfo(_debugString);
            _debugString = "";
        }

        if (_tcpStream != null && (_observersResumed == false))
        {
            CoreServices.SpatialAwarenessSystem.ResumeObservers();
            _observersResumed = true;
        }

        _mut.WaitOne();
        for (int i = 0; i < _latestDetections.Count; i++)
        {
            if (_latestDetections[i]._drawn != true)
            {
                // Have a new detection, so delete the old game objects
                for (int j = 0; j < _latestDetectionObjects.Count; j++)
                {
                    Destroy(_latestDetectionObjects[j]);
                }
                _latestDetectionObjects.Clear();

                for (int j = 0; j < _latestDetections[i]._labels.Count; j++)
                {
                    DrawBox(_latestDetections[i]._leftPoints[j], _latestDetections[i]._topPoints[j],
                            _latestDetections[i]._rightPoints[j], _latestDetections[i]._bottomPoints[j],
                            _latestDetections[i]._labels[j]);

                }
                _latestDetections[i]._drawn = true;
            }
        }
        _mut.ReleaseMutex();
    }

    void SetupSpatialMappingCapture()
    {
        IPAddress localAddr = IPAddress.Parse(_TcpServerIPAddr);

        _tcpServer = new TcpListener(localAddr, _SMTcpPort);

        // Start listening for client requests.
        _tcpServer.Start();

        // Perform a blocking call to accept requests.
        // You could also use server.AcceptSocket() here.
        _tcpClient = _tcpServer.AcceptTcpClient();
        _tcpStream = _tcpClient.GetStream();

        ListenForDetections();
    }

    private void ListenForDetections()
    {
        while (true)
        {
            // Check if there is data to read and read it if there is
            if (_tcpStream.DataAvailable)
            {
                byte[] readBuffer = new byte[1024];
                int bytesRead = 0;

                do
                {
                    bytesRead = _tcpStream.Read(readBuffer, 0, readBuffer.Length);
                }
                while (_tcpStream.DataAvailable);

                _debugString += "Bytes read = " + bytesRead.ToString() + "\n";

                _mut.WaitOne();

                // Got a detection so clear out the old detections
                _latestDetections.Clear();

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
                    //  source_stamp
                    //   -- 32 bit seconds
                    //   -- 32 bit nanoseconds
                    //  num_objects
                    //   -- 32 bit num objects
                    //  labels
                    //   -- string * num_objects
                    //  3d points: 12 * 4 * num_objects
                    //   -- left points (12 bytes)
                    //     -- 32 bit float x
                    //     -- 32 bit float y
                    //     -- 32 bit float z
                    //   -- top points (12 bytes)
                    //     -- 32 bit float x
                    //     -- 32 bit float y
                    //     -- 32 bit float z
                    //   -- right points (12 bytes)
                    //     -- 32 bit float x
                    //     -- 32 bit float y
                    //     -- 32 bit float z
                    //   -- bottom points (12 bytes)
                    //     -- 32 bit float x
                    //     -- 32 bit float y
                    //     -- 32 bit float z

                    _debugString += "Buffer index = " + bufferIndex.ToString() + "\n";
                    _debugString += "buffer @ buffer index = " + readBuffer[bufferIndex].ToString() + "\n";

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

                    // NOTE: uncomment if you need access to the other fields like
                    // the image timestamps

                    // get message length
                    byte[] lengthBytes = new byte[4];
                    Array.Copy(readBuffer, bufferIndex, syncBytes, 0, 4);
                    uint length = System.BitConverter.ToUInt32(syncBytes, 0);
                    //_debugString += "message length = " + length.ToString();
                    bufferIndex += 4;

                    // get detection stamp time
                    //byte[] detSecsBytes = new byte[4];
                    //Array.Copy(readBuffer, bufferIndex, detSecsBytes, 0, 4);
                    //uint detSecs = System.BitConverter.ToUInt32(detSecsBytes, 0);
                    //_debugString += "det seconds = " + detSecs.ToString();
                    bufferIndex += 4;

                    //byte[] detNSecsNytes = new byte[4];
                    //Array.Copy(readBuffer, bufferIndex, detNSecsNytes, 0, 4);
                    //uint detNSecs = System.BitConverter.ToUInt32(detNSecsNytes, 0);
                    //_debugString += "det nseconds = " + detNSecs.ToString() + "\n";
                    bufferIndex += 4;

                    // get string
                    int nullIndex = GetNullCharIndex(readBuffer, bufferIndex, bytesRead);
                    //_debugString += "index = " + nullIndex;

                    int sLen = nullIndex - bufferIndex;
                    //string objectId = System.Text.Encoding.UTF8.GetString(readBuffer, 16, sLen);
                    //_debugString += "object id = " + objectId;

                    bufferIndex = nullIndex + 1;

                    // get image stamp time
                    //byte[] imSecsBytes = new byte[4];
                    //Array.Copy(readBuffer, bufferIndex, imSecsBytes, 0, 4);
                    //uint imSecs = System.BitConverter.ToUInt32(imSecsBytes, 0);
                    bufferIndex += 4;

                    //byte[] imNSecsBytes = new byte[4];
                    //Array.Copy(readBuffer, bufferIndex, imNSecsBytes, 0, 4);
                    //uint imNSecs = System.BitConverter.ToUInt32(imNSecsBytes, 0);
                    bufferIndex += 4;

                    // get number of objects in this message
                    byte[] numObjectsBytes = new byte[4];
                    Array.Copy(readBuffer, bufferIndex, numObjectsBytes, 0, 4);
                    uint numObjects = System.BitConverter.ToUInt32(numObjectsBytes, 0);
                    //_debugString += "num objects = " + numObjects.ToString();
                    bufferIndex += 4;

                    // get labels
                    List<string> labels = new List<string>();
                    for (int i = 0; i < numObjects; i++)
                    {
                        nullIndex = GetNullCharIndex(readBuffer, bufferIndex, bytesRead);
                        //_debugString += "index = " + nullIndex;

                        sLen = nullIndex - bufferIndex;
                        string label = System.Text.Encoding.UTF8.GetString(readBuffer, bufferIndex, sLen);
                        //_debugString += "label = " + label + "\n";
                        labels.Add(label);

                        bufferIndex = nullIndex + 1;
                    }

                    // get the 3d points
                    List<Vector3> leftPoints = new List<Vector3>();
                    List<Vector3> topPoints = new List<Vector3>();
                    List<Vector3> rightPoints = new List<Vector3>();
                    List<Vector3> bottomPoints = new List<Vector3>();

                    for (int i = 0; i < 4; i++)
                    {
                        for (int j = 0; j < numObjects; j++)
                        {
                            byte[] pointXBytes = new byte[4];
                            Array.Copy(readBuffer, bufferIndex, pointXBytes, 0, 4);
                            bufferIndex += 4;

                            byte[] pointYBytes = new byte[4];
                            Array.Copy(readBuffer, bufferIndex, pointYBytes, 0, 4);
                            bufferIndex += 4;

                            byte[] pointZBytes = new byte[4];
                            Array.Copy(readBuffer, bufferIndex, pointZBytes, 0, 4);
                            bufferIndex += 4;

                            float pointX = System.BitConverter.ToSingle(pointXBytes, 0);
                            float pointY = System.BitConverter.ToSingle(pointYBytes, 0);
                            float pointZ = System.BitConverter.ToSingle(pointZBytes, 0);
                            Vector3 v = new Vector3(-pointX, pointY, pointZ);

                            if (i == 0)
                            {
                                leftPoints.Add(v);
                            }
                            else if (i == 1)
                            {
                                topPoints.Add(v);
                            }
                            else if (i == 2)
                            {
                                rightPoints.Add(v);
                            }
                            else if (i == 3)
                            {
                                bottomPoints.Add(v);
                            }
                        }
                    }

                    DetectionSet d = new DetectionSet(leftPoints, topPoints, rightPoints, bottomPoints, labels);
                    _latestDetections.Add(d);
                }
                _mut.ReleaseMutex();

            }
        }
    }

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

    private void OnEnable()
    {
        // Register component to listen for Mesh Observation events, typically done in OnEnable()
        CoreServices.SpatialAwarenessSystem.RegisterHandler<IMixedRealitySpatialAwarenessObservationHandler<SpatialAwarenessMeshObject>>(this);
    }

    private void OnDisable()
    {
        // Unregister component from Mesh Observation events, typically done in OnDisable()
        CoreServices.SpatialAwarenessSystem.UnregisterHandler<IMixedRealitySpatialAwarenessObservationHandler<SpatialAwarenessMeshObject>>(this);
    }

    public virtual void OnObservationAdded(MixedRealitySpatialAwarenessEventData<SpatialAwarenessMeshObject> eventData)
    {
        // Send the observation via the TCP socket
        if (_tcpStream != null)
        {
            // Copy over the mesh data we need because we cannot access it outside of the main thread
            Vector3[] vertices = eventData.SpatialObject.Filter.mesh.vertices;
            int[] triangles = eventData.SpatialObject.Filter.mesh.triangles;
            int id = eventData.SpatialObject.Id;
            Thread tSendObject = new Thread(() => SendMeshObjectAddition(vertices,
                                                                         triangles,
                                                                         id));
            tSendObject.Start();
        }
    }

    public virtual void OnObservationUpdated(MixedRealitySpatialAwarenessEventData<SpatialAwarenessMeshObject> eventData)
    {
        // Copy over the mesh data we need because we cannot access it outside of the main thread
        Vector3[] vertices = eventData.SpatialObject.Filter.mesh.vertices;
        int[] triangles = eventData.SpatialObject.Filter.mesh.triangles;
        int id = eventData.SpatialObject.Id;
        Thread tSendObject = new Thread(() => SendMeshObjectAddition(vertices,
                                                                     triangles,
                                                                     id));
        tSendObject.Start();
    }

    public virtual void OnObservationRemoved(MixedRealitySpatialAwarenessEventData<SpatialAwarenessMeshObject> eventData)
    {
        int id = eventData.SpatialObject.Id;
        Thread tSendObject = new Thread(() => SendMeshObjectRemoval(id));
        tSendObject.Start();
    }

    private void SendMeshObjectAddition(Vector3[] vertices, int[] triangles, int id)
    {
        int numVertices = vertices.Length;
        int numTriangles = triangles.Length;
        int meshId = id;

        int messageLength = 4 + 4 + 4 + (12 * numVertices) + (4 * numTriangles);

        byte[] serializedMesh = new byte[messageLength + 8];

        // Format:
        // 32-bit sync
        // 32-bit length
        // 32-bit mesh ID
        // 32-bit vertex count
        // 32-bit triangle count
        // Vertex list:
        //  - 32 bit x
        //  - 32 bit y
        //  - 32 bit z
        // Triangle list:
        //  - 32 bit index
        byte[] frameHeader = { 0x1A, 0xCF, 0xFC, 0x1D,
                                (byte)((messageLength & 0xFF000000) >> 24),
                                (byte)((messageLength & 0x00FF0000) >> 16),
                                (byte)((messageLength & 0x0000FF00) >> 8),
                                (byte)(messageLength >> 0),
                                (byte)((meshId & 0xFF000000) >> 24),
                                (byte)((meshId & 0x00FF0000) >> 16),
                                (byte)((meshId & 0x0000FF00) >> 8),
                                (byte)(meshId >> 0),
                                (byte)((numVertices & 0xFF000000) >> 24),
                                (byte)((numVertices & 0x00FF0000) >> 16),
                                (byte)((numVertices & 0x0000FF00) >> 8),
                                (byte)(numVertices >> 0),
                                (byte)((numTriangles & 0xFF000000) >> 24),
                                (byte)((numTriangles & 0x00FF0000) >> 16),
                                (byte)((numTriangles & 0x0000FF00) >> 8),
                                (byte)(numTriangles >> 0)
                            };
        System.Buffer.BlockCopy(frameHeader, 0, serializedMesh, 0, frameHeader.Length);

        // add vertices
        for (int i = 0; i < (numVertices * 12); i += 12)
        {
            System.Buffer.BlockCopy(BitConverter.GetBytes(vertices[i / 12].x), 0, serializedMesh, frameHeader.Length + i, sizeof(float));
            System.Buffer.BlockCopy(BitConverter.GetBytes(vertices[i / 12].y), 0, serializedMesh, frameHeader.Length + (i + 4), sizeof(float));
            System.Buffer.BlockCopy(BitConverter.GetBytes(vertices[i / 12].z), 0, serializedMesh, frameHeader.Length + (i + 8), sizeof(float));
        }
        for (int i = 0; i < (numTriangles * 4); i += 4)
        {
            System.Buffer.BlockCopy(BitConverter.GetBytes(triangles[i / 4]), 0, serializedMesh, frameHeader.Length + (numVertices * 12) + i, sizeof(float));
        }

        _tcpStream.Write(serializedMesh, 0, serializedMesh.Length);

    }

    private void SendMeshObjectRemoval(int id)
    {
        _debugString += "object removal thread";
        int messageLength = 4 + 4 + 4;
        byte[] serializedMesh = new byte[messageLength + 8];

        // Format:
        // 32-bit sync
        // 32-bit length
        // 32-bit mesh ID
        // 32-bit vertex count = 0
        // 32-bit triangle count = 0
        byte[] frameHeader = { 0x1A, 0xCF, 0xFC, 0x1D,
                                   (byte)((messageLength & 0xFF000000) >> 24),
                                   (byte)((messageLength & 0x00FF0000) >> 16),
                                   (byte)((messageLength & 0x0000FF00) >> 8),
                                   (byte)(messageLength >> 0),
                                   (byte)((id & 0xFF000000) >> 24),
                                   (byte)((id & 0x00FF0000) >> 16),
                                   (byte)((id & 0x0000FF00) >> 8),
                                   (byte)(id >> 0),
                                   0, 0, 0, 0, 0, 0, 0, 0
                             };
        System.Buffer.BlockCopy(frameHeader, 0, serializedMesh, 0, frameHeader.Length);

        _tcpStream.Write(serializedMesh, 0, serializedMesh.Length);
    }

    private void DrawBox(Vector3 cornerLeft, Vector3 cornerTop,
                         Vector3 cornerRight, Vector3 cornerBot,
                         string classTypeStr)
    {
        Color color = new Color(0.3f, 0.4f, 0.6f);

        GameObject lineObject = new GameObject();
        LineRenderer line = lineObject.AddComponent<LineRenderer>();
        line.startWidth = 0.01f;
        line.endWidth = 0.01f;
        line.startColor = color;
        line.endColor = color;
        line.positionCount = 5;

        line.SetPosition(0, cornerLeft);
        line.SetPosition(1, cornerTop);
        line.SetPosition(2, cornerRight);
        line.SetPosition(3, cornerBot);
        line.SetPosition(4, cornerLeft);
        line.enabled = true;

        // Draw the label
        GameObject textObject = new GameObject();
        TextMesh textMesh = textObject.AddComponent<TextMesh>();
        MeshRenderer meshRenderer = textObject.GetComponent<MeshRenderer>();
        textMesh.color = Color.blue;
        textMesh.transform.localScale = new Vector3(0.04f, 0.04f, 0.04f);

        textMesh.text = classTypeStr;
        Vector3 textPos = cornerLeft;
        textPos.y = cornerLeft.y + 0.1f;
        textMesh.transform.position = textPos;

        _latestDetectionObjects.Add(lineObject);
        _latestDetectionObjects.Add(textObject);
    }
}
