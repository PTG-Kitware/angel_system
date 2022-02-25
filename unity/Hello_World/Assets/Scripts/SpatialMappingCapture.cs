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


//using SpatialAwarenessHandler = IMixedRealitySpatialAwarenessObservationHandler<SpatialAwarenessMeshObject>;

//public class SpatialMappingCapture : DemoSpatialMeshHandler, IMixedRealitySpatialAwarenessObservationHandler<SpatialAwarenessSceneObject>
//public class SpatialMappingCapture : MonoBehaviour
public class SpatialMappingCapture : MonoBehaviour, IMixedRealitySpatialAwarenessObservationHandler<SpatialAwarenessMeshObject>
{
    System.Net.Sockets.TcpClient tcpClient;
    System.Net.Sockets.TcpListener tcpServer;
    NetworkStream tcpStream;
    public string TcpServerIPAddr = "";
    public const int SMTcpPort = 11010;

    private Logger _logger = null;

    string debugString = "";

    // Spatial awareness & scene understanding
    //List<MixedRealitySpatialAwarenessEventData<SpatialAwarenessMeshObject>> meshes = new List<MixedRealitySpatialAwarenessEventData<SpatialAwarenessMeshObject>>();
    //IMixedRealitySpatialAwarenessMeshObserver spatialAwarenessObserver = null;
    //Scene worldScene;
    //private IMixedRealitySceneUnderstandingObserver sceneUnderstandingObserver;

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

        /*
        // This call should grant the access we need.
        await SceneObserver.RequestAccessAsync();

        // Create Query settings for the scene update
        SceneQuerySettings querySettings;

        querySettings.EnableSceneObjectQuads = true;                                       // Requests that the scene updates quads.
        querySettings.EnableSceneObjectMeshes = true;                                      // Requests that the scene updates watertight mesh data.
        querySettings.EnableOnlyObservedSceneObjects = false;                              // Do not explicitly turn off quad inference.
        querySettings.EnableWorldMesh = true;                                              // Requests a static version of the spatial mapping mesh.
        querySettings.RequestedMeshLevelOfDetail = SceneMeshLevelOfDetail.Medium;          // Requests the finest LOD of the static spatial mapping mesh.

        // Initialize a new Scene
        worldScene = SceneObserver.ComputeAsync(querySettings, 3.0f).GetAwaiter().GetResult();
        sceneUnderstandingObserver = CoreServices.GetSpatialAwarenessSystemDataProvider<IMixedRealitySceneUnderstandingObserver>();

        if (sceneUnderstandingObserver == null)
        {
            log.LogInfo("Couldn't access Scene Understanding Observer!");
            return;
        }
        instantiatedPrefabs = new List<GameObject>();
        observedSceneObjects = new Dictionary<SpatialAwarenessSurfaceTypes, Dictionary<int, SpatialAwarenessSceneObject>>();

        log.LogInfo("World scene initialized!");
        log.LogInfo("Observer info:");
        log.LogInfo("Observer is running: " + sceneUnderstandingObserver.IsRunning.ToString());
        log.LogInfo("Observer auto update: " + sceneUnderstandingObserver.AutoUpdate.ToString());
        log.LogInfo("Observer scene objects: " + sceneUnderstandingObserver.SceneObjects.ToString());
        log.LogInfo("Observer update interval: " + sceneUnderstandingObserver.UpdateInterval.ToString());
        log.LogInfo("Observer surface types: " + sceneUnderstandingObserver.SurfaceTypes.ToString());

        NOTE: loop to access the meshes from the scene understanding SDK
        // Find the first quad
        foreach (KeyValuePair<int, SpatialAwarenessSceneObject> sceneObject in sceneUnderstandingObserver.SceneObjects)
        {
            //this.logger().LogInfo("object: " + sceneObject.Value.SurfaceType.ToString() + " " + sceneObject.Value.Position);

            // Get the meshes
            var meshes = sceneObject.Value.Meshes;
            foreach (var mesh in meshes)
            {
                //this.logger().LogInfo("mesh vertices: " + mesh.Vertices.ToString());
                foreach (var v in mesh.Vertices)
                {
                    this.logger().LogInfo("mesh vertices: " + v.ToString());
                }
            }
        }
        */

        try
        {
            TcpServerIPAddr = PTGUtilities.getIPv4AddressString();
        }
        catch (InvalidIPConfiguration e)
        {
            log.LogInfo(e.ToString());
            return;
        }

        Thread tSpatialMappingCapture = new Thread(SetupSpatialMappingCapture);
        tSpatialMappingCapture.Start();
        //log.LogInfo("Waiting for spatial mapping TCP connection");
    }

    // Update is called once per frame
    void Update()
    {
        if (debugString != "")
        {
            this.logger().LogInfo(debugString);
            debugString = "";
        }

        if (tcpStream != null)
        {
            // Resume Mesh Observation from all Observers
            CoreServices.SpatialAwarenessSystem.ResumeObservers();
        }
    }

    void SetupSpatialMappingCapture()
    {
        try
        {
            IPAddress localAddr = IPAddress.Parse(TcpServerIPAddr);

            tcpServer = new TcpListener(localAddr, SMTcpPort);

            // Start listening for client requests.
            tcpServer.Start();

            // Perform a blocking call to accept requests.
            // You could also use server.AcceptSocket() here.
            tcpClient = tcpServer.AcceptTcpClient();
            tcpStream = tcpClient.GetStream();
        }
        catch (Exception e)
        {
            debugString += e.ToString();
        }
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
        if (tcpStream != null)
        {
            // Copy over the mesh data we need because we cannot access it outside of the main thread
            Vector3[] vertices = eventData.SpatialObject.Filter.mesh.vertices;
            int[] triangles = eventData.SpatialObject.Filter.mesh.triangles;
            int id = eventData.SpatialObject.Id;
            Thread tSendObject = new Thread(() => SendMeshObjectAddition(vertices,
                                                                         triangles,
                                                                         id));
            tSendObject.Start();

            //SendMeshObjectAddition(eventData.SpatialObject);
            //this.logger().LogInfo("t started, sent mesh id: " + eventData.SpatialObject.Id.ToString());
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

        this.logger().LogInfo("removal! " + eventData.SpatialObject.Id.ToString());

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

        //debugString += meshObject.Filter.mesh.vertices[0].x.ToString() + " ";
        //debugString += meshObject.Filter.mesh.vertices[0].y.ToString() + " ";
        //debugString += meshObject.Filter.mesh.vertices[0].z.ToString() + " ";

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

        tcpStream.Write(serializedMesh, 0, serializedMesh.Length);

    }

    private void SendMeshObjectRemoval(int id)
    {
        debugString += "object removal thread";
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

        tcpStream.Write(serializedMesh, 0, serializedMesh.Length);
    }
}
