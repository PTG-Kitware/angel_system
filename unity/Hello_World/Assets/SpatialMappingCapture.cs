using Microsoft.MixedReality.SceneUnderstanding;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.SpatialAwareness;
using Microsoft.MixedReality.Toolkit.Examples.Demos;
using Microsoft.MixedReality.Toolkit.Experimental.SpatialAwareness;
using Microsoft.MixedReality.Toolkit.UI;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.NetworkInformation;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;


namespace Microsoft.MixedReality.Toolkit.Experimental.SceneUnderstanding
{
    public class SpatialMappingCapture : DemoSpatialMeshHandler, IMixedRealitySpatialAwarenessObservationHandler<SpatialAwarenessSceneObject>
    //public class SpatialMappingCapture : MonoBehaviour
    {
        System.Net.Sockets.TcpClient tcpClient;
        System.Net.Sockets.TcpListener tcpServer;
        NetworkStream tcpStream;
        public string TcpServerIPAddr = "";
        public const int SMTcpPort = 11010;

        private Logger _logger = null;

        string debugString = "";

        // Spatial awareness & scene understanding
        IEnumerable<SpatialAwarenessMeshObject> meshes;
        IMixedRealitySpatialAwarenessMeshObserver spatialAwarenessObserver = null;
        Scene worldScene;

        private IMixedRealitySceneUnderstandingObserver sceneUnderstandingObserver;
        private List<GameObject> instantiatedPrefabs;
        private Dictionary<SpatialAwarenessSurfaceTypes, Dictionary<int, SpatialAwarenessSceneObject>> observedSceneObjects;
        private bool InstantiatePrefabs = false;



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
        protected override void Start()
        {
            Logger log = logger();

            // Resume Mesh Observation from all Observers
            CoreServices.SpatialAwarenessSystem.ResumeObservers();

            if (!SceneObserver.IsSupported())
            {
                // Handle the error
                log.LogInfo("Scene observer not supported!");
            }

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
            */
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

            //sceneUnderstandingObserver.Disable(); // NOTE: take this out to enable observer


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
            log.LogInfo("Waiting for spatial mapping TCP connection");
        }

        // Update is called once per frame
        void Update()
        {
            // Setup the spatial awareness observer
            if (spatialAwarenessObserver == null)
            {
                var meshObservers = (CoreServices.SpatialAwarenessSystem as IMixedRealityDataProviderAccess).GetDataProviders<IMixedRealitySpatialAwarenessMeshObserver>();
                foreach (var observers in meshObservers)
                {
                    if (observers.Meshes.Count != 0)
                    {
                        spatialAwarenessObserver = observers;
                        spatialAwarenessObserver.DisplayOption = SpatialAwarenessMeshDisplayOptions.None;
                        this.logger().LogInfo("Detail level: " + spatialAwarenessObserver.LevelOfDetail.ToString());
                    }
                }
            }
            else
            {
                //this.logger().LogInfo("Num meshes: " + spatialAwarenessObserver.Meshes.Count.ToString());
                //this.logger().LogInfo("Num scene objects: " + sceneUnderstandingObserver.SceneObjects.Count.ToString());


                // Loop through all known Meshes
                foreach (SpatialAwarenessMeshObject meshObject in spatialAwarenessObserver.Meshes.Values)
                {
                    Mesh mesh = meshObject.Filter.mesh;
                    // Do something with the Mesh object
                }

                byte[] frameHeader = { 0x1A, 0xCF, 0xFC, 0x1D,
                               (byte)(((spatialAwarenessObserver.Meshes.Count) & 0xFF000000) >> 24),
                               (byte)(((spatialAwarenessObserver.Meshes.Count) & 0x00FF0000) >> 16),
                               (byte)(((spatialAwarenessObserver.Meshes.Count) & 0x0000FF00) >> 8),
                               (byte)(((spatialAwarenessObserver.Meshes.Count) & 0x000000FF) >> 0)
                             };

                // Send the meshes
                if (tcpStream != null)
                {
                    // TODO: send the meshes here

                }
            }

            /*
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

        #region MonoBehaviour Functions

        protected override void OnEnable()
        {
            RegisterEventHandlers<IMixedRealitySpatialAwarenessObservationHandler<SpatialAwarenessSceneObject>, SpatialAwarenessSceneObject>();
        }

        protected override void OnDisable()
        {
            UnregisterEventHandlers<IMixedRealitySpatialAwarenessObservationHandler<SpatialAwarenessSceneObject>, SpatialAwarenessSceneObject>();
        }

        protected override void OnDestroy()
        {
            UnregisterEventHandlers<IMixedRealitySpatialAwarenessObservationHandler<SpatialAwarenessSceneObject>, SpatialAwarenessSceneObject>();
        }

        #endregion MonoBehaviour Functions
        #region IMixedRealitySpatialAwarenessObservationHandler Implementations

        /// <inheritdoc />
        public void OnObservationAdded(MixedRealitySpatialAwarenessEventData<SpatialAwarenessSceneObject> eventData)
        {
            // This method called everytime a SceneObject created by the SU observer
            // The eventData contains everything you need do something useful

            //this.logger().LogInfo("observation added!");


            AddToData(eventData.Id);

            if (observedSceneObjects.TryGetValue(eventData.SpatialObject.SurfaceType, out Dictionary<int, SpatialAwarenessSceneObject> sceneObjectDict))
            {
                sceneObjectDict.Add(eventData.Id, eventData.SpatialObject);
            }
            else
            {
                observedSceneObjects.Add(eventData.SpatialObject.SurfaceType, new Dictionary<int, SpatialAwarenessSceneObject> { { eventData.Id, eventData.SpatialObject } });
            }

            if (InstantiatePrefabs && eventData.SpatialObject.Quads.Count > 0)
            {
                /*
                var prefab = Instantiate(InstantiatedPrefab);
                prefab.transform.SetPositionAndRotation(eventData.SpatialObject.Position, eventData.SpatialObject.Rotation);
                float sx = eventData.SpatialObject.Quads[0].Extents.x;
                float sy = eventData.SpatialObject.Quads[0].Extents.y;
                prefab.transform.localScale = new Vector3(sx, sy, .1f);
                if (InstantiatedParent)
                {
                    prefab.transform.SetParent(InstantiatedParent);
                }
                instantiatedPrefabs.Add(prefab);
                */
            }
            else
            {
                foreach (var quad in eventData.SpatialObject.Quads)
                {
                    //quad.GameObject.GetComponent<Renderer>().material.color = ColorForSurfaceType(eventData.SpatialObject.SurfaceType);
                }

            }
        }

        /// <inheritdoc />
        public void OnObservationUpdated(MixedRealitySpatialAwarenessEventData<SpatialAwarenessSceneObject> eventData)
        {
            UpdateData(eventData.Id);

            if (observedSceneObjects.TryGetValue(eventData.SpatialObject.SurfaceType, out Dictionary<int, SpatialAwarenessSceneObject> sceneObjectDict))
            {
                observedSceneObjects[eventData.SpatialObject.SurfaceType][eventData.Id] = eventData.SpatialObject;
            }
            else
            {
                observedSceneObjects.Add(eventData.SpatialObject.SurfaceType, new Dictionary<int, SpatialAwarenessSceneObject> { { eventData.Id, eventData.SpatialObject } });
            }
        }

        /// <inheritdoc />
        public void OnObservationRemoved(MixedRealitySpatialAwarenessEventData<SpatialAwarenessSceneObject> eventData)
        {
            RemoveFromData(eventData.Id);

            foreach (var sceneObjectDict in observedSceneObjects.Values)
            {
                sceneObjectDict?.Remove(eventData.Id);
            }
        }
        #endregion IMixedRealitySpatialAwarenessObservationHandler Implementations

        /// <summary>
        /// Gets the color of the given surface type
        /// </summary>
        /// <param name="surfaceType">The surface type to get color for</param>
        /// <returns>The color of the type</returns>
        private Color ColorForSurfaceType(SpatialAwarenessSurfaceTypes surfaceType)
        {
            // shout-out to solarized!

            switch (surfaceType)
            {
                case SpatialAwarenessSurfaceTypes.Unknown:
                    return new Color32(220, 50, 47, 255); // red
                case SpatialAwarenessSurfaceTypes.Floor:
                    return new Color32(38, 139, 210, 255); // blue
                case SpatialAwarenessSurfaceTypes.Ceiling:
                    return new Color32(108, 113, 196, 255); // violet
                case SpatialAwarenessSurfaceTypes.Wall:
                    return new Color32(181, 137, 0, 255); // yellow
                case SpatialAwarenessSurfaceTypes.Platform:
                    return new Color32(133, 153, 0, 255); // green
                case SpatialAwarenessSurfaceTypes.Background:
                    return new Color32(203, 75, 22, 255); // orange
                case SpatialAwarenessSurfaceTypes.World:
                    return new Color32(211, 54, 130, 255); // magenta
                case SpatialAwarenessSurfaceTypes.Inferred:
                    return new Color32(42, 161, 152, 255); // cyan
                default:
                    return new Color32(220, 50, 47, 255); // red
            }
        }
    }
}
