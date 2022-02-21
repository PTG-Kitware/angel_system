using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.SpatialAwareness;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.NetworkInformation;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

public class SpatialMappingCapture : MonoBehaviour
{
    System.Net.Sockets.TcpClient tcpClient;
    System.Net.Sockets.TcpListener tcpServer;
    NetworkStream tcpStream;
    public string TcpServerIPAddr = "";
    public const int SMTcpPort = 11010;

    private Logger _logger = null;

    string debugString = "";

    // Spatial awareness
    IEnumerable<SpatialAwarenessMeshObject> meshes;
    IMixedRealitySpatialAwarenessMeshObserver observer = null;

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
    void Start()
    {
        Logger log = logger();

        try
        {
            TcpServerIPAddr = PTGUtilities.getIPv4AddressString();
        }
        catch (InvalidIPConfiguration e)
        {
            log.LogInfo(e.ToString());
            return;
        }

        // Resume Mesh Observation from all Observers
        CoreServices.SpatialAwarenessSystem.ResumeObservers();

        log.LogInfo("Using IPv4 addr: " + TcpServerIPAddr);

        Thread tSpatialMappingCapture = new Thread(SetupSpatialMappingCapture);
        tSpatialMappingCapture.Start();
        log.LogInfo("Waiting for spatial mapping TCP connection");
    }

    // Update is called once per frame
    void Update()
    {
        // Setup the spatial awareness observer
        if (observer == null)
        {
            var meshObservers = (CoreServices.SpatialAwarenessSystem as IMixedRealityDataProviderAccess).GetDataProviders<IMixedRealitySpatialAwarenessMeshObserver>();
            foreach (var observers in meshObservers)
            {
                if (observers.Meshes.Count != 0)
                {
                    observer = observers;
                    observer.DisplayOption = SpatialAwarenessMeshDisplayOptions.Visible;
                    this.logger().LogInfo("Detail level: " + observer.LevelOfDetail.ToString());

                }
            }
        }

        this.logger().LogInfo("Num meshes: " + observer.Meshes.Count.ToString());

        byte[] frameHeader = { 0x1A, 0xCF, 0xFC, 0x1D,
                               (byte)(((observer.Meshes.Count) & 0xFF000000) >> 24),
                               (byte)(((observer.Meshes.Count) & 0x00FF0000) >> 16),
                               (byte)(((observer.Meshes.Count) & 0x0000FF00) >> 8),
                               (byte)(((observer.Meshes.Count) & 0x000000FF) >> 0)
                             };

        // Send the meshes
        if (tcpStream != null)
        {
            // TODO: send the meshes here

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
}
