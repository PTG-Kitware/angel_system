using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using UnityEngine;

public class AudioCapture : MonoBehaviour
{
    private Logger _logger = null;
    GameObject audioObject = null;

    AudioSource audioSource;
    string microphone;
    string debugString = "";

    System.Net.Sockets.TcpClient tcpClient;
    NetworkStream tcpStream;

    public const string TcpServerIPAddr = "169.254.103.120";
    public const int AudioTcpPort = 11009;

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

        // Connect to the python TCP server
        this.tcpClient = new System.Net.Sockets.TcpClient();
        try
        {
            this.tcpClient.Connect(TcpServerIPAddr, AudioTcpPort);
            log.LogInfo("TCP client PV connected!");
            this.tcpStream = this.tcpClient.GetStream();
        }
        catch (Exception e)
        {
            log.LogInfo(e.ToString());
        }

        log.LogInfo("Setting up audio capture");
        foreach (var device in Microphone.devices)
        {
            log.LogInfo("Name: " + device);
            microphone = device;
        }

        try
        {
            audioObject = new GameObject();
            audioSource = audioObject.AddComponent<AudioSource>();
            audioSource.clip = Microphone.Start(microphone, true, 1, 48000);
            audioSource.loop = true;

            while ((Microphone.GetPosition(null) <= 0)) { }
            audioSource.Play();
            audioSource.volume = 0.01f; // Reduce the volume so we don't hear it in the headset

            AudioConfiguration ac = AudioSettings.GetConfiguration();
            //log.LogInfo("sample rate: " + AudioSettings.outputSampleRate.ToString());
            //log.LogInfo("speakermode: " + ac.speakerMode.ToString());
            //log.LogInfo("dsp size: " + ac.dspBufferSize.ToString());
        }
        catch (Exception e)
        {
            log.LogInfo("Exception: " + e);
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (debugString != "")
        {
            this.logger().LogInfo(debugString);
        }
    }

    void OnAudioFilterRead(float[] data, int channels)
    {
        //debugString = "size: " + data.Length.ToString();
        //debugString += "channels: " + channels.ToString();

        // Scale the sound up to increase the volume since we reduced it earlier by 0.01x
        float[] scaledData = new float[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            scaledData[i] = data[i] * 100;
        }

        byte[] frameData = new byte[(scaledData.Length * sizeof(float)) + 8];
        
        // Add header
        byte[] frameHeader = { 0x1A, 0xCF, 0xFC, 0x1D,
                               (byte)(((data.Length * sizeof(float)) & 0xFF000000) >> 24),
                               (byte)(((data.Length * sizeof(float)) & 0x00FF0000) >> 16),
                               (byte)(((data.Length * sizeof(float)) & 0x0000FF00) >> 8),
                               (byte)(((data.Length * sizeof(float)) & 0x000000FF) >> 0) };

        System.Buffer.BlockCopy(frameHeader, 0, frameData, 0, frameHeader.Length);
        System.Buffer.BlockCopy(scaledData, 0, frameData, 8, scaledData.Length * sizeof(float));

        // Send the data through the socket.
        if (tcpStream != null)
        {
            tcpStream.Write(frameData, 0, frameData.Length);
            tcpStream.Flush();
        }

    }

}
