using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;

using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Angel;


public class AudioCapture : MonoBehaviour
{
    private Logger _logger = null;
    GameObject audioObject = null;

    AudioSource audioSource;
    string microphone;
    string debugString = "";
    string audioTopicName = "HeadsetAudioData";


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

        // Create the audio publisher
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<AudioMsg>(audioTopicName);


        foreach (var device in Microphone.devices)
        {
            log.LogInfo("Name: " + device);
            microphone = device;
        }

        // Setup the microphone to start recording
        try
        {
            audioObject = new GameObject();
            audioSource = audioObject.AddComponent<AudioSource>();
            audioSource.clip = Microphone.Start(microphone, true, 1, 48000);
            audioSource.loop = true;

            while ((Microphone.GetPosition(null) <= 0)) { }
            audioSource.Play();

            // TODO: see if there is way to have the audio source not playback in the headset
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

    // OnAudioFilterRead is called every time an audio chunk is received.
    // Audio data is an array of floats ranging from -1 to 1.
    // Note: This function is NOT executed on the application main thread,
    // so use of Unity functions is not permitted.
    // More info: https://docs.unity3d.com/ScriptReference/MonoBehaviour.OnAudioFilterRead.html
    void OnAudioFilterRead(float[] data, int channels)
    {
        // Scale the sound up to increase the volume since we reduced it earlier by 0.01x
        float[] scaledData = new float[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            scaledData[i] = data[i] * 100;
        }

        byte[] frameData = new byte[(scaledData.Length * sizeof(float)) + 8];

        System.Buffer.BlockCopy(frameHeader, 0, frameData, 0, frameHeader.Length);
        System.Buffer.BlockCopy(scaledData, 0, frameData, 8, scaledData.Length * sizeof(float));

    }

}
