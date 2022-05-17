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

    // Ros stuff
    ROSConnection ros;
    string audioTopicName = "HeadsetAudioData";

    private const int recordingDuration = 1;
    private const int sampleRate = 48000;

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
        ros.RegisterPublisher<HeadsetAudioDataMsg>(audioTopicName);

        string microphoneName = "";
        foreach (var device in Microphone.devices)
        {
            log.LogInfo("Microphone name: " + device);
            microphoneName = device;
        }

        // Setup the microphone to start recording
        audioObject = new GameObject();
        audioSource = audioObject.AddComponent<AudioSource>();

        audioSource.clip = Microphone.Start(microphoneName, // Device name
                                            true, // Loop
                                            recordingDuration, // Length of recording (sec)
                                            sampleRate); // Sample rate
        audioSource.loop = true;

        //TODO: mute?

        // Wait for recording to start
        while ((Microphone.GetPosition(null) <= 0)) { }
        audioSource.Play();

        // TODO: see if there is way to have the audio source not playback in the headset
        audioSource.volume = 0.01f; // Reduce the volume so we don't hear it in the headset
    }

    // OnAudioFilterRead is called every time an audio chunk is received (every ~20ms)
    // from the audio clip on the audio source.
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

        float duration = (1.0f / (float)sampleRate) * ((float)data.Length / (float)channels);

        // Create the ROS audio message
        HeadsetAudioDataMsg audioMsg = new HeadsetAudioDataMsg(Convert.ToByte(channels),
                                                               sampleRate,
                                                               duration,
                                                               scaledData);
        ros.Publish(audioTopicName, audioMsg);
    }

}
