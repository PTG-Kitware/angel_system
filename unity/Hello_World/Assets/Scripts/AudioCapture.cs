using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Angel;
using RosMessageTypes.BuiltinInterfaces;
using RosMessageTypes.Std;


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

    private bool running = false;

    // For filling in ROS message timestamp
    DateTime timeOrigin = new DateTime(1970, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc);

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

        // Wait for recording to start
        while ((Microphone.GetPosition(null) <= 0)) { }
        audioSource.Play();

        // In order to capture microphone audio without hearing the playback in the headset,
        // we need to scale the AudioSource volume down here, and then scale it back up when
        // the audio is processed in OnAudioFilterRead.
        // https://stackoverflow.com/questions/37787343/capture-audio-from-microphone-without-playing-it-back
        audioSource.volume = 0.01f;

        running = true;
    }

    // OnAudioFilterRead is called every time an audio chunk is received (every ~20ms)
    // from the audio clip on the audio source.
    // Audio data is an array of floats ranging from -1 to 1.
    // Note: This function is NOT executed on the application main thread,
    // so use of Unity functions is not permitted.
    // More info: https://docs.unity3d.com/ScriptReference/MonoBehaviour.OnAudioFilterRead.html
    void OnAudioFilterRead(float[] data, int channels)
    {
        // Wait for start function to finish
        if (!running)
            return;

        // Scale the sound up to increase the volume since we reduced it earlier by 0.01x
        float[] scaledData = new float[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            scaledData[i] = data[i] * 100;
        }

        float duration = (1.0f / (float)sampleRate) * ((float)data.Length / (float)channels);

        // Create the ROS audio message
        var currTime = DateTime.Now;
        TimeSpan diff = currTime.ToUniversalTime() - timeOrigin;
        var sec = Convert.ToInt32(Math.Floor(diff.TotalSeconds));
        var nsecRos = Convert.ToUInt32((diff.TotalSeconds - sec) * 1e9f);

        HeaderMsg header = new HeaderMsg(
            new TimeMsg(sec, nsecRos),
            "AudioData"
        );

        HeadsetAudioDataMsg audioMsg = new HeadsetAudioDataMsg(header,
                                                               channels,
                                                               sampleRate,
                                                               duration,
                                                               scaledData);
        ros.Publish(audioTopicName, audioMsg);
    }

}
