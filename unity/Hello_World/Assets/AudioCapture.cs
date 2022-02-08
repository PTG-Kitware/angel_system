using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AudioCapture : MonoBehaviour
{
    GameObject loggerObject = null;
    GameObject audioObject = null;

    AudioSource audioSource;
    string microphone;
    string debugString = "";

    // Start is called before the first frame update
    void Start()
    {
        this.loggerObject = GameObject.Find("Logger");

        this.loggerObject.GetComponent<Logger>().LogInfo("Setting up audio capture");
        foreach (var device in Microphone.devices)
        {
            this.loggerObject.GetComponent<Logger>().LogInfo("Name: " + device);
            microphone = device;
        }

        try
        {
            audioObject = new GameObject();
            audioSource = audioObject.AddComponent<AudioSource>();
            audioSource.clip = Microphone.Start(microphone, true, 5, 48000);
            audioSource.loop = true;

            while ((Microphone.GetPosition(null) <= 0)) { }
            audioSource.Play();

            int minFreq = 0;
            int maxFreq = 0;
            Microphone.GetDeviceCaps(microphone, out minFreq, out maxFreq);
            this.loggerObject.GetComponent<Logger>().LogInfo("clip: " + audioSource.clip.ToString());
            this.loggerObject.GetComponent<Logger>().LogInfo("caps min: " + minFreq.ToString());
            this.loggerObject.GetComponent<Logger>().LogInfo("caps max: " + maxFreq.ToString());



        }
        catch (Exception e)
        {
            this.loggerObject.GetComponent<Logger>().LogInfo("Exception: " + e);
        }


    }

    // Update is called once per frame
    void Update()
    {
        if (Microphone.IsRecording(microphone))
        {
            //this.loggerObject.GetComponent<Logger>().LogInfo("recording!"); ;
        }
        else
        {
            //this.loggerObject.GetComponent<Logger>().LogInfo("done recording!");
            //this.loggerObject.GetComponent<Logger>().LogInfo("length: " + audioSource.clip.length.ToString());
            //this.loggerObject.GetComponent<Logger>().LogInfo("channels: " + audioSource.clip.channels.ToString());
            //this.loggerObject.GetComponent<Logger>().LogInfo("samples: " + audioSource.clip.samples.ToString());




        }

        if (debugString != "")
        {
            this.loggerObject.GetComponent<Logger>().LogInfo(debugString);
        }
    }

    void OnAudioFilterRead(float[] data, int channels)
    {
        debugString = "here!";
        //this.loggerObject.GetComponent<Logger>().LogInfo("audio filter read!");

        for (int i = 0; i < 10; i++)
        {
            debugString += data[i].ToString() + " ";
        }


    }

}
