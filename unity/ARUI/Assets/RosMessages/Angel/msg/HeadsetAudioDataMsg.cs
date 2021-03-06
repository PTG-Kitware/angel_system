//Do not edit! This file was generated by Unity-ROS MessageGeneration.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;

namespace RosMessageTypes.Angel
{
    [Serializable]
    public class HeadsetAudioDataMsg : Message
    {
        public const string k_RosMessageName = "angel_msgs/HeadsetAudioData";
        public override string RosMessageName => k_RosMessageName;

        // 
        //  Message containing audio data captured from the HL2 microphones.
        // 
        //  NOTE: There does not seem to be an audio message type in the default
        //  ROS message packages. This message is based on the audio messages
        //  defined in https://github.com/ros-drivers/audio_common
        // 
        //  Header with time stamp
        public Std.HeaderMsg header;
        //  Audio meta data
        public int channels;
        //  Sampling rate [Hz]
        public int sample_rate;
        //  Duration of sample (s)
        public float sample_duration;
        //  Audio data
        public float[] data;

        public HeadsetAudioDataMsg()
        {
            this.header = new Std.HeaderMsg();
            this.channels = 0;
            this.sample_rate = 0;
            this.sample_duration = 0.0f;
            this.data = new float[0];
        }

        public HeadsetAudioDataMsg(Std.HeaderMsg header, int channels, int sample_rate, float sample_duration, float[] data)
        {
            this.header = header;
            this.channels = channels;
            this.sample_rate = sample_rate;
            this.sample_duration = sample_duration;
            this.data = data;
        }

        public static HeadsetAudioDataMsg Deserialize(MessageDeserializer deserializer) => new HeadsetAudioDataMsg(deserializer);

        private HeadsetAudioDataMsg(MessageDeserializer deserializer)
        {
            this.header = Std.HeaderMsg.Deserialize(deserializer);
            deserializer.Read(out this.channels);
            deserializer.Read(out this.sample_rate);
            deserializer.Read(out this.sample_duration);
            deserializer.Read(out this.data, sizeof(float), deserializer.ReadLength());
        }

        public override void SerializeTo(MessageSerializer serializer)
        {
            serializer.Write(this.header);
            serializer.Write(this.channels);
            serializer.Write(this.sample_rate);
            serializer.Write(this.sample_duration);
            serializer.WriteLength(this.data);
            serializer.Write(this.data);
        }

        public override string ToString()
        {
            return "HeadsetAudioDataMsg: " +
            "\nheader: " + header.ToString() +
            "\nchannels: " + channels.ToString() +
            "\nsample_rate: " + sample_rate.ToString() +
            "\nsample_duration: " + sample_duration.ToString() +
            "\ndata: " + System.String.Join(", ", data.ToList());
        }

#if UNITY_EDITOR
        [UnityEditor.InitializeOnLoadMethod]
#else
        [UnityEngine.RuntimeInitializeOnLoadMethod]
#endif
        public static void Register()
        {
            MessageRegistry.Register(k_RosMessageName, Deserialize);
        }
    }
}
