//Do not edit! This file was generated by Unity-ROS MessageGeneration.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;

namespace RosMessageTypes.Angel
{
    [Serializable]
    public class TaskNodeMsg : Message
    {
        public const string k_RosMessageName = "angel_msgs/TaskNode";
        public override string RosMessageName => k_RosMessageName;

        // 
        //  A single task node
        // 
        public string uid;
        public string name;
        //  NOTE: The following are speculative. Awaiting system-side task data-structure
        //  details to better inform the appropriate details to put here.
        //  Related objects
        public string[] object_uid;
        //  Action instruction
        public string action_name;
        public float action_time;
        //  may be zero if not applicable.

        public TaskNodeMsg()
        {
            this.uid = "";
            this.name = "";
            this.object_uid = new string[0];
            this.action_name = "";
            this.action_time = 0.0f;
        }

        public TaskNodeMsg(string uid, string name, string[] object_uid, string action_name, float action_time)
        {
            this.uid = uid;
            this.name = name;
            this.object_uid = object_uid;
            this.action_name = action_name;
            this.action_time = action_time;
        }

        public static TaskNodeMsg Deserialize(MessageDeserializer deserializer) => new TaskNodeMsg(deserializer);

        private TaskNodeMsg(MessageDeserializer deserializer)
        {
            deserializer.Read(out this.uid);
            deserializer.Read(out this.name);
            deserializer.Read(out this.object_uid, deserializer.ReadLength());
            deserializer.Read(out this.action_name);
            deserializer.Read(out this.action_time);
        }

        public override void SerializeTo(MessageSerializer serializer)
        {
            serializer.Write(this.uid);
            serializer.Write(this.name);
            serializer.WriteLength(this.object_uid);
            serializer.Write(this.object_uid);
            serializer.Write(this.action_name);
            serializer.Write(this.action_time);
        }

        public override string ToString()
        {
            return "TaskNodeMsg: " +
            "\nuid: " + uid.ToString() +
            "\nname: " + name.ToString() +
            "\nobject_uid: " + System.String.Join(", ", object_uid.ToList()) +
            "\naction_name: " + action_name.ToString() +
            "\naction_time: " + action_time.ToString();
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
