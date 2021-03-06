//Do not edit! This file was generated by Unity-ROS MessageGeneration.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;

namespace RosMessageTypes.Angel
{
    [Serializable]
    public class ObjectDetection2dMsg : Message
    {
        public const string k_RosMessageName = "angel_msgs/ObjectDetection2d";
        public override string RosMessageName => k_RosMessageName;

        // 
        //  A single object detection on a 2D image plane.
        //  This includes a predicted set of classifications.
        // 
        //  This is modeled after the `smqtk_detection.DetectImageObjects` return
        //  structure.
        // 
        //  TODO: maybe drop labels and provide a service for consumers to query for
        //        label list just once.
        // 
        //  Image-space axis-aligned bounding box. Origin is upper-left corner.
        public float left;
        public float right;
        public float top;
        public float bottom;
        //  Prediction label confidences. Should be same length as `label_vec`
        public double[] label_confidence_vec;
        //  Prediction classification labels.
        public string[] label_vec;

        public ObjectDetection2dMsg()
        {
            this.left = 0.0f;
            this.right = 0.0f;
            this.top = 0.0f;
            this.bottom = 0.0f;
            this.label_confidence_vec = new double[0];
            this.label_vec = new string[0];
        }

        public ObjectDetection2dMsg(float left, float right, float top, float bottom, double[] label_confidence_vec, string[] label_vec)
        {
            this.left = left;
            this.right = right;
            this.top = top;
            this.bottom = bottom;
            this.label_confidence_vec = label_confidence_vec;
            this.label_vec = label_vec;
        }

        public static ObjectDetection2dMsg Deserialize(MessageDeserializer deserializer) => new ObjectDetection2dMsg(deserializer);

        private ObjectDetection2dMsg(MessageDeserializer deserializer)
        {
            deserializer.Read(out this.left);
            deserializer.Read(out this.right);
            deserializer.Read(out this.top);
            deserializer.Read(out this.bottom);
            deserializer.Read(out this.label_confidence_vec, sizeof(double), deserializer.ReadLength());
            deserializer.Read(out this.label_vec, deserializer.ReadLength());
        }

        public override void SerializeTo(MessageSerializer serializer)
        {
            serializer.Write(this.left);
            serializer.Write(this.right);
            serializer.Write(this.top);
            serializer.Write(this.bottom);
            serializer.WriteLength(this.label_confidence_vec);
            serializer.Write(this.label_confidence_vec);
            serializer.WriteLength(this.label_vec);
            serializer.Write(this.label_vec);
        }

        public override string ToString()
        {
            return "ObjectDetection2dMsg: " +
            "\nleft: " + left.ToString() +
            "\nright: " + right.ToString() +
            "\ntop: " + top.ToString() +
            "\nbottom: " + bottom.ToString() +
            "\nlabel_confidence_vec: " + System.String.Join(", ", label_confidence_vec.ToList()) +
            "\nlabel_vec: " + System.String.Join(", ", label_vec.ToList());
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
