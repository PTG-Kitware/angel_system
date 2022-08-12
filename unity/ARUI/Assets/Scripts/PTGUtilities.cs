using System;
using System.Net;
using System.Net.NetworkInformation;
using UnityEngine;

using RosMessageTypes.BuiltinInterfaces;
using RosMessageTypes.Std;


public static class PTGUtilities
{
    // For filling in ROS message timestamps
    public static DateTime timeOrigin = new DateTime(1970, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc);


    /// <summary>
    /// Creates and returns a time stamped ROS std_msgs/Header.
    /// </summary>
    public static HeaderMsg getROSStdMsgsHeader(string frameId)
    {
        var currTime = DateTime.Now;
        TimeSpan diff = currTime.ToUniversalTime() - timeOrigin;
        var sec = Convert.ToInt32(Math.Floor(diff.TotalSeconds));
        var nsecRos = Convert.ToUInt32((diff.TotalSeconds - sec) * 1e9f);

        HeaderMsg header = new HeaderMsg(
            new TimeMsg(sec, nsecRos),
            frameId
        );

        return header;
    }

    /// <summary>
    /// Looks up the IPv4 address of the ethernet interfaces that is connected
    /// on the HoloLens and returns it as a string (e.g. "192.168.1.100").
    /// </summary>
    public static string getIPv4AddressString()
    {
        string ipAddr = "";
        NetworkInterface[] interfaces = NetworkInterface.GetAllNetworkInterfaces();
        foreach (NetworkInterface adapter in interfaces)
        {
            if (adapter.Supports(NetworkInterfaceComponent.IPv4) &&
                adapter.OperationalStatus == OperationalStatus.Up &&
                adapter.NetworkInterfaceType == NetworkInterfaceType.Ethernet)
            {
                foreach (UnicastIPAddressInformation ip in adapter.GetIPProperties().UnicastAddresses)
                {
                    if (ip.Address.AddressFamily == System.Net.Sockets.AddressFamily.InterNetwork)
                    {
                        ipAddr = ip.Address.ToString();
                        break;
                    }
                }
            }
        }

        if (ipAddr == "")
        {
            throw new InvalidIPConfiguration("No suitable IPv4 addresses found.");
        }

        return ipAddr;
    }

    /// <summary>
    /// Converts a 4x4 System.Numerics matrix to a 1D float array.
    /// </summary>
    public static float[] ConvertMatrixToFloatArray(System.Numerics.Matrix4x4 matrix)
    {
        return new float[16] {
            matrix.M11, matrix.M12, matrix.M13, matrix.M14,
            matrix.M21, matrix.M22, matrix.M23, matrix.M24,
            matrix.M31, matrix.M32, matrix.M33, matrix.M34,
            matrix.M41, matrix.M42, matrix.M43, matrix.M44
        };
    }

    /// <summary>
    /// Converts a 4x4 Unity matrix to a 1D float array.
    /// </summary>
    public static float[] ConvertUnityMatrixToFloatArray(Matrix4x4 matrix)
    {
        return new float[16] {
            matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3],
            matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3],
            matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3],
            matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3]
        };
    }

    /// <summary>
    /// Returns a 1D float array representing a 4x4 identity matrix.
    /// </summary>
    public static float[] GetIdentityMatrixFloatArray()
    {
        return new float[] { 1f, 0, 0, 0, 0, 1f, 0, 0, 0, 0, 1f, 0, 0, 0, 0, 1f };
    }

    /// <summary>
    /// Converts a byte array of length 64 to a 4x4 System.Numerics matrix.
    /// Each element in the returned array is a 32 bit float.
    /// </summary>
    public static System.Numerics.Matrix4x4 ConvertByteArrayToMatrix4x4(byte[] matrixAsBytes)
    {
        if (matrixAsBytes == null)
        {
            throw new ArgumentNullException("matrixAsBytes");
        }

        if (matrixAsBytes.Length != 64)
        {
            throw new Exception("Cannot convert byte[] to Matrix4x4. Size of array should be 64, but it is " + matrixAsBytes.Length);
        }

        var m = matrixAsBytes;
        return new System.Numerics.Matrix4x4(
            BitConverter.ToSingle(m, 0),
            BitConverter.ToSingle(m, 4),
            BitConverter.ToSingle(m, 8),
            BitConverter.ToSingle(m, 12),
            BitConverter.ToSingle(m, 16),
            BitConverter.ToSingle(m, 20),
            BitConverter.ToSingle(m, 24),
            BitConverter.ToSingle(m, 28),
            BitConverter.ToSingle(m, 32),
            BitConverter.ToSingle(m, 36),
            BitConverter.ToSingle(m, 40),
            BitConverter.ToSingle(m, 44),
            BitConverter.ToSingle(m, 48),
            BitConverter.ToSingle(m, 52),
            BitConverter.ToSingle(m, 56),
            BitConverter.ToSingle(m, 60)
        );
    }
}

[Serializable]
public class InvalidIPConfiguration : Exception
{
    public InvalidIPConfiguration() : base() { }
    public InvalidIPConfiguration(string message) : base(message) { }
    public InvalidIPConfiguration(string message, Exception inner) : base(message, inner) { }

    // A constructor is needed for serialization when an
    // exception propagates from a remoting server to the client.
    protected InvalidIPConfiguration(System.Runtime.Serialization.SerializationInfo info,
        System.Runtime.Serialization.StreamingContext context) : base(info, context) { }
}