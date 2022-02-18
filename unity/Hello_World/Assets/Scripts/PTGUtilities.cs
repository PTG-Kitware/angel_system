using System;
using System.Net;
using System.Net.NetworkInformation;

public static class PTGUtilities
{
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