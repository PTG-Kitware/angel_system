using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using UnityEngine.Windows.WebCam;
using System.Linq;
using System.Threading.Tasks;
using System;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Threading;
using DilmerGames.Core.Singletons;
using TMPro;


public class IPCTest : MonoBehaviour
{
    System.Net.Sockets.TcpClient sender;
    PhotoCapture photoCaptureObject = null;
    Texture2D targetTexture = null;
    GameObject loggerObject = null;
    int width = 0;
    int height = 0;

    // Start is called before the first frame update
    void Start()
    {
        this.loggerObject = GameObject.Find("Logger");

        // Connect to the python TCP server
        this.sender = new System.Net.Sockets.TcpClient();
        try
        {
            this.loggerObject.GetComponent<Logger>().LogInfo("Attempting to connect to TCP socket @ IP address 192.168.1.89");
            sender.Connect("192.168.1.89", 11000);
            this.loggerObject.GetComponent<Logger>().LogInfo("TCP client connected!");
        }
        catch (Exception e)
        {
            this.loggerObject.GetComponent<Logger>().LogInfo(e.ToString());
        }
    }

/*
    public void SendIPCMessageMemMapFile(string s)
    {
        Debug.Log("Sending message " + s);

        byte[] screenshotBytes = CaptureScreenshotAsBytes();

        using (var mmf = MemoryMappedFile.OpenExisting("ARandomTag"))
        {
            using (MemoryMappedViewStream stream = mmf.CreateViewStream())
            {
                BinaryWriter writer = new BinaryWriter(stream);
                writer.Write(screenshotBytes);
            }
        }
        
    }
*/

    /// <summary>
    /// Sends screenshot bytes to the Python TCP server
    /// </summary>
    /// <param name="s"></param>
    public void SendIPCMessageSocket(string s)
    {
        Debug.Log("Sending message " + s);
        CaptureScreenshotAsBytes();
    }

    /// <summary>
    /// Captures a screenshot of the game screen and returns a list of bytes
    /// representing the RGBA values for each pixel
    /// </summary>
    /// <returns>byte[] pixel RGBA values </returns>
    public void CaptureScreenshotAsBytes()
    {
        Resolution cameraResolution = PhotoCapture.SupportedResolutions.OrderByDescending((res) => res.width * res.height).First();
        targetTexture = new Texture2D(cameraResolution.width, cameraResolution.height, TextureFormat.RGBA32, false);

        this.width = cameraResolution.width;
        this.height = cameraResolution.height;

        PhotoCapture.CreateAsync(false, delegate (PhotoCapture captureObject) {
            photoCaptureObject = captureObject;

            CameraParameters c = new CameraParameters();
            c.cameraResolutionWidth = targetTexture.width;
            c.cameraResolutionHeight = targetTexture.height;
            c.pixelFormat = CapturePixelFormat.BGRA32;

            captureObject.StartPhotoModeAsync(c, delegate (PhotoCapture.PhotoCaptureResult result) {
                photoCaptureObject.TakePhotoAsync(OnCapturedPhotoToMemory);
            });
        });

        return;
    }

    private void OnCapturedPhotoToMemory(PhotoCapture.PhotoCaptureResult result,
                                         PhotoCaptureFrame photoCaptureFrame)
    {
        if (result.success)
        {
            //this.loggerObject.GetComponent<Logger>().LogInfo("Photo frame pixel format = "
            //                                                 + photoCaptureFrame.pixelFormat.ToString());

            List<byte> imageBufferList = new List<byte>();

            // Copy the raw IMFMediaBuffer data into our empty byte list.
            photoCaptureFrame.CopyRawImageDataIntoBuffer(imageBufferList);

            // Prepend width and length
            List<byte> screenshotBytes = new List<byte>();

            screenshotBytes.Add((byte)((this.width & 0xFF000000) >> 24));
            screenshotBytes.Add((byte)((this.width & 0x00FF0000) >> 16));
            screenshotBytes.Add((byte)((this.width & 0x0000FF00) >> 8));
            screenshotBytes.Add((byte)((this.width & 0x000000FF) >> 0));
            screenshotBytes.Add((byte)((this.height & 0xFF000000) >> 24));
            screenshotBytes.Add((byte)((this.height & 0x00FF0000) >> 16));
            screenshotBytes.Add((byte)((this.height & 0x0000FF00) >> 8));
            screenshotBytes.Add((byte)((this.height & 0x000000FF) >> 0));

            int stride = 4;
            for (int i = imageBufferList.Count - 1; i >= 0; i -= stride)
            {
                //float a = (int)(imageBufferList[i - 0]);
                //float r = (int)(imageBufferList[i - 1]);
                //float g = (int)(imageBufferList[i - 2]);
                //float b = (int)(imageBufferList[i - 3]);

                screenshotBytes.Add(imageBufferList[i - 1]);
                screenshotBytes.Add(imageBufferList[i - 2]);
                screenshotBytes.Add(imageBufferList[i - 3]);
            }

            byte [] screenshotBytesArray = AddMessageHeader(screenshotBytes.ToArray());
            //this.loggerObject.GetComponent<Logger>().LogInfo("Sending screenshot via TCP socket");
            this.loggerObject.GetComponent<Logger>().LogInfo(String.Format("Message length = {0}", screenshotBytesArray.Length));

            // Send the data through the socket.  
            NetworkStream senderStream = this.sender.GetStream();
            senderStream.Write(screenshotBytesArray, 0, screenshotBytesArray.Length);
            senderStream.Flush();
        }
        else
        {
            this.loggerObject.GetComponent<Logger>().LogInfo("Failed to save Photo to memory.");
        }
        this.photoCaptureObject.StopPhotoModeAsync(this.OnStoppedPhotoMode);
    }

    // This method is called when the photo mode is stopped and we can dispose the resources allocated.
    private void OnStoppedPhotoMode(PhotoCapture.PhotoCaptureResult result)
    {
        //this.loggerObject.GetComponent<Logger>().LogInfo("Stopping photo mode");
        this.photoCaptureObject.Dispose();
        this.photoCaptureObject = null;
    }

    /// <summary>
    /// Add a sync marker of 0x1ACFFC1D and a 4 byte length
    /// to the given message
    /// </summary>
    /// <param name="message"></param>
    /// <returns></returns>
    private static byte[] AddMessageHeader(byte[] message)
    {
        Debug.Log(String.Format("Adding sync and length marker. Message length = {0}", message.Length));
        byte[] sync = { 0x1A, 0xCF, 0xFC, 0x1D };
        byte[] length = {(byte)((message.Length & 0xFF000000) >> 24),
                         (byte)((message.Length & 0x00FF0000) >> 16),
                         (byte)((message.Length & 0x0000FF00) >> 8),
                         (byte)((message.Length & 0x000000FF) >> 0)};
        byte[] newMessage = new byte[message.Length + 8]; // 4 byte sync + 4 byte length

        System.Buffer.BlockCopy(sync, 0, newMessage, 0, sync.Length);
        System.Buffer.BlockCopy(length, 0, newMessage, sync.Length, length.Length);
        System.Buffer.BlockCopy(message, 0, newMessage, sync.Length + length.Length, message.Length);

        return newMessage;
    }

}