using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.SpatialAwareness;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Windows.WebCam;
using DilmerGames.Core.Singletons;
using TMPro;
using System.Runtime.InteropServices;




public class RenderBox : MonoBehaviour
{
    // Network stuff
    System.Net.Sockets.TcpClient tcpClient;
    NetworkStream tcpStream;

    Texture2D targetTexture = null;
    GameObject loggerObject = null;

    // Camera stuff
    PhotoCapture photoCaptureObject = null;
    Camera raycastCamera;
    Matrix4x4 projectionMatrix;
    Matrix4x4 cameraToWorldMatrix;
    Resolution cameraResolution;
    Vector3 cameraFramePosition;

    // Spatial awareness stuff
    IEnumerable<SpatialAwarenessMeshObject> meshes;
    IMixedRealitySpatialAwarenessMeshObserver observer = null;

    // Start is called before the first frame update
    void Start()
    {
        this.loggerObject = GameObject.Find("Logger");

        // Connect to the python TCP server
        this.tcpClient = new System.Net.Sockets.TcpClient();
        try
        {
            this.loggerObject.GetComponent<Logger>().LogInfo("Attempting to connect to TCP socket @ IP address 192.168.1.89");
            this.tcpClient.Connect("192.168.1.89", 11000);
            this.loggerObject.GetComponent<Logger>().LogInfo("TCP client connected!");
            this.tcpStream = this.tcpClient.GetStream();

            // Start TCP client receive co-routine
            IEnumerator coroutine = ReceiveMessages();
            StartCoroutine(coroutine);
        }
        catch (Exception e)
        {
            this.loggerObject.GetComponent<Logger>().LogInfo(e.ToString());
        }
    }

    void Update()
    {
        // Setup the spatial awareness observer
        if (observer == null)
        {
            var meshObservers = (CoreServices.SpatialAwarenessSystem as IMixedRealityDataProviderAccess).GetDataProviders<IMixedRealitySpatialAwarenessMeshObserver>();
            foreach (var observers in meshObservers)
            {
                if (observers.Meshes.Count != 0)
                {
                    observer = observers;
                    observer.DisplayOption = SpatialAwarenessMeshDisplayOptions.None;
                    //observer.LevelOfDetail = SpatialAwarenessMeshLevelOfDetail.Unlimited;
                    //observer.UpdateInterval = 0.5f;
                    this.loggerObject.GetComponent<Logger>().LogInfo("Detail level: " + observer.LevelOfDetail.ToString());
                    this.loggerObject.GetComponent<Logger>().LogInfo("Update interval: " + observer.UpdateInterval.ToString());
                }
            }
        }
    }

    public void DrawBox(Vector2 cornerBotLeft, Vector2 cornerTopLeft,
                        Vector2 cornerTopRight, Vector2 cornerBotRight,
                        string classTypeStr)
    {
        this.loggerObject.GetComponent<Logger>().LogInfo("Pre-scaled coords " +
                                                  cornerBotLeft.ToString() + " " +
                                                  cornerTopLeft.ToString() + " " +
                                                  cornerTopRight.ToString() + " " +
                                                  cornerBotRight.ToString());

        Vector3 botLeftDir = PixelCoordToWorldCoord(cornerBotLeft);
        Vector3 topLeftDir = PixelCoordToWorldCoord(cornerTopLeft);
        Vector3 topRightDir = PixelCoordToWorldCoord(cornerTopRight);
        Vector3 botRightDir = PixelCoordToWorldCoord(cornerBotRight);

        RaycastHit botLeftRayHit, topLeftRayHit, topRightRayHit, botRightRayHit;
        if (Physics.Raycast(cameraFramePosition, botLeftDir, out botLeftRayHit, 15.0f))
        {
            this.loggerObject.GetComponent<Logger>().LogInfo("Hit bot left: " + botLeftRayHit.point.ToString());
        }
        if (Physics.Raycast(cameraFramePosition, topLeftDir, out topLeftRayHit, 15.0f))
        {
            this.loggerObject.GetComponent<Logger>().LogInfo("Hit top left: " + topLeftRayHit.point.ToString());
        }
        if (Physics.Raycast(cameraFramePosition, topRightDir, out topRightRayHit, 15.0f))
        {
            this.loggerObject.GetComponent<Logger>().LogInfo("Hit top right: " + topRightRayHit.point.ToString());
        }
        if (Physics.Raycast(cameraFramePosition, botRightDir, out botRightRayHit, 15.0f))
        {
            this.loggerObject.GetComponent<Logger>().LogInfo("Hit bot right: " + botRightRayHit.point.ToString());
        }

        // Draw the box
        Color color = new Color(0.3f, 0.4f, 0.6f);

        GameObject lineObject = new GameObject();
        LineRenderer line = lineObject.AddComponent<LineRenderer>();
        line.startWidth = 0.01f;
        line.endWidth = 0.01f;
        line.startColor = color;
        line.endColor = color;
        line.positionCount = 5;

        line.SetPosition(0, botLeftRayHit.point);
        line.SetPosition(1, topLeftRayHit.point);
        line.SetPosition(2, topRightRayHit.point);
        line.SetPosition(3, botRightRayHit.point);
        line.SetPosition(4, botLeftRayHit.point);
        line.enabled = true;

        // Draw the label
        GameObject textObject = new GameObject();
        TextMesh textMesh = textObject.AddComponent<TextMesh>();
        MeshRenderer meshRenderer = textObject.GetComponent<MeshRenderer>();
        textMesh.color = Color.blue;
        textMesh.transform.localScale = new Vector3(0.04f, 0.04f, 0.04f);

        textMesh.text = classTypeStr;
        Vector3 textPos = botLeftRayHit.point;
        textPos.y = botLeftRayHit.point.y + 0.1f;
        textMesh.transform.position = textPos;
    }

    /// <summary>
    /// Captures a screenshot of the game screen and returns a list of bytes
    /// representing the RGBA values for each pixel
    /// </summary>
    /// <returns>byte[] pixel RGBA values </returns>
    public void StartPhotoCapture()
    {
        cameraResolution = PhotoCapture.SupportedResolutions.OrderByDescending((res) => res.width * res.height).First();
        targetTexture = new Texture2D(cameraResolution.width, cameraResolution.height, TextureFormat.RGBA32, false);

        PhotoCapture.CreateAsync(false, delegate (PhotoCapture captureObject) {
            photoCaptureObject = captureObject;

            CameraParameters c = new CameraParameters();
            c.hologramOpacity = 0.0f;
            c.cameraResolutionWidth = targetTexture.width;
            c.cameraResolutionHeight = targetTexture.height;
            c.pixelFormat = CapturePixelFormat.BGRA32;

            // Start the photo mode and start taking pictures
            photoCaptureObject.StartPhotoModeAsync(c, delegate (PhotoCapture.PhotoCaptureResult result) {
                InvokeRepeating("OnCapturedPhotoToMemory", 0, 5.0f);
            });

            /*
            captureObject.StartPhotoModeAsync(c, delegate (PhotoCapture.PhotoCaptureResult result) {
                photoCaptureObject.TakePhotoAsync(OnCapturedPhotoToMemory);
            });
            */
        });
        
    }

    private void OnCapturedPhotoToMemory()
    {
        if (photoCaptureObject != null)
        {
            // Take a picture
            photoCaptureObject.TakePhotoAsync(delegate (PhotoCapture.PhotoCaptureResult result, PhotoCaptureFrame photoCaptureFrame) {
                List<byte> imageBufferList = new List<byte>();
                List<byte> screenshotBytes = new List<byte>();

                photoCaptureFrame.CopyRawImageDataIntoBuffer(imageBufferList);

                photoCaptureFrame.TryGetProjectionMatrix(Camera.main.nearClipPlane, Camera.main.farClipPlane, out projectionMatrix);
                photoCaptureFrame.TryGetCameraToWorldMatrix(out cameraToWorldMatrix);

                // Convert to 3D-position
                cameraFramePosition = cameraToWorldMatrix.MultiplyPoint(Vector3.zero);

                // Prepend width and length
                screenshotBytes.Add((byte)((cameraResolution.width & 0xFF000000) >> 24));
                screenshotBytes.Add((byte)((cameraResolution.width & 0x00FF0000) >> 16));
                screenshotBytes.Add((byte)((cameraResolution.width & 0x0000FF00) >> 8));
                screenshotBytes.Add((byte)((cameraResolution.width & 0x000000FF) >> 0));
                screenshotBytes.Add((byte)((cameraResolution.height & 0xFF000000) >> 24));
                screenshotBytes.Add((byte)((cameraResolution.height & 0x00FF0000) >> 16));
                screenshotBytes.Add((byte)((cameraResolution.height & 0x0000FF00) >> 8));
                screenshotBytes.Add((byte)((cameraResolution.height & 0x000000FF) >> 0));

                for (int i = imageBufferList.Count - 1; i >= 0; i -= 4)
                {
                    screenshotBytes.Add(imageBufferList[i - 1]); // Red component
                    screenshotBytes.Add(imageBufferList[i - 2]); // Green component
                    screenshotBytes.Add(imageBufferList[i - 3]); // Blue component
                }

                byte[] screenshotBytesArray = AddMessageHeader(screenshotBytes.ToArray());

                // Send the data through the socket.  
                this.tcpStream.Write(screenshotBytesArray, 0, screenshotBytesArray.Length);
                this.tcpStream.Flush();
            });
        }
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
        //Debug.Log(String.Format("Adding sync and length marker. Message length = {0}", message.Length));
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


    private IEnumerator ReceiveMessages()
    {
        while (true)
        {
            // Check if there is data to read and read it if there is
            if (this.tcpStream.DataAvailable)
            {
                byte[] readBuffer = new byte[1024];
                int numberOfBytesRead = 0;

                do
                {
                    numberOfBytesRead = this.tcpStream.Read(readBuffer, 0, readBuffer.Length);
                }
                while (this.tcpStream.DataAvailable);

                for (int i = 0; i < numberOfBytesRead; i += 20)
                {
                    byte[] val1 = new byte[4];
                    byte[] val2 = new byte[4];
                    byte[] val3 = new byte[4];
                    byte[] val4 = new byte[4];
                    byte[] val5 = new byte[4];

                    Array.Copy(readBuffer, i, val1, 0, 4);
                    Array.Copy(readBuffer, i + 4, val2, 0, 4);
                    Array.Copy(readBuffer, i + 8, val3, 0, 4);
                    Array.Copy(readBuffer, i + 12, val4, 0, 4);
                    Array.Copy(readBuffer, i + 16, val5, 0, 4);

                    uint objectType = System.BitConverter.ToUInt32(val1, 0);
                    float minVertex0 = System.BitConverter.ToSingle(val2, 0);
                    float minVertex1 = System.BitConverter.ToSingle(val3, 0);
                    float maxVertex0 = System.BitConverter.ToSingle(val4, 0);
                    float maxVertex1 = System.BitConverter.ToSingle(val5, 0);

                    string classTypeStr = ConvertClassNumToStr(objectType);

                    // Draw the box
                    Vector2 cornerBotLeft = new Vector2(minVertex0, minVertex1);
                    Vector2 cornerTopLeft = new Vector2(minVertex0, maxVertex1);
                    Vector2 cornerTopRight = new Vector2(maxVertex0, maxVertex1);
                    Vector2 cornerBotRight = new Vector2(maxVertex0, minVertex1);

                    this.loggerObject.GetComponent<Logger>().LogInfo("Raw vectors " +
                                                                      cornerBotLeft.ToString() + " " +
                                                                      cornerTopLeft.ToString() + " " +
                                                                      cornerTopRight.ToString() + " " +
                                                                      cornerBotRight.ToString());

                    // Flip the x components since pictures are mirrored on HoloLens
                    cornerBotRight.x = cameraResolution.width - 1 - cornerBotRight.x;
                    cornerTopRight.x = cameraResolution.width - 1 - cornerTopRight.x;
                    cornerTopLeft.x = cameraResolution.width - 1 - cornerTopLeft.x;
                    cornerBotLeft.x = cameraResolution.width - 1 - cornerBotLeft.x;
                    
                    DrawBox(cornerBotLeft, cornerTopLeft, cornerTopRight, cornerBotRight, classTypeStr);
                }
            }
            yield return new WaitForSeconds(0.05f);
        }
    }

    public string ConvertClassNumToStr(uint objectType)
    {
        var conversionDict = new Dictionary<string, string>()
        {
           {"0", "unlabeled"},
           {"1", "person"},
           {"2", "bicycle"},
           {"3", "car"},
           {"4", "motorcycle"},
           {"5", "airplane"},
           {"6", "bus"},
           {"7", "train"},
           {"8", "truck"},
           {"9", "boat"},
           {"10", "traffic light"},
           {"11", "fire hydrant"},
           {"12", "street sign"},
           {"13", "stop sign"},
           {"14", "parking meter"},
           {"15", "bench"},
           {"16", "bird"},
           {"17", "cat"},
           {"18", "dog"},
           {"19", "horse"},
           {"20", "sheep"},
           {"21", "cow"},
           {"22", "elephant"},
           {"23", "bear"},
           {"24", "zebra"},
           {"25", "giraffe"},
           {"26", "hat"},
           {"27", "backpack"},
           {"28", "umbrella"},
           {"29", "shoe"},
           {"30", "eye glasses"},
           {"31", "handbag"},
           {"32", "tie"},
           {"33", "suitcase"},
           {"34", "frisbee"},
           {"35", "skis"},
           {"36", "snowboard"},
           {"37", "sports ball"},
           {"38", "kite"},
           {"39", "baseball bat"},
           {"40", "baseball glove"},
           {"41", "skateboard"},
           {"42", "surfboard"},
           {"43", "tennis racket"},
           {"44", "bottle"},
           {"45", "plate"},
           {"46", "wine glass"},
           {"47", "cup"},
           {"48", "fork"},
           {"49", "knife"},
           {"50", "spoon"},
           {"51", "bowl"},
           {"52", "banana"},
           {"53", "apple"},
           {"54", "sandwich"},
           {"55", "orange"},
           {"56", "broccoli"},
           {"57", "carrot"},
           {"58", "hot dog"},
           {"59", "pizza"},
           {"60", "donut"},
           {"61", "cake"},
           {"62", "chair"},
           {"63", "couch"},
           {"64", "potted plant"},
           {"65", "bed"},
           {"66", "mirror"},
           {"67", "dining table"},
           {"68", "window"},
           {"69", "desk"},
           {"70", "toilet"},
           {"71", "door"},
           {"72", "tv"},
           {"73", "laptop"},
           {"74", "mouse"},
           {"75", "remote"},
           {"76", "keyboard"},
           {"77", "cell phone"},
           {"78", "microwave"},
           {"79", "oven"},
           {"80", "toaster"},
           {"81", "sink"},
           {"82", "refrigerator"},
           {"83", "blender"},
           {"84", "book"},
           {"85", "clock"},
           {"86", "vase"},
           {"87", "scissors"},
           {"88", "teddy bear"},
           {"89", "hair drier"},
           {"90", "toothbrush"},
           {"91", "hair brush"},
           {"92", "banner"},
           {"93", "blanket"},
           {"94", "branch"},
           {"95", "bridge"},
           {"96", "building - other"},
           {"97", "bush"},
           {"98", "cabinet"},
           {"99", "cage"}
        };

        return conversionDict[objectType.ToString()];

    }

    // Taken from https://github.com/VulcanTechnologies/HoloLensCameraStream
    public Vector3 PixelCoordToWorldCoord(Vector2 pixelCoordinates)
    {
        pixelCoordinates = ConvertPixelCoordsToScaledCoords(pixelCoordinates); // -1 to 1 coords

        float focalLengthX = projectionMatrix.GetColumn(0).x;
        float focalLengthY = projectionMatrix.GetColumn(1).y;
        float centerX = projectionMatrix.GetColumn(2).x;
        float centerY = projectionMatrix.GetColumn(2).y;

        // On Microsoft Webpage the centers are normalized 
        float normFactor = projectionMatrix.GetColumn(2).z;
        centerX = centerX / normFactor;
        centerY = centerY / normFactor;

        Vector3 dirRay = new Vector3((pixelCoordinates.x - centerX) / focalLengthX,
                                     (pixelCoordinates.y - centerY) / focalLengthY,
                                      1.0f / normFactor); // Direction is in camera space

        Vector3 direction = new Vector3(Vector3.Dot(cameraToWorldMatrix.GetRow(0), dirRay),
                                        Vector3.Dot(cameraToWorldMatrix.GetRow(1), dirRay),
                                        Vector3.Dot(cameraToWorldMatrix.GetRow(2), dirRay));

        return direction;
    }

    // Taken from https://github.com/VulcanTechnologies/HoloLensCameraStream
    /// <summary>
    /// Converts pixel coordinates to screen-space coordinates that span from -1 to 1 on both axes.
    /// This is the format that is required to determine the z-depth of a given pixel taken by the HoloLens camera.
    /// </summary>
    /// <param name="pixelCoords">The coordinate of the pixel that should be converted to screen-space.</param>
    /// <param name="res">The resolution of the image that the pixel came from.</param>
    /// <returns>A 2D vector with values between -1 and 1, representing the left-to-right scale within the image dimensions.</returns>
    public Vector2 ConvertPixelCoordsToScaledCoords(Vector2 pixelCoords)
    {
        float halfWidth = (float)cameraResolution.width / 2f;
        float halfHeight = (float)cameraResolution.height / 2f;

        // Translate registration to image center;
        pixelCoords.x -= halfWidth;
        pixelCoords.y -= halfHeight;

        // Scale pixel coords to percentage coords (-1 to 1)
        pixelCoords = new Vector2(pixelCoords.x / halfWidth, pixelCoords.y / halfHeight * -1f);

        return pixelCoords;
    }

}