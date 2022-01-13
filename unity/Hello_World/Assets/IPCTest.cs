using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.SpatialAwareness;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Windows.WebCam;
using DilmerGames.Core.Singletons;
using TMPro;


public class IPCTest : MonoBehaviour
{
    System.Net.Sockets.TcpClient tcpClient;
    NetworkStream tcpStream;
    PhotoCapture photoCaptureObject = null;
    Texture2D targetTexture = null;
    GameObject loggerObject = null;
    LineRenderer line = null;
    TextMesh textMesh = null;
    MeshRenderer meshRenderer = null;
    Camera cam;
    uint objectType;
    string classTypeStr;
    float minVertex0 = -1.0f;
    float minVertex1 = -1.0f;
    float maxVertex0 = -1.0f;
    float maxVertex1 = -1.0f;
    Matrix4x4 projectionMatrix;
    Matrix4x4 cameraToWorldMatrix;
    Resolution cameraResolution;
    IEnumerable<SpatialAwarenessMeshObject> meshes;
    IMixedRealitySpatialAwarenessMeshObserver observer = null;
    Vector3 cameraFramePosition;

    // Start is called before the first frame update
    void Start()
    {
        this.loggerObject = GameObject.Find("Logger");

        // Test drawing a line
        Color color = new Color(0.3f, 0.4f, 0.6f);

        GameObject lineObject = new GameObject("Line");
        line = lineObject.AddComponent<LineRenderer>();
        line.startWidth = 0.01f;
        line.endWidth = 0.01f;
        line.startColor = color;
        line.endColor = color;
        line.positionCount = 5;
        line.enabled = false;

        GameObject textObject = new GameObject("Text");
        textMesh = textObject.AddComponent<TextMesh>();
        meshRenderer = textObject.GetComponent<MeshRenderer>();
        textMesh.text = "PTG demo";
        textMesh.color = Color.blue;
        textMesh.transform.position = new Vector3(-0.5f, 0.2f, 1f);
        textMesh.transform.localScale = new Vector3(0.04f, 0.04f, 0.04f);

        this.cam = Camera.main;

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
                    //this.loggerObject.GetComponent<Logger>().LogInfo("Num meshes: " + observers.Meshes.Count.ToString());
                    //this.loggerObject.GetComponent<Logger>().LogInfo("Detail level: " + observers.LevelOfDetail.ToString());

                    observer = observers;
                    observer.DisplayOption = SpatialAwarenessMeshDisplayOptions.None;
                    observer.LevelOfDetail = SpatialAwarenessMeshLevelOfDetail.Unlimited;
                    observer.UpdateInterval = 1.0f;
                    this.loggerObject.GetComponent<Logger>().LogInfo("Detail level: " + observer.LevelOfDetail.ToString());
                    this.loggerObject.GetComponent<Logger>().LogInfo("Update interval: " + observer.UpdateInterval.ToString());

                }
            }
        }

        if (minVertex0 != -1.0f && minVertex1 != -1.0f 
            && maxVertex0 != -1.0f && maxVertex1 != -1.0f)
        {
            // flip the x components since pictures are mirrored on HoloLens
            Vector2 cornerBotLeft = new Vector2(minVertex0, minVertex1);
            Vector2 cornerTopLeft = new Vector2(minVertex0, maxVertex1);
            Vector2 cornerTopRight = new Vector2(maxVertex0, maxVertex1);
            Vector2 cornerBotRight = new Vector2(maxVertex0, minVertex1);
            cornerBotRight.x = cameraResolution.width - 1 - cornerBotLeft.x;
            cornerTopRight.x = cameraResolution.width - 1 - cornerTopLeft.x;
            cornerTopLeft.x = cameraResolution.width - 1 - cornerTopRight.x;
            cornerBotLeft.x = cameraResolution.width - 1 - cornerBotRight.x;

            DrawBox(cornerBotLeft, cornerTopLeft, cornerTopRight, cornerBotRight);
        }

        minVertex0 = -1.0f;
    }

    public void DrawBox(Vector2 cornerBotLeft, Vector2 cornerTopLeft,
                        Vector2 cornerTopRight, Vector2 cornerBotRight)
    {
        Vector3 worldCordBotLeft3D = PixelCoordToWorldCoord(cornerBotLeft);
        Vector3 worldCordTopLeft3D = PixelCoordToWorldCoord(cornerTopLeft);
        Vector3 worldCordTopRight3D = PixelCoordToWorldCoord(cornerTopRight);
        Vector3 worldCordBotRight3D = PixelCoordToWorldCoord(cornerBotRight);

        this.loggerObject.GetComponent<Logger>().LogInfo("bot left dir " + worldCordBotLeft3D.ToString());
        this.loggerObject.GetComponent<Logger>().LogInfo("top left dir " + worldCordTopLeft3D.ToString());
        this.loggerObject.GetComponent<Logger>().LogInfo("top right dir " + worldCordTopRight3D.ToString());
        this.loggerObject.GetComponent<Logger>().LogInfo("bot right dir " + worldCordBotRight3D.ToString());

        RaycastHit botLeftRayHit, topLeftRayHit, topRightRayHit, botRightRayHit;
        if (Physics.Raycast(cameraFramePosition, worldCordBotLeft3D, out botLeftRayHit, Mathf.Infinity))
        {
            this.loggerObject.GetComponent<Logger>().LogInfo("Hit bot left: " + botLeftRayHit.point.ToString());
        }
        if (Physics.Raycast(cameraFramePosition, worldCordTopLeft3D, out topLeftRayHit, Mathf.Infinity))
        {
            this.loggerObject.GetComponent<Logger>().LogInfo("Hit top left: " + topLeftRayHit.point.ToString());
        }
        if (Physics.Raycast(cameraFramePosition, worldCordTopRight3D, out topRightRayHit, Mathf.Infinity))
        {
            this.loggerObject.GetComponent<Logger>().LogInfo("Hit top right: " + topRightRayHit.point.ToString());
        }
        if (Physics.Raycast(cameraFramePosition, worldCordBotRight3D, out botRightRayHit, Mathf.Infinity))
        {
            this.loggerObject.GetComponent<Logger>().LogInfo("Hit bot right: " + botRightRayHit.point.ToString());
        }

        /*
        Vector2 detectionCenter = new Vector2();
        detectionCenter.x = (cornerBotLeft.x + cornerTopRight.x) / 2;
        detectionCenter.y = (cornerBotLeft.y + cornerTopRight.y) / 2;

        Vector3 detectionCenterWorld = PixelCoordToWorldCoord(detectionCenter);
        this.loggerObject.GetComponent<Logger>().LogInfo("Object center (pixelcoord function): " + detectionCenterWorld.ToString());

        Vector3 pointScreenToWorld = cam.ScreenToWorldPoint(new Vector3(detectionCenter.x, detectionCenter.y, cam.nearClipPlane));
        this.loggerObject.GetComponent<Logger>().LogInfo("Object center (screen to world): " + pointScreenToWorld.ToString());

        double closestDistance = -1.0;
        float closestZ = 0.0f;
        foreach (SpatialAwarenessMeshObject meshObject in meshes)
        {
            Mesh mesh = meshObject.Filter.mesh;
            Bounds meshBounds = mesh.bounds;

            Vector3 meshCenter = meshBounds.center;
            //this.loggerObject.GetComponent<Logger>().LogInfo("Mesh center: " + meshCenter.ToString());

            if (meshCenter.z < 0.0f)
            {
                continue;
            }

            Vector3 meshScreenPoint = cam.WorldToScreenPoint(meshCenter);
            Vector2 meshScreenPoint2D = new Vector2(meshScreenPoint.x, meshScreenPoint.y);

            //this.loggerObject.GetComponent<Logger>().LogInfo("Submesh count = " + mesh.subMeshCount.ToString());

            //Vector3 closestPoint = meshBounds.ClosestPoint(detectionCenter);
            double distance = ComputeVectorDistance(detectionCenter, meshScreenPoint2D);

            if (closestDistance == -1.0 || (distance < closestDistance))
            {
                closestDistance = distance;
                closestZ = meshCenter.z;
                //this.loggerObject.GetComponent<Logger>().LogInfo("New minimum center" + meshCenter.ToString());
                //this.loggerObject.GetComponent<Logger>().LogInfo("Closest point = " + closestPoint.ToString());
                //this.loggerObject.GetComponent<Logger>().LogInfo("Distance = " + closestDistance.ToString());
            }

            // check submeshes
            for (int i = 0; i < mesh.subMeshCount; i++)
            {
                SubMeshDescriptor subMesh = mesh.GetSubMesh(i);
                Bounds subMeshBounds = subMesh.bounds;

                Vector3 subMeshCenter = subMeshBounds.center;
                Vector3 subMeshScreenPoint = cam.WorldToScreenPoint(subMeshCenter);
                Vector2 subMeshScreenPoint2D = new Vector2(subMeshScreenPoint.x, subMeshScreenPoint.y);

                //Vector3 closestSubMeshPoint = subMeshBounds.ClosestPoint(detectionCenterWorld);
                distance = ComputeVectorDistance(detectionCenter, subMeshScreenPoint2D);

                if (closestDistance == -1.0 || (distance < closestDistance))
                {
                    closestDistance = distance;
                    closestZ = meshCenter.z;
                    //this.loggerObject.GetComponent<Logger>().LogInfo("New minimum center" + meshCenter.ToString());
                    //this.loggerObject.GetComponent<Logger>().LogInfo("Closest sub point = " + closestPoint.ToString());
                    //this.loggerObject.GetComponent<Logger>().LogInfo("Distance submesh = " + closestDistance.ToString());
                }
            }
        }

        worldCordBotLeft3D.z = closestZ;
        worldCordTopLeft3D.z = closestZ;
        worldCordTopRight3D.z = closestZ;
        worldCordBotRight3D.z = closestZ;
        
        */

        //this.loggerObject.GetComponent<Logger>().LogInfo("Drawing box at " + worldCordBotLeft3D.ToString()
        //                                                + " " + worldCordTopLeft3D.ToString()
        //                                                + " " + worldCordTopRight3D.ToString()
        //                                                + " " + worldCordBotRight3D.ToString());
        // Draw the box
        line.SetPosition(0, botLeftRayHit.point);
        line.SetPosition(1, topLeftRayHit.point);
        line.SetPosition(2, topRightRayHit.point);
        line.SetPosition(3, botRightRayHit.point);
        line.SetPosition(4, botLeftRayHit.point);
        line.enabled = true;

        // Draw the label
        textMesh.text = classTypeStr;
        Vector3 textPos = botLeftRayHit.point;
        textPos.y = botLeftRayHit.point.y + 0.1f;
        textMesh.transform.position = textPos;
    }

    /// <summary>
    /// Sends screenshot bytes to the Python TCP server
    /// </summary>
    /// <param name="s"></param>
    public void SendIPCMessageSocket(string s)
    {
        //Debug.Log("Sending message " + s);

        //IEnumerator coroutine = CaptureScreenshotAsBytes();
        //StartCoroutine(coroutine);

        CaptureScreenshotAsBytes();
    }

    /// <summary>
    /// Captures a screenshot of the game screen and returns a list of bytes
    /// representing the RGBA values for each pixel
    /// </summary>
    /// <returns>byte[] pixel RGBA values </returns>
    public void CaptureScreenshotAsBytes()
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

            captureObject.StartPhotoModeAsync(c, delegate (PhotoCapture.PhotoCaptureResult result) {
                photoCaptureObject.TakePhotoAsync(OnCapturedPhotoToMemory);
            });
        });
        
    }

    private void OnCapturedPhotoToMemory(PhotoCapture.PhotoCaptureResult result,
                                         PhotoCaptureFrame photoCaptureFrame)
    {
        if (result.success)
        {
            //this.loggerObject.GetComponent<Logger>().LogInfo("Photo frame pixel format = "
            //                                                 + photoCaptureFrame.pixelFormat.ToString());
            List<byte> screenshotBytes = new List<byte>();
            List<byte> imageBufferList = new List<byte>();

            photoCaptureFrame.TryGetProjectionMatrix(Camera.main.nearClipPlane, Camera.main.farClipPlane, out projectionMatrix);
            photoCaptureFrame.TryGetCameraToWorldMatrix(out cameraToWorldMatrix);

            //this.loggerObject.GetComponent<Logger>().LogInfo("Projection matrix = " + projectionMatrix.ToString());
            //this.loggerObject.GetComponent<Logger>().LogInfo("Camera world matrix = " + cameraToWorldMatrix.ToString());

            // Convert to 3D-position
            cameraFramePosition = cameraToWorldMatrix.MultiplyPoint(Vector3.zero);

            //Quaternion rotation = Quaternion.LookRotation(-cameraToWorldMatrix.GetColumn(2),
            //                                              cameraToWorldMatrix.GetColumn(1));
            //this.loggerObject.GetComponent<Logger>().LogInfo("World matrix = " + cameraToWorldMatrix.ToString());
            //this.loggerObject.GetComponent<Logger>().LogInfo("Camera position = " + position.ToString());
            //this.loggerObject.GetComponent<Logger>().LogInfo("Camera rotation = " + rotation.ToString());

            // Capture the meshes at the time of the frame capture
            //this.loggerObject.GetComponent<Logger>().LogInfo("Num meshes: " + observer.Meshes.Count.ToString());
            //meshes = observer.Meshes.Values;

            // Copy the raw IMFMediaBuffer data into our empty byte list.
            photoCaptureFrame.CopyRawImageDataIntoBuffer(imageBufferList);

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

            byte [] screenshotBytesArray = AddMessageHeader(screenshotBytes.ToArray());

            // Send the data through the socket.  
            // TODO: Use coroutine here?
            this.tcpStream.Write(screenshotBytesArray, 0, screenshotBytesArray.Length);
            this.tcpStream.Flush();
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
                //this.loggerObject.GetComponent<Logger>().LogInfo("Stream data available!");
                byte[] readBuffer = new byte[1024];
                int numberOfBytesRead = 0;

                do
                {
                    numberOfBytesRead = this.tcpStream.Read(readBuffer, 0, readBuffer.Length);
                }
                while (this.tcpStream.DataAvailable);

                byte[] val1 = new byte[4];
                byte[] val2 = new byte[4];
                byte[] val3 = new byte[4];
                byte[] val4 = new byte[4];
                byte[] val5 = new byte[4];

                Array.Copy(readBuffer, 0, val1, 0, 4);
                Array.Copy(readBuffer, 4, val2, 0, 4);
                Array.Copy(readBuffer, 8, val3, 0, 4);
                Array.Copy(readBuffer, 12, val4, 0, 4);
                Array.Copy(readBuffer, 16, val5, 0, 4);

                objectType = System.BitConverter.ToUInt32(val1, 0);
                minVertex0 = System.BitConverter.ToSingle(val2, 0);
                minVertex1 = System.BitConverter.ToSingle(val3, 0);
                maxVertex0 = System.BitConverter.ToSingle(val4, 0);
                maxVertex1 = System.BitConverter.ToSingle(val5, 0);

                classTypeStr = ConvertClassNumToStr(objectType);
            }

            yield return new WaitForSeconds(0.25f);
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

    public Vector3 PixelCoordToWorldCoord(Vector2 pixelCoordinates)
    {
        pixelCoordinates = ConvertPixelCoordsToScaledCoords(pixelCoordinates); // -1 to 1 coords

        //this.loggerObject.GetComponent<Logger>().LogInfo("Scaled pixel coords = " + pixelCoordinates.ToString());

        float focalLengthX = projectionMatrix.GetColumn(0).x;
        float focalLengthY = projectionMatrix.GetColumn(1).y;
        float centerX = projectionMatrix.GetColumn(2).x;
        float centerY = projectionMatrix.GetColumn(2).y;

        //this.loggerObject.GetComponent<Logger>().LogInfo("project matrix " + projectionMatrix.ToString());

        // On Microsoft Webpage the centers are normalized 
        float normFactor = projectionMatrix.GetColumn(2).z;
        centerX = centerX / normFactor;
        centerY = centerY / normFactor;

        //this.loggerObject.GetComponent<Logger>().LogInfo("project matrix " + projectionMatrix.ToString());
        //this.loggerObject.GetComponent<Logger>().LogInfo("focal x " + focalLengthX.ToString());
        //this.loggerObject.GetComponent<Logger>().LogInfo("focal y " + focalLengthY.ToString());
        //this.loggerObject.GetComponent<Logger>().LogInfo("center x " + centerX.ToString());
        //this.loggerObject.GetComponent<Logger>().LogInfo("center y " + centerY.ToString());
        //this.loggerObject.GetComponent<Logger>().LogInfo("norm factor " + normFactor.ToString());

        Vector3 dirRay = new Vector3((pixelCoordinates.x - centerX) / focalLengthX,
                                     (pixelCoordinates.y - centerY) / focalLengthY,
                                      1.0f / normFactor); //Direction is in camera space

        //this.loggerObject.GetComponent<Logger>().LogInfo("dir ray " + dirRay.ToString());

        Vector3 direction = new Vector3(Vector3.Dot(cameraToWorldMatrix.GetRow(0), dirRay),
                                        Vector3.Dot(cameraToWorldMatrix.GetRow(1), dirRay),
                                        Vector3.Dot(cameraToWorldMatrix.GetRow(2), dirRay));

        return direction;
    }

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

        //Translate registration to image center;
        pixelCoords.x -= halfWidth;
        pixelCoords.y -= halfHeight;

        //Scale pixel coords to percentage coords (-1 to 1)
        pixelCoords = new Vector2(pixelCoords.x / halfWidth, pixelCoords.y / halfHeight * -1f);

        return pixelCoords;
    }

}