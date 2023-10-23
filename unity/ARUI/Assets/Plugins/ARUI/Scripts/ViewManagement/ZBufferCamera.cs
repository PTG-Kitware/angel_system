using UnityEngine;
using UnityEngine.UI;

public class ZBufferCamera :MonoBehaviour
{
    private Camera _camera;

    public void Start()
    {
        _camera = GetComponent<Camera>();

        _camera.clearFlags = CameraClearFlags.Color;
        _camera.backgroundColor = Color.black;
        _camera.cullingMask = 1 << Utils.GetLayerInt(StringResources.zBuffer_layer);

        _camera.nearClipPlane = 0.1f;
        _camera.targetDisplay = 1;
        _camera.targetTexture = new RenderTexture(Resources.Load(StringResources.zBufferTexture_path) as RenderTexture);
        _camera.allowHDR = false;
        _camera.allowMSAA = false;

        _camera.nearClipPlane = AngelARUI.Instance.ARCamera.nearClipPlane;

        _camera.gameObject.AddComponent<ProcessObjectVisibility>();        
        
        //RawImage image =GameObject.Find("DBTEST").GetComponent<RawImage>();
        //image.texture = _camera.targetTexture;
    }

    public void Update()
    {
        _camera.fieldOfView = AngelARUI.Instance.ARCamera.fieldOfView;
    }

}