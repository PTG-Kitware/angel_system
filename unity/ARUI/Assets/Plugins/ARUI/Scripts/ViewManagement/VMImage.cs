using System.Collections;
using UnityEngine;

public class VMImage : VMControllable
{
    private bool _isMoving = false;
    public bool IsLooking { get; internal set; }

    void Update()
    {
        base.Update();
        transform.LookAt(Camera.main.transform.position);

        if (AngelARUI.Instance.IsVMActiv && !_isMoving)
            StartCoroutine(UpdatePos());
    }

    private IEnumerator UpdatePos()
    {
        Vector3 targetPos = Vector3.zero;
        Rect getBest = ViewManagement.Instance.GetBestEmptyRect(this);
        if (getBest != Rect.zero)
        {
            float depth = Mathf.Min(ARUISettings.TasksMaxDistToUser, (transform.position - AngelARUI.Instance.ARCamera.transform.position).magnitude);
            depth = Mathf.Max(depth, ARUISettings.TasksMinDistToUser);

            Vector3 pivot = new Vector3(getBest.x + getBest.width / 2, getBest.y + getBest.height / 2, depth);
            targetPos = AngelARUI.Instance.ARCamera.ScreenToWorldPoint(pivot);
            _isMoving = true;
        }

        if (targetPos!=Vector3.zero)
        {
            Vector3 startPos = transform.position;
            float elapsedTime = Time.deltaTime;
            while (Vector3.Distance(transform.position, targetPos) > 0.001f)
            {
                transform.position = Vector3.Lerp(startPos, targetPos, (elapsedTime / 1f));
                elapsedTime += Time.deltaTime;
                yield return new WaitForEndOfFrame();
            }

            transform.position = targetPos;
            yield return new WaitForEndOfFrame();

            _isMoving = false;
        }
    }
}
