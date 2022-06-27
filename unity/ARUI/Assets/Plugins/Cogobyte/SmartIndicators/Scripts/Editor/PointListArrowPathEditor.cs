using UnityEditor;
using UnityEditorInternal;
using UnityEngine;
namespace Cogobyte.SmartProceduralIndicators
{
    [CustomEditor(typeof(PointListArrowPath), true)]
    [CanEditMultipleObjects]
    public class PointListArrowPathEditor : ArrowPathEditor
    {
        PointListArrowPath arrowPath;
        bool needsRepaint;

        void OnEnable()
        {
            SceneView.duringSceneGui += OnSceneGUI;
            arrowPath = target as PointListArrowPath;
        }
        void OnDisable()
        {
            SceneView.duringSceneGui -= OnSceneGUI;
        }
        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            EditorGUI.BeginChangeCheck();
            PathSettings();
            EditorGUILayout.PropertyField(serializedObject.FindProperty("local"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("customPath"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("upDirectionType"));
            if (arrowPath.upDirectionType != PointListArrowPath.UpDirectionMode.RelativeToPoint)
            {
                EditorGUILayout.PropertyField(serializedObject.FindProperty("upDirection"));
            }
            if (arrowPath.upDirectionType == PointListArrowPath.UpDirectionMode.RelativeToPoint || arrowPath.upDirectionType == PointListArrowPath.UpDirectionMode.RelativeToLine)
            {
                EditorGUILayout.PropertyField(serializedObject.FindProperty("upReferencePoint"));
            }
            ObstacleSettings();
            if (arrowPath.obstacleCheck)
            {
                EditorGUILayout.PropertyField(serializedObject.FindProperty("obstacleCheckLevelOfDetail"));
            }
            EditorGUI.EndChangeCheck();
            serializedObject.ApplyModifiedProperties();
        }
        public override void RenderControls()
        {
            if (renderControls)
            {
                Undo.RecordObject(arrowPath, "Record");
                EditorGUI.BeginChangeCheck();
                Vector3 p = Vector3.zero;
                Quaternion r = Quaternion.identity;
                if (arrowPath.local)
                {
                    p = arrowPath.transform.position;
                    r = arrowPath.transform.rotation;
                }

                for (int i = 0; i < arrowPath.customPath.Count; i++)
                {
                    Handles.color = Color.blue;
                    Handles.CubeHandleCap(1, p + r * arrowPath.customPath[i], Quaternion.identity, HandleUtility.GetHandleSize(p + r * arrowPath.customPath[i]) * pickSize, EventType.Repaint);
                    arrowPath.customPath[i] = Quaternion.Inverse(r) * (Handles.DoPositionHandle(p + r * arrowPath.customPath[i], Quaternion.identity) - p);
                    arrowPath.customPath[i] = SmartArrowUtilities.Utilities.RoundVector(arrowPath.customPath[i]);
                }
                if (arrowPath.upDirectionType != PointListArrowPath.UpDirectionMode.RelativeToPoint)
                {
                    if(arrowPath.upDirectionType == PointListArrowPath.UpDirectionMode.RelativeToLine)
                    {
                        arrowPath.upDirection = Quaternion.Inverse(r) * (Handles.DoPositionHandle(p + r * (arrowPath.upDirection + arrowPath.upReferencePoint), Quaternion.identity) - p) - arrowPath.upReferencePoint;
                        Handles.color = Color.green;
                        Handles.DrawLine(p+r*arrowPath.upReferencePoint, p+ r * arrowPath.upReferencePoint + arrowPath.upDirection, 3);
                    }
                    else
                    {
                        arrowPath.upDirection = Quaternion.Inverse(r) * (Handles.DoPositionHandle(p + r * (arrowPath.upDirection + arrowPath.customPath[0]), Quaternion.identity) - p) - arrowPath.customPath[0];
                        Handles.color = Color.green;
                        Handles.DrawLine(p + r * arrowPath.customPath[0], p + r * arrowPath.customPath[0] + r*arrowPath.upDirection, 3);
                    }
                }
                if (arrowPath.upDirectionType == PointListArrowPath.UpDirectionMode.RelativeToPoint || arrowPath.upDirectionType == PointListArrowPath.UpDirectionMode.RelativeToLine)
                {
                    Handles.color = Color.red;
                    Handles.CubeHandleCap(1, p + r * arrowPath.upReferencePoint, Quaternion.identity, HandleUtility.GetHandleSize(p + r * arrowPath.upReferencePoint) * pickSize, EventType.Repaint);
                    arrowPath.upReferencePoint = Quaternion.Inverse(r) * (Handles.DoPositionHandle(p + r * arrowPath.upReferencePoint, Quaternion.identity) - p);
                }
                arrowPath.upDirection = SmartArrowUtilities.Utilities.RoundVector(arrowPath.upDirection);
                arrowPath.upReferencePoint = SmartArrowUtilities.Utilities.RoundVector(arrowPath.upReferencePoint);
                EditorGUI.EndChangeCheck();
            }
        }
    }
}