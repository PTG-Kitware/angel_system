using UnityEditor;
using UnityEditorInternal;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    [CustomEditor(typeof(CircleArrowPath), true)]
    [CanEditMultipleObjects]
    public class CircleArrowPathEditor : ArrowPathEditor
    {
        CircleArrowPath arrowPath;
        bool needsRepaint;

        void OnEnable()
        {
            SceneView.duringSceneGui += OnSceneGUI;
            arrowPath = target as CircleArrowPath;
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
            EditorGUILayout.PropertyField(serializedObject.FindProperty("center"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("radius"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("axis"));
            if (GUILayout.Button("Axis correction"))
            {
                Undo.RecordObject(arrowPath, "Record");
                arrowPath.axis = Quaternion.LookRotation(arrowPath.radius,arrowPath.axis) * Vector3.up * arrowPath.axis.magnitude;
            }
            EditorGUILayout.PropertyField(serializedObject.FindProperty("levelOfDetail"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("startAngle"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("endAngle"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("upDirectionRollAngle"));
            ObstacleSettings();
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
                Handles.CubeHandleCap(1, p + r * arrowPath.center, Quaternion.identity, HandleUtility.GetHandleSize(p + r * arrowPath.center) * pickSize, EventType.Repaint);
                arrowPath.center = Quaternion.Inverse(r) * (Handles.DoPositionHandle(p + r * arrowPath.center, Quaternion.identity) - p);
                arrowPath.center = SmartArrowUtilities.Utilities.RoundVector(arrowPath.center);
                Handles.color = Color.red;
                Handles.DrawLine(p + r * arrowPath.center, r * arrowPath.center + p + r * arrowPath.radius, 2);
                Handles.color = Color.cyan;
                Handles.DrawLine(p + r * arrowPath.center,  r * arrowPath.center + p + r * arrowPath.axis, 2);
                arrowPath.radius = Quaternion.Inverse(r) * (Handles.DoPositionHandle(p + r * (arrowPath.radius + arrowPath.center), Quaternion.identity) - p) - arrowPath.center;
                arrowPath.radius = SmartArrowUtilities.Utilities.RoundVector(arrowPath.radius);
                arrowPath.axis = Quaternion.Inverse(r) * (Handles.DoPositionHandle(p + r * (arrowPath.axis + arrowPath.center), Quaternion.identity) - p) - arrowPath.center;
                arrowPath.axis = SmartArrowUtilities.Utilities.RoundVector(arrowPath.axis);
                EditorGUI.EndChangeCheck();
            }
        }
    }
}