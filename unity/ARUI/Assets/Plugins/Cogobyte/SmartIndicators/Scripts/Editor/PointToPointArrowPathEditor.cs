using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    [CustomEditor(typeof(PointToPointArrowPath), true)]
    [CanEditMultipleObjects]
    public class PointToPointArrowPathEditor : ArrowPathEditor
    {
        PointToPointArrowPath arrowPath;

        void OnEnable()
        {
            SceneView.duringSceneGui += OnSceneGUI;
            arrowPath = target as PointToPointArrowPath;
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
            EditorGUILayout.PropertyField(serializedObject.FindProperty("pointA"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("pointB")); 
            EditorGUILayout.PropertyField(serializedObject.FindProperty("upDirection"));
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
                Handles.color = Color.red;
                Vector3 p = Vector3.zero;
                Quaternion r = Quaternion.identity;
                if (arrowPath.local)
                {
                    p = arrowPath.transform.position;
                    r = arrowPath.transform.rotation;
                }
                Handles.Label(p + r * arrowPath.pointA, "Point A");
                Handles.CubeHandleCap(1, p+ r*arrowPath.pointA, Quaternion.identity, HandleUtility.GetHandleSize(p+r*arrowPath.pointB) * pickSize, EventType.Repaint);
                arrowPath.pointA =  Quaternion.Inverse(r) * (Handles.DoPositionHandle(p+r*arrowPath.pointA, Quaternion.identity) - p);
                arrowPath.pointA = SmartArrowUtilities.Utilities.RoundVector(arrowPath.pointA);
                arrowPath.upDirection = Quaternion.Inverse(r) * (Handles.DoPositionHandle(p + r * (arrowPath.upDirection + arrowPath.pointA), Quaternion.identity) - p) - arrowPath.pointA;
                arrowPath.upDirection = SmartArrowUtilities.Utilities.RoundVector(arrowPath.upDirection);
                Handles.color = Color.green;
                Handles.DrawLine(p + r * arrowPath.pointA, p + r * arrowPath.pointA + r * arrowPath.upDirection, 3);
                Handles.CubeHandleCap(1, p+r*arrowPath.pointB, Quaternion.identity, HandleUtility.GetHandleSize(p+r*arrowPath.pointB) * pickSize, EventType.Repaint);
                Handles.Label(p + r * arrowPath.pointB, "Point B");
                arrowPath.pointB = Quaternion.Inverse(r) * (Handles.DoPositionHandle(p+r*arrowPath.pointB, Quaternion.identity)-p);
                arrowPath.pointB = SmartArrowUtilities.Utilities.RoundVector(arrowPath.pointB);
                EditorGUI.EndChangeCheck();
            }
        }
        
    }
}