using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using static Cogobyte.SmartProceduralIndicators.BezierArrowPath;

namespace Cogobyte.SmartProceduralIndicators
{
    [CustomEditor(typeof(BezierArrowPath), true)]
    [CanEditMultipleObjects]
    public class BezierArrowPathEditor : ArrowPathEditor
    {
        BezierArrowPath arrowPath;
        private BezierSpline spline;
        private const int stepsPerCurve = 10;
        private const float directionScale = 0.5f;

        void OnEnable()
        {
            SceneView.duringSceneGui += OnSceneGUI;
            arrowPath = target as BezierArrowPath;
            spline = arrowPath.bezierSpline;
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
            EditorGUILayout.PropertyField(serializedObject.FindProperty("upDirectionType"));
            if (arrowPath.upDirectionType != BezierArrowPath.UpDirectionMode.RelativeToPoint && arrowPath.upDirectionType != BezierArrowPath.UpDirectionMode.DefineEachPoint) { 
                EditorGUILayout.PropertyField(serializedObject.FindProperty("upDirection"));
            }
            if(arrowPath.upDirectionType == BezierArrowPath.UpDirectionMode.RelativeToPoint || arrowPath.upDirectionType == BezierArrowPath.UpDirectionMode.RelativeToLine)
            {
                EditorGUILayout.PropertyField(serializedObject.FindProperty("upReferencePoint"));
            }
            EditorGUILayout.PropertyField(serializedObject.FindProperty("levelOfDetail"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("bezierSpline"));
            ObstacleSettings();
            EditorGUI.EndChangeCheck();
            serializedObject.ApplyModifiedProperties();
        }
        public override void RenderControls()
        {
            if (renderControls)
            {
                Vector3 p = Vector3.zero;
                Quaternion r = Quaternion.identity;
                if (arrowPath.local)
                {
                    p = arrowPath.transform.position;
                    r = arrowPath.transform.rotation;
                }
                Undo.RecordObject(arrowPath, "Record");
                EditorGUI.BeginChangeCheck();
                if (spline.points.Count < 2) spline.Reset();
                for (int i = 0; i < spline.points.Count; i++)
                {
                    float size = HandleUtility.GetHandleSize(spline.points[i].position);
                    Handles.Label(p + r * spline.points[i].position, spline.points[i].position.ToString());
                    Vector3 point = Quaternion.Inverse(r) * (Handles.DoPositionHandle(p + r * spline.points[i].position, Quaternion.identity) - p);
                    spline.points[i].MovePoint(point);
                    if (i != 0)
                    {
                        if (arrowPath.upDirectionType == BezierArrowPath.UpDirectionMode.DefineEachPoint)
                        {
                            Handles.color = Color.green;
                            Handles.DrawLine(p+r*spline.points[i].position, p+r*spline.points[i].inReferenceUp);
                            Handles.CubeHandleCap(18, p + r * spline.points[i].inReferenceUp, Quaternion.identity, size * handleSize, EventType.Repaint);
                            spline.points[i].inReferenceUp = Quaternion.Inverse(r) * (Handles.DoPositionHandle(p + r * spline.points[i].inReferenceUp, Quaternion.identity) - p);
                            spline.points[i].inReferenceUp = SmartArrowUtilities.Utilities.RoundVector(spline.points[i].inReferenceUp);
                        }
                        Handles.CubeHandleCap(18, p + r * spline.points[i].inTangent, Quaternion.identity, size * handleSize, EventType.Repaint);
                        spline.points[i].MoveInTangentControlPoint(Quaternion.Inverse(r) * (Handles.DoPositionHandle(p + r * spline.points[i].inTangent, Quaternion.identity) - p));
                        Handles.color = Color.cyan;
                        Handles.DrawLine(p + r * spline.points[i].position, p + r * spline.points[i].inTangent);
                    }
                    if (i != spline.points.Count - 1)
                    {
                        if (arrowPath.upDirectionType == BezierArrowPath.UpDirectionMode.DefineEachPoint)
                        {
                            Handles.color = Color.green;
                            Handles.DrawLine(p + r * spline.points[i].position, p + r * spline.points[i].outReferenceUp);
                            Handles.CubeHandleCap(18, p + r * spline.points[i].outReferenceUp, Quaternion.identity, size * handleSize, EventType.Repaint);
                            spline.points[i].outReferenceUp = Quaternion.Inverse(r) * (Handles.DoPositionHandle(p + r * spline.points[i].outReferenceUp, Quaternion.identity) - p);
                            spline.points[i].outReferenceUp = SmartArrowUtilities.Utilities.RoundVector(spline.points[i].outReferenceUp);
                        }
                        Handles.color = Color.cyan;
                        Handles.CubeHandleCap(17, p+r*spline.points[i].outTangent, Quaternion.identity, size * handleSize, EventType.Repaint);
                        spline.points[i].MoveOutTangentControlPoint(Quaternion.Inverse(r) * (Handles.DoPositionHandle(p + r * spline.points[i].outTangent, Quaternion.identity) - p));
                        Handles.DrawLine(p + r * spline.points[i].position, p + r * spline.points[i].outTangent);
                    }
                }
                if (arrowPath.upDirectionType != BezierArrowPath.UpDirectionMode.DefineEachPoint)
                {
                    if (arrowPath.upDirectionType != BezierArrowPath.UpDirectionMode.RelativeToPoint)
                    {
                        if (arrowPath.upDirectionType == BezierArrowPath.UpDirectionMode.RelativeToLine)
                        {
                            arrowPath.upDirection = Quaternion.Inverse(r) * (Handles.DoPositionHandle(p + r * (arrowPath.upDirection + arrowPath.upReferencePoint), Quaternion.identity) - p) - arrowPath.upReferencePoint;
                            Handles.color = Color.green;
                            Handles.DrawLine(p + r * arrowPath.upReferencePoint, p + r * arrowPath.upReferencePoint + arrowPath.upDirection, 3);
                        }
                        else
                        {
                            arrowPath.upDirection = Quaternion.Inverse(r) * (Handles.DoPositionHandle(p + r * (arrowPath.upDirection + arrowPath.bezierSpline.points[0].position), Quaternion.identity) - p) - arrowPath.bezierSpline.points[0].position;
                            Handles.color = Color.green;
                            Handles.DrawLine(p + r * arrowPath.bezierSpline.points[0].position, p + r * arrowPath.bezierSpline.points[0].position + r * arrowPath.upDirection, 3);
                        }
                    }
                    if (arrowPath.upDirectionType == BezierArrowPath.UpDirectionMode.RelativeToPoint || arrowPath.upDirectionType == BezierArrowPath.UpDirectionMode.RelativeToLine)
                    {
                        Handles.color = Color.red;
                        Handles.CubeHandleCap(1, p + r * arrowPath.upReferencePoint, Quaternion.identity, HandleUtility.GetHandleSize(p + r * arrowPath.upReferencePoint) * pickSize, EventType.Repaint);
                        arrowPath.upReferencePoint = Quaternion.Inverse(r) * (Handles.DoPositionHandle(p + r * arrowPath.upReferencePoint, Quaternion.identity) - p);
                    }
                    arrowPath.upDirection = SmartArrowUtilities.Utilities.RoundVector(arrowPath.upDirection);
                    arrowPath.upReferencePoint = SmartArrowUtilities.Utilities.RoundVector(arrowPath.upReferencePoint);
                }
                EditorGUI.EndChangeCheck();
            }
        }
    }
}