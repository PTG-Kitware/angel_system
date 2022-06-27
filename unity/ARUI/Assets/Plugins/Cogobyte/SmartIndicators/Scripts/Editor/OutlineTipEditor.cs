using System.Collections.Generic;
using UnityEditor;
using UnityEditorInternal;
using UnityEngine;

//Editor For ArrowPath Asset
namespace Cogobyte.SmartProceduralIndicators
{
    [CustomEditor(typeof(OutlineTip))]
    [CanEditMultipleObjects]
    public class OutlineTipEditor : Editor
    {
        internal int m_tab = 0;
        internal OutlineTip m_tip;
        internal static bool showOutline,showEnding = false;
        internal OutlineEditorItem m_simpleOutlineEditorItem = new OutlineEditorItem();

        void OnEnable()
        {
            m_tip = target as OutlineTip;
            m_simpleOutlineEditorItem.outline = m_tip.outline;
            m_simpleOutlineEditorItem.editor = this;
            m_simpleOutlineEditorItem.serializedObject = serializedObject;
            m_simpleOutlineEditorItem.undoObject = m_tip;
            m_simpleOutlineEditorItem.outlineProperty = serializedObject.FindProperty("outline");
            m_simpleOutlineEditorItem.EnableLists();
        }
        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            EditorGUI.BeginChangeCheck();
            EditorGUILayout.PropertyField(serializedObject.FindProperty("size"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("endPointSize"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("followPath"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("colorMode"));
            if (m_tip.colorMode == OutlineTip.OutlineTipColorMode.TwoColorGradient)
            {
                EditorGUILayout.PropertyField(serializedObject.FindProperty("startPointColor"));
                EditorGUILayout.PropertyField(serializedObject.FindProperty("endPointColor"));
            }
            m_simpleOutlineEditorItem.OnGUI();
            EditorGUI.EndChangeCheck();
            serializedObject.ApplyModifiedProperties();
        }
    }
}