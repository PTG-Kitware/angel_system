using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

namespace Cogobyte.SmartProceduralIndicators
{
    [CustomEditor(typeof(OutlineTemplate), true)]
    public class OutlineTemplateEditor : Editor
    {
        OutlineEditorItem m_simpleOutlineEditorItem = new OutlineEditorItem();
        OutlineTemplate outlineTemplate;

        void OnEnable()
        {
            outlineTemplate = target as OutlineTemplate;
            m_simpleOutlineEditorItem.outline = outlineTemplate.outline;
            m_simpleOutlineEditorItem.editor = this;
            m_simpleOutlineEditorItem.serializedObject = serializedObject;
            m_simpleOutlineEditorItem.undoObject = outlineTemplate;
            m_simpleOutlineEditorItem.outlineProperty = serializedObject.FindProperty("outline");
            m_simpleOutlineEditorItem.EnableLists();
        }
        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            EditorGUI.BeginChangeCheck();
            m_simpleOutlineEditorItem.OnGUI();
            EditorGUI.EndChangeCheck();
            serializedObject.ApplyModifiedProperties();
        }
    }
}