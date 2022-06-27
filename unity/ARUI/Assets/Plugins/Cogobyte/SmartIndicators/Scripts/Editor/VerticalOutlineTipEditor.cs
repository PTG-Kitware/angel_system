using System.Collections.Generic;
using UnityEditor;
using UnityEditorInternal;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    [CustomEditor(typeof(VerticalOutlineTip))]
    [CanEditMultipleObjects]
    public class VerticalOutlineTipEditor : Editor
    {
        internal VerticalOutlineTip m_tip;
        internal static bool showOutline, showEnding = false;
        internal OutlineEditorItem m_simpleOutlineEditorItem = new OutlineEditorItem();
        internal ReorderableList m_backFaceMeshesList;

        void OnEnable()
        {
            m_tip = target as VerticalOutlineTip;
            m_simpleOutlineEditorItem.outline = m_tip.outline;
            m_simpleOutlineEditorItem.editor = this;
            m_simpleOutlineEditorItem.serializedObject = serializedObject;
            m_simpleOutlineEditorItem.undoObject = m_tip;
            m_simpleOutlineEditorItem.outlineProperty = serializedObject.FindProperty("outline");
            m_simpleOutlineEditorItem.EnableLists();
            m_backFaceMeshesList = new ReorderableList(serializedObject, serializedObject.FindProperty("backFaceMeshes"), true, true, true, true);
            SmartArrowEditor.GenerateMeshList(this, m_backFaceMeshesList, m_tip.backFaceMeshes, "Back Face Mesh");
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            EditorGUI.BeginChangeCheck();
            EditorGUILayout.PropertyField(serializedObject.FindProperty("size"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("lengthScale"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("colorMode"));
            m_simpleOutlineEditorItem.OnGUI();
            GUILayout.BeginHorizontal(EditorStyles.toolbar);
            EditorGUILayout.LabelField("Mesh Tip Settings", EditorStyles.boldLabel);
            if (GUILayout.Button("Actions", EditorStyles.toolbarDropDown))
            {
                GenericMenu toolsMenu = new GenericMenu();
                toolsMenu.AddItem(new GUIContent("Generate Back Face From Outline"), false, () =>
                {
                    string assetName = EditorUtility.OpenFilePanel("Choose Outline", "", "asset");
                    assetName = assetName.Replace(Application.dataPath, "Assets");
                    if (assetName.Length != 0)
                    {
                        Undo.RecordObject(target, "Generate Outline");
                        OutlineTemplate o = (OutlineTemplate)AssetDatabase.LoadAssetAtPath(assetName, typeof(OutlineTemplate));
                        MeshItem meshItem = new MeshItem();
                        Outline.GenerateFaceMesh(meshItem,o.outline, false);
                        m_tip.backFaceMeshes.Add(meshItem);
                    }
                });
                toolsMenu.DropDown(new Rect(Event.current.mousePosition.x - 100, Event.current.mousePosition.y, 0, 16));
                GUIUtility.ExitGUI();
            }
            GUILayout.EndHorizontal();
            m_backFaceMeshesList.DoLayoutList();
            EditorGUI.EndChangeCheck();
            serializedObject.ApplyModifiedProperties();
        }
    }
}