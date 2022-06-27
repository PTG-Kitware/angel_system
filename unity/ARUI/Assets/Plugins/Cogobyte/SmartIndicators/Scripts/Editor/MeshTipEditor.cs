using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEditorInternal;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    [CustomEditor(typeof(MeshTip))]
    [CanEditMultipleObjects]
    public class MeshTipEditor : Editor
    {
        internal MeshTip m_meshTip;
        internal ReorderableList m_mainMeshesList;
        internal ReorderableList m_backFaceMeshesList;

        void OnEnable()
        {
            m_meshTip = target as MeshTip;
            m_mainMeshesList = new ReorderableList(serializedObject, serializedObject.FindProperty("meshes"), true, true, true, true);
            SmartArrowEditor.GenerateMeshList(this, m_mainMeshesList, m_meshTip.meshes, "Mesh");
            m_backFaceMeshesList = new ReorderableList(serializedObject, serializedObject.FindProperty("backFaceMeshes"), true, true, true, true);
            SmartArrowEditor.GenerateMeshList(this, m_backFaceMeshesList, m_meshTip.backFaceMeshes, "Back Face Mesh");
        }
        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            EditorGUI.BeginChangeCheck();
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
                        Undo.RecordObject(target, "Add back face");
                        OutlineTemplate o = (OutlineTemplate) AssetDatabase.LoadAssetAtPath(assetName, typeof(OutlineTemplate));
                        MeshItem meshItem = new MeshItem();
                        Outline.GenerateFaceMesh(meshItem,o.outline, false);
                        m_meshTip.backFaceMeshes.Add(meshItem);
                    }
                });
                toolsMenu.DropDown(new Rect(Event.current.mousePosition.x - 100, Event.current.mousePosition.y, 0, 16));
                GUIUtility.ExitGUI();
            }
            GUILayout.EndHorizontal();
            EditorGUILayout.PropertyField(serializedObject.FindProperty("size"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("lengthScale"));
            m_mainMeshesList.DoLayoutList();
            m_backFaceMeshesList.DoLayoutList();
            EditorGUI.EndChangeCheck();
            serializedObject.ApplyModifiedProperties();
        }
    }
}
