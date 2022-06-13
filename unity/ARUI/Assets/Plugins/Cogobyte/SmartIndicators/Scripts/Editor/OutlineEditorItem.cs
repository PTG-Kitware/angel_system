using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

namespace Cogobyte.SmartProceduralIndicators
{
    public class OutlineEditorItem
    {
        public Outline outline;
        public Editor editor;
        public SerializedObject serializedObject;
        public SerializedProperty outlineProperty;
        public Object undoObject;
        ReorderableList m_edgeList;
        ReorderableList m_backFaceMeshesList;
        ReorderableList m_frontFaceMeshesList;
        Dictionary<string, ReorderableList> m_pointLists = new Dictionary<string, ReorderableList>();
        static bool hideOutlineSettings = true;
        static bool hideOutlineFaceSettings = true;
        public void EnableLists()
        {
            m_pointLists = new Dictionary<string, ReorderableList>();
            m_edgeList = new ReorderableList(serializedObject, outlineProperty.FindPropertyRelative("edges"), true, true, true, true)
            {
                onCanRemoveCallback = (ReorderableList l) =>
                {
                    return l.count > 1;
                },
                drawHeaderCallback = rect =>
                {
                    EditorGUI.LabelField(rect, "Edges", EditorStyles.boldLabel);
                },
                drawElementCallback = (rect, index, a, h) =>
                {
                    var element = m_edgeList.serializedProperty.GetArrayElementAtIndex(index);
                    var InnerList = element.FindPropertyRelative("points");
                    string listKey = element.propertyPath;
                    ReorderableList innerReorderableList;
                    if (m_pointLists.ContainsKey(listKey))
                    {
                        innerReorderableList = m_pointLists[listKey];
                    }
                    else
                    {
                        innerReorderableList = new ReorderableList(element.serializedObject, InnerList)
                        {
                            displayAdd = true,
                            displayRemove = true,
                            draggable = true,
                            onCanRemoveCallback = (ReorderableList l) =>
                            {
                                return l.count > 2;
                            },
                            drawHeaderCallback = innerRect =>
                            {
                                EditorGUI.LabelField(innerRect, "Edge " + index + " points");
                            },
                            drawElementCallback = (innerRect, innerIndex, innerA, innerH) =>
                            {
                                var innerElement = InnerList.GetArrayElementAtIndex(innerIndex);
                                var r1 = new Rect(innerRect.x, innerRect.y, innerRect.width / 3, EditorGUIUtility.singleLineHeight);
                                var r2 = new Rect(innerRect.x + innerRect.width / 3, innerRect.y, innerRect.width / 3, EditorGUIUtility.singleLineHeight);
                                var r3 = new Rect(innerRect.x + (innerRect.width * 2) / 3 + 5, innerRect.y, innerRect.width / 3 - 5, EditorGUIUtility.singleLineHeight);
                                EditorGUI.PropertyField(r1, innerElement.FindPropertyRelative("position"), GUIContent.none);
                                EditorGUI.PropertyField(r2, innerElement.FindPropertyRelative("normal"), GUIContent.none);
                                EditorGUI.PropertyField(r3, innerElement.FindPropertyRelative("color"), GUIContent.none);
                            }
                        };
                        innerReorderableList.onAddCallback = (ReorderableList l) =>
                        {
                            var InnerIndex = innerReorderableList.serializedProperty.arraySize;
                            innerReorderableList.serializedProperty.arraySize++;
                            innerReorderableList.index = InnerIndex;
                            var innerElement = innerReorderableList.serializedProperty.GetArrayElementAtIndex(InnerIndex);
                            if (InnerIndex > 0)
                            {
                                var lastElement = innerReorderableList.serializedProperty.GetArrayElementAtIndex(InnerIndex - 1);
                                innerElement.FindPropertyRelative("position").vector2Value = lastElement.FindPropertyRelative("position").vector2Value;
                                innerElement.FindPropertyRelative("normal").vector2Value = lastElement.FindPropertyRelative("normal").vector2Value;
                                innerElement.FindPropertyRelative("color").colorValue = lastElement.FindPropertyRelative("color").colorValue;
                            }
                            else
                            {
                                innerElement.FindPropertyRelative("position").vector2Value = Vector2.zero;
                                innerElement.FindPropertyRelative("normal").vector2Value = Vector2.up;
                                innerElement.FindPropertyRelative("color").colorValue = Color.white;
                            }
                        };
                        m_pointLists[listKey] = innerReorderableList;
                    }
                    // Setup the inner list
                    var height = (InnerList.arraySize + 3) * EditorGUIUtility.singleLineHeight;
                    EditorGUI.LabelField(new Rect(rect.x, rect.y, rect.width, EditorGUIUtility.singleLineHeight), "Edge " + index, EditorStyles.boldLabel);
                    outline.edges[index].color = EditorGUI.ColorField(new Rect(rect.width - 100 + rect.x, rect.y + 1, 100, EditorGUIUtility.singleLineHeight), outline.edges[index].color);
                    outline.edges[index].meshIndex = EditorGUI.IntField(new Rect(rect.x, rect.y + 5 + EditorGUIUtility.singleLineHeight, rect.width, EditorGUIUtility.singleLineHeight), "Arrow Mesh", outline.edges[index].meshIndex);
                    innerReorderableList.DoList(new Rect(rect.x, rect.y + 10 + 2 * EditorGUIUtility.singleLineHeight, rect.width, height - EditorGUIUtility.singleLineHeight));
                },
                elementHeightCallback = index =>
                {
                    var element = m_edgeList.serializedProperty.GetArrayElementAtIndex(index);
                    var innerList = element.FindPropertyRelative("points");
                    return (innerList.arraySize + 8) * EditorGUIUtility.singleLineHeight + innerList.arraySize * 5;
                }
            };
            m_backFaceMeshesList = new ReorderableList(serializedObject, outlineProperty.FindPropertyRelative("backFaceMeshes"), true, true, true, true);

            SmartArrowEditor.GenerateMeshList(editor, m_backFaceMeshesList, outline.backFaceMeshes, "Back Face Meshes");

            m_frontFaceMeshesList = new ReorderableList(serializedObject, outlineProperty.FindPropertyRelative("frontFaceMeshes"), true, true, true, true);
            SmartArrowEditor.GenerateMeshList(editor, m_frontFaceMeshesList, outline.frontFaceMeshes, "Front Face Meshes");
        }
        public void OnGUI()
        {
            GUILayout.BeginHorizontal(EditorStyles.toolbar);
            EditorGUILayout.LabelField("Outline Settings", EditorStyles.boldLabel);
            if (GUILayout.Button("Actions", EditorStyles.toolbarDropDown))
            {
                GenericMenu toolsMenu = new GenericMenu();
                toolsMenu.AddItem(new GUIContent("Visual Editor"), false, () =>
                {
                    OutlineEditorWindow.LoadWindow(outline,undoObject);
                });
                toolsMenu.AddItem(new GUIContent("Outline/Save Outline"), false, () =>
                {
                    string assetName = EditorUtility.SaveFilePanelInProject("Save Outline", "Outline", "asset", "Please enter a file name to save the outline to");
                    if (assetName.Length != 0)
                    {
                        OutlineTemplate o = ScriptableObject.CreateInstance<OutlineTemplate>();
                        o.outline = new Outline(outline);
                        AssetDatabase.CreateAsset(o, assetName);
                    }
                });
                toolsMenu.AddItem(new GUIContent("Outline/Load Outline"), false, () =>
                {
                    string assetName = EditorUtility.OpenFilePanel("Load Outline", "", "asset");
                    assetName = assetName.Replace(Application.dataPath, "Assets");
                    if (assetName.Length != 0)
                    {
                        OutlineTemplate o = (OutlineTemplate)UnityEditor.AssetDatabase.LoadAssetAtPath(assetName, typeof(OutlineTemplate));
                        Undo.RecordObject(undoObject, "Simple outline load");
                        outline.CopyOutline(o.outline);
                    }
                    else
                    {
                        EditorUtility.DisplayDialog("Could not load outline", "Wrong scriptable object type", "Ok");
                    }
                });
                toolsMenu.AddItem(new GUIContent("Calculate Normals"), false, () =>
                {
                    Undo.RecordObject(undoObject, "Calculate Normals");
                    for (int i = 0; i < outline.edges.Count; i++)
                    {
                        outline.edges[i].CaluclateNormals();
                    }
                });
                toolsMenu.AddItem(new GUIContent("Calculate Smooth Normals"), false, () =>
                {
                    Undo.RecordObject(undoObject, "Calculate Normals");
                    for (int i = 0; i < outline.edges.Count; i++)
                    {
                        outline.CalculateSmoothNormals();
                    }
                });
                toolsMenu.AddItem(new GUIContent("Hide Outline Settings"), hideOutlineSettings, () =>
                {
                    hideOutlineSettings = !hideOutlineSettings;
                });
                toolsMenu.DropDown(new Rect(Event.current.mousePosition.x-100, Event.current.mousePosition.y, 0, 16));
                GUIUtility.ExitGUI();
            }
            GUILayout.EndHorizontal();
            if (!hideOutlineSettings)
            {
                EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                {
                    outline.color = EditorGUILayout.ColorField("Color", outline.color);
                    m_edgeList.DoLayoutList();
                }
                EditorGUILayout.EndVertical();
            }
            GUILayout.BeginHorizontal(EditorStyles.toolbar);
            EditorGUILayout.LabelField("Outline Face Settings", EditorStyles.boldLabel);
            if (GUILayout.Button("Actions", EditorStyles.toolbarDropDown))
            {
                GenericMenu toolsMenu = new GenericMenu();
                toolsMenu.AddItem(new GUIContent("Generate Back Face"), false, () =>
                {
                    MeshItem meshItem = new MeshItem();
                    Outline.GenerateFaceMesh(meshItem,outline, false);
                    outline.backFaceMeshes.Add(meshItem);
                });
                toolsMenu.AddItem(new GUIContent("Generate Front Face"), false, () =>
                {
                    MeshItem meshItem = new MeshItem();
                    Outline.GenerateFaceMesh(meshItem,outline, true);
                    outline.frontFaceMeshes.Add(meshItem);
                });
                toolsMenu.AddItem(new GUIContent("Hide Outline Face Settings"), hideOutlineFaceSettings, () =>
                {
                    hideOutlineFaceSettings = !hideOutlineFaceSettings;
                });
                // Offset menu from right of editor window
                toolsMenu.DropDown(new Rect(Event.current.mousePosition.x - 20, Event.current.mousePosition.y, 0, 16));
                GUIUtility.ExitGUI();
            }
            GUILayout.EndHorizontal();
            if (!hideOutlineFaceSettings)
            {
                m_backFaceMeshesList.DoLayoutList();
                m_frontFaceMeshesList.DoLayoutList();
            }
        }

    }
}