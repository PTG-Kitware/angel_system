using Cogobyte.SmartProceduralIndicators.SmartArrowUtilities;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEditorInternal;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    [CustomEditor(typeof(OutlineBodyRenderer), true)]
    [CanEditMultipleObjects]
    public class OutlineBodyRendererEditor : Editor
    {
        internal OutlineBodyRenderer m_simpleBodyRenderer;
        internal OutlineEditorItem m_simpleOutlineEditorItem = new OutlineEditorItem();
        internal ReorderableList m_tailMeshesList;
        internal ReorderableList m_headMeshesList;
        internal Dictionary<string, ReorderableList> m_innerVertsLists = new Dictionary<string, ReorderableList>();
        public ReorderableList list;
        static bool hideBodyTipFaceOptions = false;

        void OnEnable()
        {
            m_simpleBodyRenderer = target as OutlineBodyRenderer;
            EnableLists();
        }

        public virtual void EnableLists()
        {
            list = new ReorderableList(serializedObject, serializedObject.FindProperty("colors"), true, true, true, true);
            list.onCanRemoveCallback = (ReorderableList l) =>
            {
                return l.count > 1;
            };
            list.onAddCallback = (ReorderableList l) =>
            {
                var index = l.serializedProperty.arraySize;
                l.serializedProperty.arraySize++;
                l.index = index;
                var element = l.serializedProperty.GetArrayElementAtIndex(index);
                element.FindPropertyRelative("colorMode").intValue = 0;
                element.FindPropertyRelative("length").floatValue = 1;
                element.FindPropertyRelative("percentage").floatValue = 1;
                element.FindPropertyRelative("startColor").colorValue = Color.white;
                element.FindPropertyRelative("endColor").colorValue = Color.white;
            };
            list.drawHeaderCallback = (Rect position) => {
                var amountRect = new Rect(position.x + 18, position.y, (position.width - 115) * 1 / 3, position.height);
                var valRect = new Rect(position.x + 25 + (position.width - 115) * 1 / 3, position.y, (position.width - 115) * 2 / 3, position.height);
                var sColorRect = new Rect(position.x + position.width - 105, position.y, 50, position.height);
                var eColorRect = new Rect(position.x + position.width - 50, position.y, 50, position.height);
                EditorGUI.LabelField(amountRect, "Length type");
                EditorGUI.LabelField(valRect, "Value");
                EditorGUI.LabelField(sColorRect, "Start");
                EditorGUI.LabelField(eColorRect, "End");
            };
            list.drawElementCallback =
            (Rect position, int index, bool isActive, bool isFocused) => {
                var element = list.serializedProperty.GetArrayElementAtIndex(index);
                position.y += 2;
                var indent = EditorGUI.indentLevel;
                EditorGUI.indentLevel = 0;
                var amountRect = new Rect(position.x, position.y, (position.width - 115) * 1 / 3, position.height);
                var valRect = new Rect(position.x + 5 + (position.width - 115) * 1 / 3, position.y, (position.width - 115) * 2 / 3, position.height - 5);
                var sColorRect = new Rect(position.x + position.width - 105, position.y, 50, position.height - 5);
                var eColorRect = new Rect(position.x + position.width - 50, position.y, 50, position.height - 5);
                EditorGUI.PropertyField(amountRect, element.FindPropertyRelative("colorMode"), GUIContent.none);
                var category = element.FindPropertyRelative("colorMode");
                if (category.intValue == (int)PathColor.PathColorMode.Fixed)
                {
                    float tempLW = EditorGUIUtility.labelWidth;
                    EditorGUIUtility.labelWidth = 15;
                    EditorGUI.PropertyField(valRect, element.FindPropertyRelative("length"), new GUIContent("L:"));
                    EditorGUIUtility.labelWidth = tempLW;
                }
                else if (category.intValue == (int)PathColor.PathColorMode.Percentage)
                {
                    float tempLW = EditorGUIUtility.labelWidth;
                    EditorGUIUtility.labelWidth = 15;
                    EditorGUI.PropertyField(valRect, element.FindPropertyRelative("percentage"), new GUIContent("%:"));
                    EditorGUIUtility.labelWidth = tempLW;
                }
                EditorGUI.LabelField(sColorRect, new GUIContent("", "Start Color"));
                EditorGUI.PropertyField(sColorRect, element.FindPropertyRelative("startColor"), GUIContent.none);
                EditorGUI.LabelField(eColorRect, new GUIContent("", "End Color"));
                EditorGUI.PropertyField(eColorRect, element.FindPropertyRelative("endColor"), GUIContent.none);
                EditorGUI.indentLevel = indent;
            };

            m_simpleOutlineEditorItem.outline = m_simpleBodyRenderer.outline;
            m_simpleOutlineEditorItem.editor = this;
            m_simpleOutlineEditorItem.serializedObject = serializedObject;
            m_simpleOutlineEditorItem.undoObject = m_simpleBodyRenderer;
            m_simpleOutlineEditorItem.outlineProperty = serializedObject.FindProperty("outline");
            m_simpleOutlineEditorItem.EnableLists();
            m_tailMeshesList = new ReorderableList(serializedObject, serializedObject.FindProperty("tailMeshConnector"), true, true, true, true);
            GenerateMeshWithInnerVerticesList(m_tailMeshesList, "Body Tail Face Meshes");
            m_headMeshesList = new ReorderableList(serializedObject, serializedObject.FindProperty("headMeshConnector"), true, true, true, true);
            GenerateMeshWithInnerVerticesList(m_headMeshesList, "Body Head Face Meshes");
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            EditorGUI.BeginChangeCheck();
            GenerateUI();
            EditorGUI.EndChangeCheck();
            serializedObject.ApplyModifiedProperties();
        }
        
        internal virtual void GenerateUI()
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("colorMode"));
            bool renderLodField = false;
            if (m_simpleBodyRenderer.colorMode == OutlineBodyRenderer.OutlineBodyRendererColorMode.ColorAlongBody)
            {
                list.DoLayoutList();
            }
            else if(m_simpleBodyRenderer.colorMode == OutlineBodyRenderer.OutlineBodyRendererColorMode.ColorAlongBodyGradient)
            {
                EditorGUILayout.PropertyField(serializedObject.FindProperty("colorsGradient"));
                BodyRenderer.SetColorsToGradient(m_simpleBodyRenderer.colors, m_simpleBodyRenderer.colorsGradient);
            }
            m_simpleBodyRenderer.curveModes[0] = (OutlineBodyRenderer.CurveFunctionMode) EditorGUILayout.EnumPopup("Width Mode", m_simpleBodyRenderer.curveModes[0]);
            m_simpleBodyRenderer.curveModes[1] = (OutlineBodyRenderer.CurveFunctionMode) EditorGUILayout.EnumPopup("Height Mode", m_simpleBodyRenderer.curveModes[1]);
            if (m_simpleBodyRenderer.curveModes[0] == OutlineBodyRenderer.CurveFunctionMode.Fixed && m_simpleBodyRenderer.curveModes[1]== OutlineBodyRenderer.CurveFunctionMode.Fixed)
            {
                EditorGUILayout.PropertyField(serializedObject.FindProperty("size"));
            }
            else if (m_simpleBodyRenderer.curveModes[0] != OutlineBodyRenderer.CurveFunctionMode.Fixed && m_simpleBodyRenderer.curveModes[1] != OutlineBodyRenderer.CurveFunctionMode.Fixed)
            {
                renderLodField = true;
                m_simpleBodyRenderer.curveFunction[0] = EditorGUILayout.CurveField("Width Function", m_simpleBodyRenderer.curveFunction[0]);
                m_simpleBodyRenderer.curveFunction[1] = EditorGUILayout.CurveField("Height Function", m_simpleBodyRenderer.curveFunction[1]);
            }
            else if (m_simpleBodyRenderer.curveModes[0] != OutlineBodyRenderer.CurveFunctionMode.Fixed )
            {
                renderLodField = true;
                m_simpleBodyRenderer.curveFunction[0] = EditorGUILayout.CurveField("Width Function", m_simpleBodyRenderer.curveFunction[0]);
                m_simpleBodyRenderer.size.y = EditorGUILayout.FloatField("Height", m_simpleBodyRenderer.size.y);
            }
            else 
            {
                renderLodField = true;
                m_simpleBodyRenderer.size.x = EditorGUILayout.FloatField("Width", m_simpleBodyRenderer.size.x);
                m_simpleBodyRenderer.curveFunction[1] = EditorGUILayout.CurveField("Height Function", m_simpleBodyRenderer.curveFunction[1]);
            }

            m_simpleBodyRenderer.curveModes[2] = (OutlineBodyRenderer.CurveFunctionMode)EditorGUILayout.EnumPopup("Roll Mode", m_simpleBodyRenderer.curveModes[2]);
            if (m_simpleBodyRenderer.curveModes[2] != OutlineBodyRenderer.CurveFunctionMode.Fixed)
            {
                renderLodField = true;
                m_simpleBodyRenderer.curveFunction[2] = EditorGUILayout.CurveField("Roll Function", m_simpleBodyRenderer.curveFunction[2]);
            }
            else
            {
                EditorGUILayout.PropertyField(serializedObject.FindProperty("roll"));
            }
            if (renderLodField) { 
                EditorGUILayout.PropertyField(serializedObject.FindProperty("lod"));
            }
            EditorGUILayout.PropertyField(serializedObject.FindProperty("smoothPath"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("scaleCorners"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("dashed"));
            if (m_simpleBodyRenderer.dashed)
            {
                EditorGUILayout.PropertyField(serializedObject.FindProperty("emptyLength"));
                EditorGUILayout.PropertyField(serializedObject.FindProperty("dashLength"));
            }
            m_simpleOutlineEditorItem.OnGUI();
            GUILayout.BeginHorizontal(EditorStyles.toolbar);
            EditorGUILayout.LabelField("Body Tip Face Options", EditorStyles.boldLabel);
            if (GUILayout.Button("Actions", EditorStyles.toolbarDropDown))
            {
                GenericMenu toolsMenu = new GenericMenu();
                toolsMenu.AddItem(new GUIContent("Generate Body Tail Face"), false, () =>
                {
                    string assetName = EditorUtility.OpenFilePanel("Outline object", "", "asset");
                    assetName = assetName.Replace(Application.dataPath, "Assets");
                    if (assetName.Length != 0)
                    {
                        OutlineTemplate o = (OutlineTemplate) AssetDatabase.LoadAssetAtPath(assetName, typeof(OutlineTemplate));
                        Outline outline = new Outline(o.outline);
                        Undo.RecordObject(target, "Generate Outline");
                        BodyTipMeshItem meshItem = new BodyTipMeshItem();
                        Outline.GenerateOutlineToOutlineFace(meshItem,outline, m_simpleBodyRenderer.outline, true);
                        m_simpleBodyRenderer.tailMeshConnector.Add(meshItem);

                    }
                });
                toolsMenu.AddItem(new GUIContent("Generate Body Head Face"), false, () =>
                {
                    string assetName = EditorUtility.OpenFilePanel("Outline object", "", "asset");
                    assetName = assetName.Replace(Application.dataPath, "Assets");
                    if (assetName.Length != 0)
                    {
                        OutlineTemplate o = (OutlineTemplate)AssetDatabase.LoadAssetAtPath(assetName, typeof(OutlineTemplate));
                        Undo.RecordObject(target, "Generate Outline");
                        BodyTipMeshItem meshItem = new BodyTipMeshItem();
                        Outline.GenerateOutlineToOutlineFace(meshItem,o.outline, m_simpleBodyRenderer.outline, false);
                        m_simpleBodyRenderer.headMeshConnector.Add(meshItem);
                    }
                });
                toolsMenu.AddItem(new GUIContent("Generate Same Body Tail Outline Face"), false, () =>
                {
                    Undo.RecordObject(target, "Generate Outline");
                    BodyTipMeshItem meshItem = new BodyTipMeshItem();
                    Outline.GenerateOutlineToOutlineFace(meshItem, m_simpleBodyRenderer.outline, m_simpleBodyRenderer.outline, true);
                    m_simpleBodyRenderer.tailMeshConnector.Add(meshItem);
                });
                toolsMenu.AddItem(new GUIContent("Generate Same Body Head Outline Face"), false, () =>
                {
                    Undo.RecordObject(target, "Generate Outline");
                    BodyTipMeshItem meshItem = new BodyTipMeshItem();
                    Outline.GenerateOutlineToOutlineFace(meshItem, m_simpleBodyRenderer.outline, m_simpleBodyRenderer.outline, false);
                    m_simpleBodyRenderer.headMeshConnector.Add(meshItem);
                });
                toolsMenu.AddItem(new GUIContent("Generate Tail Outline Face"), false, () =>
                {
                    string assetName = EditorUtility.OpenFilePanel("Outline object", "", "asset");
                    assetName = assetName.Replace(Application.dataPath, "Assets");
                    if (assetName.Length != 0)
                    {
                        OutlineTemplate o = (OutlineTemplate)AssetDatabase.LoadAssetAtPath(assetName, typeof(OutlineTemplate));
                        Outline outline = new Outline(o.outline);
                        Undo.RecordObject(target, "Generate Outline");
                        BodyTipMeshItem meshItem = new BodyTipMeshItem();
                        Outline.GenerateFaceMesh(meshItem, o.outline, true);
                        m_simpleBodyRenderer.tailMeshConnector.Add(meshItem);

                    }
                });
                toolsMenu.AddItem(new GUIContent("Generate Head Outline Face"), false, () =>
                {
                    string assetName = EditorUtility.OpenFilePanel("Outline object", "", "asset");
                    assetName = assetName.Replace(Application.dataPath, "Assets");
                    if (assetName.Length != 0)
                    {
                        OutlineTemplate o = (OutlineTemplate)AssetDatabase.LoadAssetAtPath(assetName, typeof(OutlineTemplate));
                        Undo.RecordObject(target, "Generate Outline");
                        BodyTipMeshItem meshItem = new BodyTipMeshItem();
                        Outline.GenerateFaceMesh(meshItem, o.outline, false);
                        m_simpleBodyRenderer.headMeshConnector.Add(meshItem);
                    }
                });
                toolsMenu.AddItem(new GUIContent("Hide Body Tip Face Options"), hideBodyTipFaceOptions, () =>
                {
                    hideBodyTipFaceOptions = !hideBodyTipFaceOptions;
                });
                // Offset menu from right of editor window
                toolsMenu.DropDown(new Rect(Event.current.mousePosition.x - 20, Event.current.mousePosition.y, 0, 16));
                GUIUtility.ExitGUI();
            }
            GUILayout.EndHorizontal();
            if (!hideBodyTipFaceOptions)
            {
                m_tailMeshesList.DoLayoutList();
                m_headMeshesList.DoLayoutList();
            }
            EditorGUILayout.EndVertical();
        }

        internal void GenerateMeshWithInnerVerticesList(ReorderableList list, string title)
        {
            list.onAddCallback = (ReorderableList l) =>
            {
                var index = list.serializedProperty.arraySize;
                list.serializedProperty.arraySize++;
                list.index = index;
                var element = list.serializedProperty.GetArrayElementAtIndex(index);
                element.FindPropertyRelative("color").colorValue = Color.gray;
            };
            list.elementHeightCallback = (index) =>
            {
                Repaint();
                var element = list.serializedProperty.GetArrayElementAtIndex(index);
                var innerList = element.FindPropertyRelative("innerVertices");
                if (element.FindPropertyRelative("colorMode").enumValueIndex == (int) MeshItem.MeshItemColorMode.SingleColor)
                {
                    return (innerList.arraySize + 8) * EditorGUIUtility.singleLineHeight;
                }
                return (innerList.arraySize + 7) * EditorGUIUtility.singleLineHeight+ 5*innerList.arraySize;
            };
            list.drawHeaderCallback = (Rect position) =>
            {
                EditorGUI.LabelField(position, title);
            };
            list.drawElementCallback =
            (Rect position, int index, bool isActive, bool isFocused) => {
                var element = list.serializedProperty.GetArrayElementAtIndex(index);
                var InnerList = element.FindPropertyRelative("innerVertices");
                string listKey = element.propertyPath;
                ReorderableList innerReorderableList;
                if (m_innerVertsLists.ContainsKey(listKey))
                {
                    innerReorderableList = m_innerVertsLists[listKey];
                }
                else
                {
                    // create reorderabl list and store it in dict
                    innerReorderableList = new ReorderableList(element.serializedObject, InnerList)
                    {
                        displayAdd = true,
                        displayRemove = true,
                        draggable = true,
                        drawHeaderCallback = innerRect =>
                        {
                            EditorGUI.LabelField(innerRect, "Inner Vertices");
                        },
                        drawElementCallback = (innerRect, innerIndex, innerA, innerH) =>
                        {
                            var innerElement = InnerList.GetArrayElementAtIndex(innerIndex);
                            innerElement.intValue = EditorGUI.IntField(innerRect, innerElement.intValue);
                        }
                    };
                    m_innerVertsLists[listKey] = innerReorderableList;
                }
                position.y += 2;
                var r1 = new Rect(position.x, position.y, position.width, EditorGUIUtility.singleLineHeight);
                var r2 = new Rect(position.x, position.y + EditorGUIUtility.singleLineHeight, position.width, EditorGUIUtility.singleLineHeight);
                var r3 = new Rect(position.x, position.y + 2 * EditorGUIUtility.singleLineHeight, position.width, EditorGUIUtility.singleLineHeight);
                var r4 = new Rect(position.x, position.y + 3 * EditorGUIUtility.singleLineHeight, position.width, EditorGUIUtility.singleLineHeight);
                var height = (InnerList.arraySize + 3) * EditorGUIUtility.singleLineHeight;
                var r5 = new Rect(position.x, position.y + 4 * EditorGUIUtility.singleLineHeight, position.width, height - EditorGUIUtility.singleLineHeight);
                EditorGUI.PropertyField(r1, element.FindPropertyRelative("meshIndex"));
                EditorGUI.PropertyField(r2, element.FindPropertyRelative("mesh.mesh"));
                EditorGUI.PropertyField(r3, element.FindPropertyRelative("colorMode"));
                var category = element.FindPropertyRelative("colorMode");
                if (category.intValue == (int)MeshItem.MeshItemColorMode.SingleColor)
                {
                    EditorGUI.PropertyField(r4, element.FindPropertyRelative("color"));
                }
                else
                {
                    r5 = r4;
                }
                innerReorderableList.DoList(r5);
            };
        }
    }
}
