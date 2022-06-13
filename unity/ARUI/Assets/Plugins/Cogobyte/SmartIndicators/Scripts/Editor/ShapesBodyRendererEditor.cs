using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEditorInternal;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    [CustomEditor(typeof(ShapesBodyRenderer), true)]
    [CanEditMultipleObjects]
    public class ShapesBodyRendererEditor : Editor
    {
        internal ReorderableList m_shapeList;
        internal Dictionary<string, ReorderableList> m_meshLists;
        internal ShapesBodyRenderer m_shapeRenderer;
        public ReorderableList list;

        public void OnEnable()
        {
            m_shapeRenderer = target as ShapesBodyRenderer;
            m_meshLists = new Dictionary<string, ReorderableList>();
            m_shapeList = new ReorderableList(serializedObject, serializedObject.FindProperty("shapes"), true, true, true, true)
            {
                drawHeaderCallback = (rect) =>
                {
                    EditorGUI.LabelField(rect, "Shapes");
                },
                onAddCallback = (ReorderableList l) =>
                {
                    var index = l.serializedProperty.arraySize;
                    l.serializedProperty.arraySize++;
                    l.index = index;
                    var element = l.serializedProperty.GetArrayElementAtIndex(index);
                    element.FindPropertyRelative("spaceBefore").floatValue = 0.5f;
                    element.FindPropertyRelative("spaceAfter").floatValue = 0.5f;
                    element.FindPropertyRelative("scale").vector3Value = Vector3.one;
                },
                elementHeightCallback = index =>
                {
                    var element = m_shapeList.serializedProperty.GetArrayElementAtIndex(index);
                    var innerList = element.FindPropertyRelative("meshes");
                    return (innerList.arraySize + 10) * EditorGUIUtility.singleLineHeight + 60 * innerList.arraySize;
                },
                drawElementCallback = (rect, index, a, h) =>
                {
                    var element = m_shapeList.serializedProperty.GetArrayElementAtIndex(index);
                    var InnerList = element.FindPropertyRelative("meshes");
                    string listKey = element.propertyPath;
                    ReorderableList innerReorderableList;
                    if (m_meshLists.ContainsKey(listKey))
                    {
                        innerReorderableList = m_meshLists[listKey];
                    }
                    else
                    {
                        innerReorderableList = new ReorderableList(element.serializedObject, InnerList);
                        SmartArrowEditor.GenerateMeshList(this, innerReorderableList, m_shapeRenderer.shapes[index].meshes, "Shape " + index + " meshes");
                        m_meshLists[listKey] = innerReorderableList;
                    }
                    // Setup the inner list
                    var height = (InnerList.arraySize + 3) * EditorGUIUtility.singleLineHeight;
                    EditorGUI.LabelField(new Rect(rect.x, rect.y, rect.width, EditorGUIUtility.singleLineHeight), "Shape " + index, EditorStyles.boldLabel);
                    EditorGUI.PropertyField(new Rect(rect.x, rect.y + EditorGUIUtility.singleLineHeight, rect.width, EditorGUIUtility.singleLineHeight), element.FindPropertyRelative("spaceBefore"));
                    EditorGUI.PropertyField(new Rect(rect.x, rect.y + EditorGUIUtility.singleLineHeight * 2, rect.width, EditorGUIUtility.singleLineHeight), element.FindPropertyRelative("spaceAfter"));
                    EditorGUI.PropertyField(new Rect(rect.x, rect.y + EditorGUIUtility.singleLineHeight * 3, rect.width, EditorGUIUtility.singleLineHeight), element.FindPropertyRelative("offset"));
                    EditorGUI.PropertyField(new Rect(rect.x, rect.y + EditorGUIUtility.singleLineHeight * 4, rect.width, EditorGUIUtility.singleLineHeight), element.FindPropertyRelative("rotation"));
                    EditorGUI.PropertyField(new Rect(rect.x, rect.y + EditorGUIUtility.singleLineHeight * 5, rect.width, EditorGUIUtility.singleLineHeight), element.FindPropertyRelative("scale"));
                    Rect r = new Rect(rect.x, rect.y + 6 * EditorGUIUtility.singleLineHeight + 2, rect.width, height - EditorGUIUtility.singleLineHeight);
                    innerReorderableList.DoList(r);
                }
            };
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
        }
        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            EditorGUI.BeginChangeCheck();
            EditorGUILayout.PropertyField(serializedObject.FindProperty("remainingSpaceMode"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("colorMode"));
            if (m_shapeRenderer.colorMode == ShapesBodyRenderer.ShapesBodyRendererColorMode.ColorAlongBody)
            {
                list.DoLayoutList();
            }
            else
            {
                EditorGUILayout.PropertyField(serializedObject.FindProperty("colorsGradient"));
                BodyRenderer.SetColorsToGradient(m_shapeRenderer.colors,m_shapeRenderer.colorsGradient);
            }
            m_shapeList.DoLayoutList();
            EditorGUI.EndChangeCheck();
            serializedObject.ApplyModifiedProperties();
        }
    }
}