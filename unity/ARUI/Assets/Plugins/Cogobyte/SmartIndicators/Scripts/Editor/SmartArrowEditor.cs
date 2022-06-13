using System.Collections.Generic;
using UnityEditor;
using UnityEditorInternal;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    [CustomEditor(typeof(SmartArrow))]
    [CanEditMultipleObjects]
    public class SmartArrowEditor : Editor
    {
        public static bool updateInEdtior = true;
        internal SmartArrow m_smartArrow;
        internal string m_assetName;
        private ReorderableList m_bodyList;
        private ReorderableList m_rendererList;

        [MenuItem("Tools/Cogobyte/SmartIndicators/Generate Outline Arrow")]
        static void GenerateOutlineArrow()
        {
            string assetName = EditorUtility.OpenFilePanel("Load Outline", "", "asset");
            assetName = assetName.Replace(Application.dataPath, "Assets");
            if (assetName.Length != 0)
            {
                OutlineTemplate o = (OutlineTemplate)AssetDatabase.LoadAssetAtPath(assetName, typeof(OutlineTemplate));
                Undo.IncrementCurrentGroup();
                Undo.SetCurrentGroupName("Added Body Renderer");
                var undoGroupIndex = Undo.GetCurrentGroup();
                GameObject smartArrow = new GameObject();
                Undo.RegisterCreatedObjectUndo(smartArrow, "Created GO");
                SmartArrow s = smartArrow.AddComponent<SmartArrow>();

                if (assetName.Length != 0)
                {
                    Material m = (Material)UnityEditor.AssetDatabase.LoadAssetAtPath("Assets/Cogobyte/SmartIndicators/Materials/DefaultIndicatorsMaterial.mat", typeof(Material));
                    if (m != null)
                    {
                        smartArrow.GetComponent<MeshRenderer>().sharedMaterial = m;
                    }
                    Undo.RecordObject(smartArrow, "Simple outline load");
                }
                s.bodyRenderers.Add(new MultiBodyItem() { bodyRenderer = Undo.AddComponent<OutlineBodyRenderer>(smartArrow) });
                OutlineBodyRenderer r = ((OutlineBodyRenderer)s.bodyRenderers[0].bodyRenderer);
                r.outline.CopyOutline(o.outline);
                MeshItem meshItem;
                if (r.outline.backFaceMeshes.Count == 0)
                {
                    meshItem = new MeshItem();
                    Outline.GenerateFaceMesh(meshItem, r.outline, false);
                    r.outline.backFaceMeshes.Add(meshItem);
                }   
                if (r.outline.frontFaceMeshes.Count == 0)
                {
                    meshItem = new MeshItem();
                    Outline.GenerateFaceMesh(meshItem, r.outline, true);
                    r.outline.frontFaceMeshes.Add(meshItem);
                }
                s.arrowHead = Undo.AddComponent<OutlineTip>(smartArrow);
                BodyTipMeshItem bodyTipMeshItem = new BodyTipMeshItem();
                Outline.GenerateOutlineToOutlineFace(bodyTipMeshItem,o.outline, r.outline, true);
                r.tailMeshConnector.Add(bodyTipMeshItem);
                bodyTipMeshItem = new BodyTipMeshItem();
                Outline.GenerateOutlineToOutlineFace(bodyTipMeshItem,o.outline, r.outline, false);
                r.headMeshConnector.Add(bodyTipMeshItem);


                meshItem = new MeshItem();
                OutlineTip t = (OutlineTip)s.arrowHead;
                t.outline = new Outline(r.outline);
                Outline.GenerateFaceMesh(meshItem,t.outline, false);
                t.outline.backFaceMeshes.Add(meshItem);
                meshItem = new MeshItem();
                Outline.GenerateFaceMesh(meshItem,t.outline, true);
                t.outline.frontFaceMeshes.Add(meshItem);
                t.size = new Vector3(2, 2, 2);
                s.arrowPath = smartArrow.AddComponent<PointToPointArrowPath>();
                ((PointToPointArrowPath)s.arrowPath).pointB = new Vector3(0, 0, 10);
                Selection.activeGameObject = smartArrow;
                smartArrow.name = "ArrowObject";
                Undo.CollapseUndoOperations(undoGroupIndex);
            }
            else
            {
                EditorUtility.DisplayDialog("Could not load outline", "Wrong scriptable object type", "Ok");
            }

        }
        public override bool RequiresConstantRepaint()
        {
            return true;
        }
        void OnEnable()
        {
            m_smartArrow = target as SmartArrow;
            m_bodyList = new ReorderableList(serializedObject, serializedObject.FindProperty("bodyRenderers"), true, true, true, true)
            {
                drawHeaderCallback = (Rect position) =>
                {
                    EditorGUI.LabelField(position, "Body Renderers");
                },
                onAddCallback = (ReorderableList list) =>
                {
                    var index = list.serializedProperty.arraySize;
                    list.serializedProperty.arraySize++;
                    list.index = index;
                    var element = list.serializedProperty.GetArrayElementAtIndex(index);
                    element.FindPropertyRelative("lengthMode").enumValueIndex = (int)MultiBodyItem.LengthMode.Fill;
                    element.FindPropertyRelative("length").floatValue = 1;
                    element.FindPropertyRelative("percentage").floatValue = 1;
                    element.FindPropertyRelative("displacement").floatValue = 0;
                    element.FindPropertyRelative("colorDisplacement").floatValue = 0;
                },
                elementHeightCallback = (index) =>
                {
                    Repaint();
                    return EditorGUIUtility.singleLineHeight * 4f + 4;
                },
                drawElementCallback = (Rect position, int index, bool isActive, bool isFocused) =>
                {
                    var element = m_bodyList.serializedProperty.GetArrayElementAtIndex(index);
                    position.y += 2;
                    EditorGUI.indentLevel = 0;
                    var r1 = new Rect(position.x, position.y, (position.x + EditorGUIUtility.labelWidth) / 2, EditorGUIUtility.singleLineHeight);
                    var r2 = new Rect(position.x + EditorGUIUtility.labelWidth, position.y, EditorGUIUtility.labelWidth / 2, EditorGUIUtility.singleLineHeight);
                    var r3 = new Rect(5 + position.x + (EditorGUIUtility.labelWidth + EditorGUIUtility.labelWidth / 2), position.y, position.width - (5 + (EditorGUIUtility.labelWidth + EditorGUIUtility.labelWidth / 2)), EditorGUIUtility.singleLineHeight);
                    var r4 = new Rect(position.x, position.y + EditorGUIUtility.singleLineHeight, position.width, EditorGUIUtility.singleLineHeight);
                    var r5 = new Rect(position.x, position.y + 2 * EditorGUIUtility.singleLineHeight, position.width, EditorGUIUtility.singleLineHeight);
                    var r6 = new Rect(position.x, position.y + 3 * EditorGUIUtility.singleLineHeight, position.width, EditorGUIUtility.singleLineHeight);
                    EditorGUI.LabelField(r1, "Length ");
                    EditorGUI.PropertyField(r2, element.FindPropertyRelative("lengthMode"), GUIContent.none);
                    var category = element.FindPropertyRelative("lengthMode");
                    if (category.intValue == (int)MultiBodyItem.LengthMode.Fixed)
                    {
                        float tempLW = EditorGUIUtility.labelWidth;
                        EditorGUIUtility.labelWidth = 15;
                        EditorGUI.PropertyField(r3, element.FindPropertyRelative("length"), new GUIContent("L:"));
                        EditorGUIUtility.labelWidth = tempLW;
                    }
                    else if (category.intValue == (int)MultiBodyItem.LengthMode.Percentage)
                    {
                        float tempLW = EditorGUIUtility.labelWidth;
                        EditorGUIUtility.labelWidth = 15;
                        EditorGUI.PropertyField(r3, element.FindPropertyRelative("percentage"), new GUIContent("%:"));
                        EditorGUIUtility.labelWidth = tempLW;
                    }
                    EditorGUI.PropertyField(r4, element.FindPropertyRelative("displacement"), new GUIContent("Renderer displacement"));
                    EditorGUI.PropertyField(r5, element.FindPropertyRelative("colorDisplacement"), new GUIContent("Color displacement"));
                    EditorGUI.PropertyField(r6, element.FindPropertyRelative("bodyRenderer"), new GUIContent("Body Renderer"));
                }
            };
            m_rendererList = new ReorderableList(serializedObject, serializedObject.FindProperty("arrowRenderers"), true, true, true, true)
            {
                drawHeaderCallback = (Rect position) =>
                {
                    EditorGUI.LabelField(position, "Arrow Renderers");
                },
                elementHeightCallback = (index) =>
                {
                    Repaint();
                    return EditorGUIUtility.singleLineHeight;
                },
                drawElementCallback = (Rect position, int index, bool isActive, bool isFocused) =>
                {
                    var element = m_rendererList.serializedProperty.GetArrayElementAtIndex(index);
                    EditorGUI.indentLevel = 0;
                    var r1 = new Rect(position.x, position.y, position.width, EditorGUIUtility.singleLineHeight);
                    EditorGUI.PropertyField(r1, element.FindPropertyRelative("meshFilter"));
                }
            };
        }
        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            EditorGUI.BeginChangeCheck();
            GUILayout.BeginVertical(EditorStyles.helpBox);
            {
                GUILayout.BeginHorizontal(EditorStyles.toolbar);
                EditorGUILayout.LabelField("Smart Arrow Settings", EditorStyles.boldLabel);
                if (GUILayout.Button("Actions", EditorStyles.toolbarDropDown))
                {
                    GenericMenu toolsMenu = new GenericMenu();
                    toolsMenu.AddItem(new GUIContent("Add Mesh Filter"), false, ()=> {
                        Undo.IncrementCurrentGroup();
                        Undo.SetCurrentGroupName("Added Mesh Filters");
                        var undoGroupIndex = Undo.GetCurrentGroup();
                        Undo.RecordObject(m_smartArrow, "Smar Arrow Edit");
                        GameObject g = m_smartArrow.AddMeshFilter(m_smartArrow.GetComponent<MeshRenderer>().sharedMaterial);
                        Undo.RegisterCreatedObjectUndo(g, "Created GO");
                        Undo.CollapseUndoOperations(undoGroupIndex);
                    });
                    toolsMenu.AddItem(new GUIContent("Add Path/Point To Point"), false, () => {
                        Undo.IncrementCurrentGroup();
                        Undo.SetCurrentGroupName("Added Paths");
                        var undoGroupIndex = Undo.GetCurrentGroup();
                        Undo.RecordObject(m_smartArrow, "Add Path");
                        ArrowPath p = Undo.AddComponent<PointToPointArrowPath>(m_smartArrow.gameObject);
                        m_smartArrow.arrowPath = p;
                        Undo.CollapseUndoOperations(undoGroupIndex);
                    });
                    toolsMenu.AddItem(new GUIContent("Add Path/Point List"), false, () => {
                        Undo.IncrementCurrentGroup();
                        Undo.SetCurrentGroupName("Added Paths");
                        var undoGroupIndex = Undo.GetCurrentGroup();
                        Undo.RecordObject(m_smartArrow, "Add Path");
                        ArrowPath p = Undo.AddComponent<PointListArrowPath>(m_smartArrow.gameObject);
                        m_smartArrow.arrowPath = p;
                        Undo.CollapseUndoOperations(undoGroupIndex);
                    });
                    toolsMenu.AddItem(new GUIContent("Add Path/Circle Path "), false, () => {
                        Undo.IncrementCurrentGroup();
                        Undo.SetCurrentGroupName("Added Paths");
                        var undoGroupIndex = Undo.GetCurrentGroup();
                        Undo.RecordObject(m_smartArrow, "Add Path");
                        ArrowPath p = Undo.AddComponent<CircleArrowPath>(m_smartArrow.gameObject);
                        m_smartArrow.arrowPath = p;
                        Undo.CollapseUndoOperations(undoGroupIndex);
                    });
                    toolsMenu.AddItem(new GUIContent("Add Path/Bezier Path"), false, () => {
                        Undo.IncrementCurrentGroup();
                        Undo.SetCurrentGroupName("Added Paths");
                        var undoGroupIndex = Undo.GetCurrentGroup();
                        Undo.RecordObject(m_smartArrow, "Add Path");
                        ArrowPath p = Undo.AddComponent<BezierArrowPath>(m_smartArrow.gameObject);
                        m_smartArrow.arrowPath = p;
                        Undo.CollapseUndoOperations(undoGroupIndex);
                    });
                    toolsMenu.AddItem(new GUIContent("Add Tail/Outline Tip"), false, () => {
                        Undo.IncrementCurrentGroup();
                        Undo.SetCurrentGroupName("Added Tip");
                        var undoGroupIndex = Undo.GetCurrentGroup();
                        Undo.RecordObject(m_smartArrow, "Add Tip");
                        ArrowTip t = Undo.AddComponent<OutlineTip>(m_smartArrow.gameObject);
                        m_smartArrow.arrowTail = t;
                        Undo.CollapseUndoOperations(undoGroupIndex);
                    });
                    toolsMenu.AddItem(new GUIContent("Add Tail/Vertical Outline Tip"), false, () => {
                        Undo.IncrementCurrentGroup();
                        Undo.SetCurrentGroupName("Added Tip");
                        var undoGroupIndex = Undo.GetCurrentGroup();
                        Undo.RecordObject(m_smartArrow, "Add Tip");
                        ArrowTip t = Undo.AddComponent<VerticalOutlineTip>(m_smartArrow.gameObject);
                        m_smartArrow.arrowTail = t;
                        Undo.CollapseUndoOperations(undoGroupIndex);
                    });
                    toolsMenu.AddItem(new GUIContent("Add Tail/Mesh Tip"), false, () => {
                        Undo.IncrementCurrentGroup();
                        Undo.SetCurrentGroupName("Added Tip");
                        var undoGroupIndex = Undo.GetCurrentGroup();
                        Undo.RecordObject(m_smartArrow, "Add Body");
                        ArrowTip t = Undo.AddComponent<MeshTip>(m_smartArrow.gameObject);
                        m_smartArrow.arrowTail = t;
                        Undo.CollapseUndoOperations(undoGroupIndex);
                    });
                    toolsMenu.AddItem(new GUIContent("Add Head/Outline Tip"), false, () => {
                        Undo.IncrementCurrentGroup();
                        Undo.SetCurrentGroupName("Added Tip");
                        var undoGroupIndex = Undo.GetCurrentGroup();
                        Undo.RecordObject(m_smartArrow, "Add Tip");
                        ArrowTip t = Undo.AddComponent<OutlineTip>(m_smartArrow.gameObject);
                        m_smartArrow.arrowHead = t;
                        Undo.CollapseUndoOperations(undoGroupIndex);
                    });
                    toolsMenu.AddItem(new GUIContent("Add Head/Vertical Outline Tip"), false, () => {
                        Undo.IncrementCurrentGroup();
                        Undo.SetCurrentGroupName("Added Tip");
                        var undoGroupIndex = Undo.GetCurrentGroup();
                        Undo.RecordObject(m_smartArrow, "Add Tip");
                        ArrowTip t = Undo.AddComponent<VerticalOutlineTip>(m_smartArrow.gameObject);
                        m_smartArrow.arrowHead = t;
                        Undo.CollapseUndoOperations(undoGroupIndex);
                    });
                    toolsMenu.AddItem(new GUIContent("Add Head/Mesh Tip"), false, () => {
                        Undo.IncrementCurrentGroup();
                        Undo.SetCurrentGroupName("Added Tip");
                        var undoGroupIndex = Undo.GetCurrentGroup();
                        Undo.RecordObject(m_smartArrow, "Add Body");
                        ArrowTip t = Undo.AddComponent<MeshTip>(m_smartArrow.gameObject);
                        m_smartArrow.arrowHead = t;
                        Undo.CollapseUndoOperations(undoGroupIndex);
                    });
                    toolsMenu.AddItem(new GUIContent("Add Body Renderer/Outline Body Renderer"), false, () => {
                        Undo.IncrementCurrentGroup();
                        Undo.SetCurrentGroupName("Added Body Renderer");
                        var undoGroupIndex = Undo.GetCurrentGroup();
                        Undo.RecordObject(m_smartArrow, "Add Outline Body");
                        m_smartArrow.bodyRenderers.Add(new MultiBodyItem() { bodyRenderer = Undo.AddComponent<OutlineBodyRenderer>(m_smartArrow.gameObject) });
                        Undo.CollapseUndoOperations(undoGroupIndex);
                    });
                    toolsMenu.AddItem(new GUIContent("Add Body Renderer/Shapes Body Renderer"), false, () => {
                        Undo.IncrementCurrentGroup();
                        Undo.SetCurrentGroupName("Added Body Renderer");
                        var undoGroupIndex = Undo.GetCurrentGroup();
                        Undo.RecordObject(m_smartArrow, "Add Body");
                        m_smartArrow.bodyRenderers.Add(new MultiBodyItem() { bodyRenderer = Undo.AddComponent<ShapesBodyRenderer>(m_smartArrow.gameObject) });
                        Undo.CollapseUndoOperations(undoGroupIndex);
                    });
                    toolsMenu.AddItem(new GUIContent("Save Meshes"), false, ()=> {
                        m_assetName = EditorUtility.SaveFilePanelInProject("Save Mesh", m_assetName, "asset", "Please enter a file name to save the mesh to");
                        for (int i = 0; i < m_smartArrow.arrowRenderers.Count; i++)
                        {
                            if (i != 0)
                                AssetDatabase.CreateAsset(m_smartArrow.arrowRenderers[i].meshFilter.sharedMesh, m_assetName + i);
                            else
                                AssetDatabase.CreateAsset(m_smartArrow.arrowRenderers[i].meshFilter.sharedMesh, m_assetName);
                        }
                    });
                    toolsMenu.AddItem(new GUIContent("Reset Meshes"), false, () => {
                        for (int i = 0; i < m_smartArrow.arrowRenderers.Count; i++)
                        {
                            m_smartArrow.arrowRenderers[i].arrowMesh.mesh = new Mesh();
                        }
                    });
                    toolsMenu.DropDown(new Rect(Screen.width - 216 - 40, 0, 0, 16));
                    GUIUtility.ExitGUI();
                }
                GUILayout.EndHorizontal();
                EditorGUI.indentLevel++;
                m_rendererList.DoLayoutList();
                updateInEdtior = EditorGUILayout.Toggle("Update in Editor", updateInEdtior);
                EditorGUILayout.PropertyField(serializedObject.FindProperty("arrowPath"));
                EditorGUILayout.MinMaxSlider("Path Percentage", ref m_smartArrow.StartPercentage, ref m_smartArrow.EndPercentage, 0, 1);
                EditorGUILayout.PropertyField(serializedObject.FindProperty("arrowTail"));
                EditorGUILayout.PropertyField(serializedObject.FindProperty("arrowHead"));
                EditorGUILayout.PropertyField(serializedObject.FindProperty("displacement"));
                EditorGUILayout.Space();
                EditorGUILayout.Space();
                m_bodyList.DoLayoutList();
                EditorGUI.indentLevel--;
            }
            GUILayout.EndVertical();
            
            if (updateInEdtior && m_smartArrow.gameObject.scene.name != null)
            {
                try
                {
                    m_smartArrow.UpdateArrow();
                }
                catch (System.Exception e)
                {
                    Debug.LogException(e, this);
                }
            }
            EditorGUI.EndChangeCheck();
            serializedObject.ApplyModifiedProperties();
        }
        public static void GenerateMeshList(Editor e,ReorderableList list,List<MeshItem> meshItems, string title)
        {
            list.onAddCallback = (ReorderableList l) =>
            {
                var index = list.serializedProperty.arraySize;
                list.serializedProperty.arraySize++;
                list.index = index;
                var element = list.serializedProperty.GetArrayElementAtIndex(index);
                element.FindPropertyRelative("color").colorValue = Color.white;
            };
            list.elementHeightCallback = (index) =>
            {
                e.Repaint();
                return EditorGUIUtility.singleLineHeight * 4f + 5;
            };
            list.drawHeaderCallback = (Rect position) =>
            {
                EditorGUI.LabelField(position, title);
            };
            list.drawElementCallback = (Rect position, int index, bool isActive, bool isFocused) => {
                var element = list.serializedProperty.GetArrayElementAtIndex(index);
                position.y += 2;
                var r1 = new Rect(position.x, position.y, position.width - 95, EditorGUIUtility.singleLineHeight);
                var r2 = new Rect(position.x, position.y + EditorGUIUtility.singleLineHeight, position.width , EditorGUIUtility.singleLineHeight);
                var r3 = new Rect(position.x, position.y + 2 * EditorGUIUtility.singleLineHeight, position.width, EditorGUIUtility.singleLineHeight);
                var r4 = new Rect(position.x, position.y + 3 * EditorGUIUtility.singleLineHeight, position.width, EditorGUIUtility.singleLineHeight);
                EditorGUI.PropertyField(r4, element.FindPropertyRelative("mesh.mesh"));
                if(meshItems[index].mesh.mesh!=null)
                meshItems[index].LoadMesh(meshItems[index].mesh.mesh);
                EditorGUI.PropertyField(r1, element.FindPropertyRelative("meshIndex"));
                EditorGUI.PropertyField(r2, element.FindPropertyRelative("colorMode"));
                var category = element.FindPropertyRelative("colorMode");
                if (category.intValue == (int) MeshItem.MeshItemColorMode.SingleColor)
                {
                    EditorGUI.PropertyField(r3, element.FindPropertyRelative("color"));
                }

            };
        }
    }
}
