using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

namespace Cogobyte.SmartProceduralIndicators {
    public class OutlineEditorWindow : EditorWindow
    {
        public Object undoObject;

        Outline m_outline;

        bool windowOptions = false;
        Color32 gridColor = new Color32(210, 210, 210, 255);
        Color32 secondaryGridColor = new Color32(100, 100, 100, 255);
        Color32 editorBackgroundColor = new Color32(170, 170, 170, 255);
        Color32 menuBackgroundColor = new Color32(190, 190, 190, 255);
        Color32 gizmoColor = Color.blue;
        Color32 firstPointColor = Color.red;
        float menuWidth = 400;
        public float zoom = 2.5f;
        float grid = 0.1f;
        int gridSilder = 10;
        bool verticalOutline =false;

        Rect editorRect;
        Rect menuRect;

        const float pixelLengthRatio = 100;

        int m_edgeMode = 0;
        string[] m_edgeModes = new string[]
        {
            "Edge Edit","Line", "Curve", "Circle"
        };

        public static void LoadWindow(Outline outline,Object uObject)
        {
            OutlineEditorWindow w = ((OutlineEditorWindow)GetWindow(typeof(OutlineEditorWindow), false, "Outline Visual Editor"));
            w.m_outline = outline;
            w.undoObject = uObject;
        }

        




        List<OutlineEdge> m_selectedEdges = new List<OutlineEdge>();
        OutlineEdgePoint m_draggedPoint = null;
        Vector2 edgeDrag = Vector2.zero;
        Vector2 scrollPosition = Vector2.zero;
        Vector2[] bezier = new Vector2[] { new Vector2(-0.5f, 0), new Vector2(0.5f, 0), new Vector2(-0.5f, 0.5f), new Vector2(0.5f, 0.5f) };
        bool dragging = false;
        bool selecting = false;
        int dragBez = 0;
        int lineLod = 1;
        int curveLod = 10;
        float radius = 0.5f;
        float circleStart = 0;
        float circleLength = 360;
        int circleLod = 8;
        Color32 startColor = Color.white;
        Color32 endColor = Color.white;
        Gradient m_colorsGradient = new Gradient()
        {
            colorKeys = new GradientColorKey[2] {
            new GradientColorKey(Color.white, 0),
            new GradientColorKey(Color.white, 1)
            },
            alphaKeys = new GradientAlphaKey[2] {
            new GradientAlphaKey(1, 0),
            new GradientAlphaKey(1, 1)
            }
        };

        public void Update()
        {
            Repaint();
        }

        public void RenderWindowMenu()
        {
            Undo.RecordObject(undoObject, "Changed Area Of Effect");
            menuRect = new Rect(position.width - menuWidth + 10, 5, menuWidth - 20, position.height - 10);
            EditorGUI.DrawRect(new Rect(position.width - menuWidth, 0, menuWidth, position.height), menuBackgroundColor);
            Rect menuItemR = new Rect(menuRect.x, 5 + menuRect.y, menuRect.width, EditorGUIUtility.singleLineHeight + 5);
            #region WindowsSettings
            windowOptions = EditorGUI.Foldout(menuItemR, windowOptions, "Editor Options");
            menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
            if (windowOptions)
            {
                editorBackgroundColor = EditorGUI.ColorField(menuItemR, "Background Color", editorBackgroundColor);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                gridColor = EditorGUI.ColorField(menuItemR, "Grid Color", gridColor);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                secondaryGridColor = EditorGUI.ColorField(menuItemR, "Secondary Grid Color", secondaryGridColor);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                gizmoColor = EditorGUI.ColorField(menuItemR, "Gizmo color", gizmoColor);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                firstPointColor = EditorGUI.ColorField(menuItemR, "First point color", firstPointColor);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                zoom = EditorGUI.Slider(menuItemR, "Zoom", zoom, 0.1f, 4);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                gridSilder = EditorGUI.IntSlider(menuItemR, "Grid", gridSilder, 1, 100);
                grid = gridSilder * 0.01f;
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                verticalOutline = EditorGUI.Toggle(menuItemR,"Vertical Outline",verticalOutline);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
            }
            #endregion
            m_edgeMode = EditorGUI.Popup(menuItemR, "Mode", m_edgeMode, m_edgeModes);
            menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
            if (m_edgeMode == 0)
            {
                if (m_selectedEdges.Count != 0)
                {
                    if (GUI.Button(new Rect(menuItemR.x, menuItemR.y, menuItemR.width / 3 - 5, menuItemR.height), "Auto Normals"))
                    {
                        foreach (OutlineEdge e in m_selectedEdges)
                        {
                            e.CaluclateNormals();
                        }
                    }
                    if (GUI.Button(new Rect(menuItemR.x + menuItemR.width / 3, menuItemR.y, menuItemR.width / 3 - 5, menuItemR.height), "Smooth Normals"))
                    {
                        foreach (OutlineEdge e in m_selectedEdges)
                        {
                            m_outline.CalculateSmoothNormals(e);
                        }
                    }
                    if (GUI.Button(new Rect(menuItemR.x + 5 + 2 * menuItemR.width / 3, menuItemR.y, menuItemR.width / 3 - 5, menuItemR.height), "Clone"))
                    {
                        List<OutlineEdge> tempEdges = new List<OutlineEdge>();
                        foreach (OutlineEdge e in m_selectedEdges)
                        {
                            OutlineEdge tempEdge = new OutlineEdge(e);
                            m_outline.edges.Add(tempEdge);
                            tempEdges.Add(tempEdge);
                        }
                        m_selectedEdges = tempEdges;
                    }
                    menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                    if (GUI.Button(new Rect(menuItemR.x, menuItemR.y, menuItemR.width / 3 - 5, menuItemR.height), "Flip vertical"))
                    {
                        foreach (OutlineEdge e in m_selectedEdges)
                        {
                            e.FlipVertical();
                        }
                    }
                    if (GUI.Button(new Rect(menuItemR.x + menuItemR.width / 3, menuItemR.y, menuItemR.width / 3 - 5, menuItemR.height), "Flip horizontal"))
                    {
                        foreach (OutlineEdge e in m_selectedEdges)
                        {
                            e.FlipHorizontal();
                        }
                    }
                    if (GUI.Button(new Rect(menuItemR.x + 5 + 2 * menuItemR.width / 3, menuItemR.y, menuItemR.width / 3 - 5, menuItemR.height), "Roll"))
                    {
                        foreach (OutlineEdge e in m_selectedEdges)
                        {
                            e.FlipAxes();
                        }
                    }
                    menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                    if ((Event.current.keyCode == KeyCode.Delete && Event.current.type == EventType.KeyUp) || GUI.Button(menuItemR, "Delete Edge"+((m_selectedEdges.Count>1) ?"s":"")))
                    {
                        foreach (OutlineEdge e in m_selectedEdges)
                        {
                            m_outline.edges.Remove(e);
                        }
                        m_selectedEdges.Clear();
                    }
                    menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                }
                if (m_selectedEdges.Count == 1)
                {
                    EditorGUI.LabelField(menuItemR, "Selected Edge");
                    menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                    OutlineEdge e = m_selectedEdges[0];
                    e.meshIndex = EditorGUI.IntField(menuItemR, "Arrow Mesh Index", e.meshIndex);
                    menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                    m_colorsGradient = EditorGUI.GradientField(new Rect(menuItemR.x, menuItemR.y, menuItemR.width / 2 - 5, menuItemR.height), m_colorsGradient);
                    if (GUI.Button(new Rect(menuItemR.x + menuItemR.width / 2 + 5, menuItemR.y, menuItemR.width / 2 - 5, menuItemR.height), "Set Colors"))
                    {
                        e.SetPointsColors(m_colorsGradient);
                    }
                    menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                    if (GUI.Button(menuItemR, "Add Point"))
                    {
                        e.points.Add(new OutlineEdgePoint()
                        {
                            position = e.points[e.points.Count - 1].position + (e.points[e.points.Count - 1].position - e.points[e.points.Count - 2].position).normalized * 0.1f,
                            normal = e.points[e.points.Count - 1].normal,
                            color = e.points[e.points.Count - 1].color
                        });
                    }
                    menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                    scrollPosition = GUI.BeginScrollView(new Rect(menuItemR.x, menuItemR.y, menuWidth - 20, editorRect.height - menuItemR.y), scrollPosition, new Rect(0, 0, menuWidth - 40, e.points.Count * (5 * EditorGUIUtility.singleLineHeight + 15)));
                    Rect pointsRect = new Rect(0, 0, menuItemR.width - 30, menuItemR.height);
                    for (int i = 0; i < e.points.Count; i++)
                    {
                        e.points[i].position = EditorGUI.Vector2Field(pointsRect, "Point " + i, e.points[i].position);
                        pointsRect.y += 5 + EditorGUIUtility.singleLineHeight * 2;
                        e.points[i].normal = EditorGUI.Vector2Field(pointsRect, "Normal ", e.points[i].normal);
                        pointsRect.y += 5 + EditorGUIUtility.singleLineHeight * 2;
                        EditorGUI.LabelField(new Rect(pointsRect.x, pointsRect.y, pointsRect.width / 4 - 5, pointsRect.height), "Color");
                        e.points[i].color = EditorGUI.ColorField(new Rect(pointsRect.x + pointsRect.width / 4, pointsRect.y, pointsRect.width / 4 - 5, pointsRect.height), e.points[i].color);
                        if (e.points.Count > 2)
                        {
                            if (i != 0 && i != e.points.Count - 1)
                            {
                                if (GUI.Button(new Rect(pointsRect.x + pointsRect.width / 2, pointsRect.y, pointsRect.width / 4 - 5, pointsRect.height), "Split"))
                                {
                                    OutlineEdge splitEdge = new OutlineEdge() { meshIndex = e.meshIndex, color = e.color };
                                    splitEdge.points.Clear();
                                    splitEdge.points.Add(new OutlineEdgePoint(e.points[i]));
                                    for (int j = i + 1; j < e.points.Count; j++)
                                    {
                                        splitEdge.points.Add(new OutlineEdgePoint(e.points[j]));
                                    }
                                    e.points.RemoveRange(i + 1, e.points.Count - i - 1);
                                    m_outline.edges.Add(splitEdge);
                                }
                            }
                            if (GUI.Button(new Rect(pointsRect.x + (pointsRect.width * 3) / 4, pointsRect.y, pointsRect.width / 4, pointsRect.height), "Delete"))
                            {
                                e.points.RemoveAt(i);
                            }
                        }
                        pointsRect.y += 5 + EditorGUIUtility.singleLineHeight;
                    }
                    GUI.EndScrollView();

                }
                if (m_selectedEdges.Count > 1)
                {
                    EditorGUI.LabelField(menuItemR, m_selectedEdges.Count + " Edges Selected");
                    menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                    if (GUI.Button(menuItemR, "Join"))
                    {
                        OutlineEdge mainEdge = m_selectedEdges[0];
                        for (int i = 0; i < m_selectedEdges.Count; i++)
                        {
                            for (int j = 0; j < m_selectedEdges.Count; j++)
                            {
                                if (m_selectedEdges[i] != m_selectedEdges[j])
                                {
                                    if (Vector2.Distance(m_selectedEdges[j].points[0].position, m_selectedEdges[i].points[m_selectedEdges[i].points.Count - 1].position) < SmartArrowUtilities.Utilities.errorRate)
                                    {
                                        for (int k = 0; k < m_selectedEdges[j].points.Count; k++)
                                        {
                                            m_selectedEdges[i].points.Add(new OutlineEdgePoint(m_selectedEdges[j].points[k]));
                                        }
                                        m_outline.edges.Remove(m_selectedEdges[j]);
                                        m_selectedEdges.RemoveAt(j);
                                        if (j > i)
                                        {
                                            i--;
                                            break;
                                        }
                                        else
                                        {
                                            i -= 2;
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                    }
                    menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                }
            }
            if (m_edgeMode == 1)
            {
                if (Event.current.keyCode == KeyCode.P)
                {
                    bezier[1] = SnapMousePointToGrid(Event.current.mousePosition);
                }
                lineLod = EditorGUI.IntSlider(menuItemR, "Sections", lineLod, 1, 10);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                bezier[0] = EditorGUI.Vector2Field(menuItemR, new GUIContent("Point A"), bezier[0]);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight * 2;
                bezier[1] = EditorGUI.Vector2Field(menuItemR, "Point B", bezier[1]);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight * 2;
                startColor = EditorGUI.ColorField(menuItemR, "Start Color", startColor);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                endColor = EditorGUI.ColorField(menuItemR, "End Color", endColor);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                if ((Event.current.keyCode == KeyCode.P && Event.current.type == EventType.KeyUp) || GUI.Button(menuItemR, "Add Line"))
                {
                    Vector2 rPos = bezier[0];
                    Vector2 rPos2 = bezier[1];
                    Vector2 dir = rPos2 - rPos;
                    OutlineEdge e = new OutlineEdge();
                    e.points = new List<OutlineEdgePoint>();
                    for (int i = 0; i <= lineLod; i++)
                    {
                        rPos2 = rPos + (dir * i) / lineLod;
                        e.points.Add(new OutlineEdgePoint() { position = rPos2, normal = Vector2.up, color = Color.Lerp(startColor, endColor, i * 1f / lineLod) });
                    }
                    e.CaluclateNormals();
                    m_outline.edges.Add(e);
                    Vector2 temp = bezier[1];
                    bezier[1] += (bezier[1] - bezier[0]);
                    bezier[0] = temp;
                }
            }
            if (m_edgeMode == 2)
            {
                curveLod = EditorGUI.IntSlider(menuItemR, "Sections", curveLod, 1, 10);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                bezier[0] = EditorGUI.Vector2Field(menuItemR, "Point A", bezier[0]);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight * 2;
                bezier[1] = EditorGUI.Vector2Field(menuItemR, "Point B", bezier[1]);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight * 2;
                bezier[2] = EditorGUI.Vector2Field(menuItemR, "Tangent A", bezier[2]);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight * 2;
                bezier[3] = EditorGUI.Vector2Field(menuItemR, "Tangent B", bezier[3]);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight * 2;
                startColor = EditorGUI.ColorField(menuItemR, "Start Color", startColor);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                endColor = EditorGUI.ColorField(menuItemR, "End Color", endColor);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                if (GUI.Button(menuItemR, "Add Curve"))
                {
                    OutlineEdge e = new OutlineEdge();
                    e.points = new List<OutlineEdgePoint>();
                    for (int i = 0; i <= curveLod; i++)
                    {
                        Vector3 a = SmartArrowUtilities.Bezier.GetPoint(
                            bezier[0],
                            bezier[2],
                            bezier[3],
                            bezier[1], ((float)i) / curveLod);
                        e.points.Add(new OutlineEdgePoint() { position = a, color = Color32.Lerp(startColor,endColor,((float)i)/curveLod) });
                    }
                    e.CaluclateNormals();
                    m_outline.edges.Add(e);
                    Vector2 temp = bezier[1];
                    MoveCurvePoint(1, 2 * bezier[1] - bezier[0]);
                    MoveCurvePoint(0, temp);
                }
            }
            if (m_edgeMode == 3)
            {
                bezier[0] = EditorGUI.Vector2Field(menuItemR, "Center", bezier[0]);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight * 2;
                radius = EditorGUI.FloatField(menuItemR, "Radius", radius);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                circleStart = EditorGUI.Slider(menuItemR, "Circle Start", circleStart, 0, 360);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                circleLength = EditorGUI.Slider(menuItemR, "Circle Length", circleLength, 0, 360);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                circleLod = EditorGUI.IntSlider(menuItemR, "Segments", circleLod, 3, 16);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                startColor = EditorGUI.ColorField(menuItemR, "Start Color", startColor);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                endColor = EditorGUI.ColorField(menuItemR, "End Color", endColor);
                menuItemR.y += 5 + EditorGUIUtility.singleLineHeight;
                if (GUI.Button(menuItemR, "Add Circle"))
                {
                    OutlineEdge e = new OutlineEdge();
                    e.points = new List<OutlineEdgePoint>();
                    for (int i = 0; i <= circleLod; i++)
                    {
                        Vector2 rPos = bezier[0] + radius * new Vector2(Mathf.Cos(i * 1f / circleLod * circleLength * Mathf.Deg2Rad + circleStart * Mathf.Deg2Rad), -Mathf.Sin(i * 1f / circleLod * circleLength * Mathf.Deg2Rad + circleStart * Mathf.Deg2Rad));
                        e.points.Add(new OutlineEdgePoint() { position = rPos, normal = new Vector2((rPos - bezier[0]).x, -(rPos - bezier[0]).y), color = Color.Lerp(startColor, endColor, i * 1f / circleLod) });
                    }
                    m_outline.edges.Add(e);
                    e.CaluclateNormals();
                }
            }
        }

        public void RenderWindowGizmos()
        {
            #region drawGrid
            int gridlines = (int)(((editorRect.height) / 2) / (grid * 100 * zoom));
            Handles.color = gridColor;
            for (int i = -gridlines; i <= gridlines; i++)
            {
                Handles.DrawLine(new Vector2(editorRect.center.x + i * grid * 100 * zoom, editorRect.y), new Vector2(editorRect.center.x + i * grid * 100 * zoom, editorRect.y + editorRect.height));
            }
            for (int i = -gridlines; i <= gridlines; i++)
            {
                Handles.DrawLine(new Vector2(editorRect.x, editorRect.center.y + i * grid * 100 * zoom), new Vector2(editorRect.x + editorRect.width, editorRect.center.y + i * grid * 100 * zoom));
            }
            float vOffset = 0;
            if (verticalOutline) vOffset = -50 * zoom;
            Handles.color = secondaryGridColor;
            Handles.DrawLine(new Vector2(editorRect.center.x + 50 * zoom, editorRect.y), new Vector2(editorRect.center.x + 50 * zoom, editorRect.y + editorRect.height));
            Handles.DrawLine(new Vector2(editorRect.center.x - 50 * zoom, editorRect.y), new Vector2(editorRect.center.x - 50 * zoom, editorRect.y + editorRect.height));
            Handles.DrawLine(new Vector2(editorRect.center.x, editorRect.y), new Vector2(editorRect.center.x, editorRect.y + editorRect.height));
            Handles.DrawLine(new Vector2(editorRect.x, editorRect.center.y + 50 * zoom + vOffset), new Vector2(editorRect.x + editorRect.width, editorRect.center.y + 50 * zoom + vOffset));
            Handles.DrawLine(new Vector2(editorRect.x, editorRect.center.y - 50 * zoom + vOffset), new Vector2(editorRect.x + editorRect.width, editorRect.center.y - 50 * zoom + vOffset));
            Handles.DrawLine(new Vector2(editorRect.x, editorRect.center.y + vOffset), new Vector2(editorRect.x + editorRect.width, editorRect.center.y + vOffset));
            #endregion

            Vector2 rPos;
            Vector2 rPos2 = Vector2.zero;
            Vector2 norm = Vector2.up;
            for (int i = 0; i < m_outline.edges.Count; i++)
            {
                for (int j = 0; j < m_outline.edges[i].points.Count; j++)
                {
                    rPos = GetGuiPosFromOutlinPos(m_outline.edges[i].points[j].position); ;
                    DrawNormal(rPos, new Vector2(m_outline.edges[i].points[j].normal.x, -m_outline.edges[i].points[j].normal.y));
                    EditorGUI.DrawRect(new Rect(rPos.x - 2, rPos.y - 2, 5, 5), gizmoColor);
                    if (j != m_outline.edges[i].points.Count - 1)
                    {
                        rPos2 = GetGuiPosFromOutlinPos(m_outline.edges[i].points[j + 1].position);
                        Handles.color = m_outline.edges[i].points[j].color;
                        Handles.DrawLine(rPos, rPos + (rPos2 - rPos) / 2);
                        Handles.color = m_outline.edges[i].points[j + 1].color;
                        Handles.DrawLine(rPos + (rPos2 - rPos) / 2, rPos2);
                    }
                }
            }
            if (m_edgeMode == 0)
            {
                bool isThere = false;
                for (int i = 0; i < m_outline.edges.Count; i++)
                {
                    rPos = GetGuiPosFromOutlinPos(m_outline.edges[i].points[0].position); ;
                    if (m_selectedEdges.Contains(m_outline.edges[i]))
                    {
                        if (m_selectedEdges.Count == 1 && dragging == false)
                        {
                            for (int j = 0; j < m_outline.edges[i].points.Count; j++)
                            {
                                rPos2 = GetGuiPosFromOutlinPos(m_outline.edges[i].points[j].position); ;
                                EditorGUI.DrawRect(new Rect(rPos2.x - 5, rPos2.y - 5, 11, 11), Color.green);
                            }
                        }
                        else
                        {
                            EditorGUI.DrawRect(new Rect(rPos.x - 5, rPos.y - 5, 11, 11), Color.yellow);
                        }
                        if (Event.current.type == EventType.MouseDown && editorRect.Contains(Event.current.mousePosition))
                        {
                            for (int j = 0; j < m_outline.edges[i].points.Count; j++)
                            {
                                if (j != 0 || m_selectedEdges.Count == 1)
                                {
                                    rPos2 = GetGuiPosFromOutlinPos(m_outline.edges[i].points[j].position);
                                    if (Vector2.Distance(rPos2, Event.current.mousePosition) < 10)
                                    {
                                        m_draggedPoint = m_outline.edges[i].points[j];
                                        dragging = false;
                                        selecting = false;
                                        isThere = true;
                                    }
                                }
                            }
                            if (m_draggedPoint == null && Vector2.Distance(rPos, Event.current.mousePosition) < 10)
                            {
                                edgeDrag = GetOutlinePosFromGuiPos(Event.current.mousePosition);
                                isThere = true;
                                dragging = true;
                            }
                        }
                    }
                    else
                    {
                        EditorGUI.DrawRect(new Rect(rPos.x - 5, rPos.y - 5, 11, 11), gizmoColor);
                        if (Event.current.type == EventType.MouseUp && editorRect.Contains(Event.current.mousePosition))
                        {
                            if (Vector2.Distance(rPos, Event.current.mousePosition) < 10)
                            {
                                m_selectedEdges.Add(m_outline.edges[i]);
                                isThere = true;
                            }
                        }
                        if (Event.current.type == EventType.MouseDown && editorRect.Contains(Event.current.mousePosition))
                        {
                            if (m_draggedPoint == null && Vector2.Distance(rPos, Event.current.mousePosition) < 10)
                            {
                                m_selectedEdges.Clear();
                                m_selectedEdges.Add(m_outline.edges[i]);
                                edgeDrag = SnapMousePointToGrid(Event.current.mousePosition);
                                isThere = true;
                                dragging = true;
                            }
                        }
                    }
                }
                if (Event.current.type == EventType.MouseDrag && editorRect.Contains(Event.current.mousePosition))
                {
                    if (m_draggedPoint != null)
                    {
                        m_draggedPoint.position = SnapMousePointToGrid(Event.current.mousePosition);
                    }
                    Vector2 dragDest = Vector2.zero;
                    if (dragging)
                    {
                        for (int i = 0; i < m_selectedEdges.Count; i++)
                        {
                            for (int j = 0; j < m_selectedEdges[i].points.Count; j++)
                            {
                                dragDest = (SnapMousePointToGrid(Event.current.mousePosition) - edgeDrag);
                                m_selectedEdges[i].points[j].position += dragDest;
                            }
                        }
                        edgeDrag += dragDest;
                    }
                }
                if (selecting)
                {
                    Vector2 rectWidth = Event.current.mousePosition - edgeDrag;
                    EditorGUI.DrawRect(new Rect(edgeDrag.x, edgeDrag.y, rectWidth.x, rectWidth.y), new Color32(0, 0, 100, 100));
                }

                if ((Event.current.type == EventType.MouseUp && editorRect.Contains(Event.current.mousePosition)) && isThere == false)
                {
                    if (dragging)
                    {
                        dragging = false;
                    }
                    else if (selecting)
                    {
                        Vector2 rectWidth = Event.current.mousePosition - edgeDrag;
                        rectWidth = new Vector2(Mathf.Abs(rectWidth.x), Mathf.Abs(rectWidth.y));
                        Rect selRect = new Rect(edgeDrag.x, edgeDrag.y, rectWidth.x, rectWidth.y);
                        selRect.x = (Event.current.mousePosition.x < selRect.x) ? Event.current.mousePosition.x : selRect.x;
                        selRect.y = (Event.current.mousePosition.y < selRect.y) ? Event.current.mousePosition.y : selRect.y;
                        m_selectedEdges.Clear();
                        for (int i = 0; i < m_outline.edges.Count; i++)
                        {
                            if (selRect.Contains(GetGuiPosFromOutlinPos(m_outline.edges[i].points[0].position)))
                            {
                                m_selectedEdges.Add(m_outline.edges[i]);
                            }
                        }
                        selecting = false;
                    }
                    else
                    {
                        if (m_draggedPoint == null)
                            m_selectedEdges.Clear();
                    }
                    m_draggedPoint = null;
                }

                if ((Event.current.type == EventType.MouseDown && editorRect.Contains(Event.current.mousePosition)) && isThere == false)
                {
                    dragging = false;
                    selecting = true;
                    edgeDrag = Event.current.mousePosition;
                }                
            }
            if (m_edgeMode == 1)
            {
                if (Event.current.keyCode == KeyCode.P)
                {
                    bezier[1] = SnapMousePointToGrid(Event.current.mousePosition);
                }
                rPos = GetGuiPosFromOutlinPos(bezier[0]);
                rPos2 = GetGuiPosFromOutlinPos(bezier[1]);
                EditorGUI.DrawRect(new Rect(rPos.x - 5, rPos.y - 5, 10, 10), firstPointColor);
                EditorGUI.DrawRect(new Rect(rPos2.x - 5, rPos2.y - 5, 10, 10), gizmoColor);
                Handles.DrawLine(rPos, rPos2);
                Vector2 dir = (rPos2 - rPos);
                norm = Vector3.Cross(dir, Vector3.forward).normalized;
                for (int i = 0; i <= lineLod; i++)
                {
                    rPos2 = rPos + (dir * i) / lineLod;
                    EditorGUI.DrawRect(new Rect(rPos2.x - 3, rPos2.y - 3, 6, 6), Color.Lerp(startColor, endColor, i * 1f / lineLod));
                    DrawNormal(rPos2, norm);
                }
            }
            if (m_edgeMode == 2)
            {
                for (int i = 0; i < 4; i++)
                {
                    rPos = GetGuiPosFromOutlinPos(bezier[i]);
                    EditorGUI.DrawRect(new Rect(rPos.x - 5, rPos.y - 5, 10, 10), gizmoColor);
                }
                Handles.color = firstPointColor;
                Handles.DrawLine(GetGuiPosFromOutlinPos(bezier[0]), GetGuiPosFromOutlinPos(bezier[2]));
                Handles.color = gizmoColor;
                Handles.DrawLine(GetGuiPosFromOutlinPos(bezier[1]), GetGuiPosFromOutlinPos(bezier[3]));

                Vector2 lastPoint = GetGuiPosFromOutlinPos(bezier[0]);
                Vector2 p = lastPoint;
                Vector2 nextPoint = GetGuiPosFromOutlinPos(SmartArrowUtilities.Bezier.GetPoint(bezier[0], bezier[2], bezier[3], bezier[1], 1f / curveLod));
                Vector2 lastNorm = Vector3.Cross((nextPoint - p), Vector3.forward).normalized;
                for (int i = 1; i <= curveLod; i++)
                {
                    norm = Vector3.Cross((nextPoint - p), Vector3.forward).normalized;
                    DrawNormal(p, (lastNorm + norm) / 2);
                    Handles.DrawLine(p, nextPoint);
                    lastPoint = p;
                    p = nextPoint;
                    lastNorm = norm;
                    nextPoint = GetGuiPosFromOutlinPos(SmartArrowUtilities.Bezier.GetPoint(
                        bezier[0],
                        bezier[2],
                        bezier[3],
                        bezier[1],
                        ((float)i + 1) / curveLod));
                }
                Handles.DrawLine(lastPoint, p);
                DrawNormal(p, norm);
            }
            if (m_edgeMode == 3)
            {
                rPos = GetGuiPosFromOutlinPos(bezier[0]);
                EditorGUI.DrawRect(new Rect(rPos.x - 5, rPos.y - 5, 10, 10), gizmoColor);
                for (int i = 0; i <= circleLod; i++)
                {
                    rPos = GetGuiPosFromOutlinPos(bezier[0]) + radius * 100 * zoom * new Vector2(Mathf.Cos(i * 1f / circleLod * circleLength * Mathf.Deg2Rad + circleStart * Mathf.Deg2Rad), Mathf.Sin(i * 1f / circleLod * circleLength * Mathf.Deg2Rad + circleStart * Mathf.Deg2Rad));
                    EditorGUI.DrawRect(new Rect(rPos.x - 3, rPos.y - 3, 6, 6), Color.Lerp(startColor, endColor, i * 1f / circleLod));
                    DrawNormal(rPos, rPos - GetGuiPosFromOutlinPos(bezier[0]));
                }
                
            }
            if (Event.current.type == EventType.MouseDown && editorRect.Contains(Event.current.mousePosition))
            {
                dragBez = -1;
                for (int i = 0; i < 4; i++)
                {
                    if (Vector2.Distance(new Vector2(bezier[i].x, -bezier[i].y), (Event.current.mousePosition - new Vector2(editorRect.center.x, editorRect.center.y)) / (100 * zoom)) < 0.1f)
                    {
                        dragBez = i;
                    }
                }
            }
            if (Event.current.type == EventType.MouseDrag && dragBez != -1 && editorRect.Contains(Event.current.mousePosition))
            {
                MoveCurvePoint(dragBez, SnapMousePointToGrid(Event.current.mousePosition));
            }
        }


        public void OnGUI()
        {
            #region editorBounds
            editorRect = new Rect(0, 0, position.width - menuWidth, position.width - menuWidth);
            if (editorRect.width > position.height) editorRect.width = editorRect.height = position.height;
            EditorGUI.DrawRect(editorRect, editorBackgroundColor);
            #endregion
            RenderWindowGizmos();
            RenderWindowMenu();
        }
        public void MoveCurvePoint(int index,Vector2 newPosition)
        {
            Vector2 beforeBez0 = bezier[0];
            Vector2 beforeBez1 = bezier[1];
            bezier[index] = newPosition;
            if (index < 2)
            {
                Vector3 temp = Quaternion.FromToRotation(beforeBez1 - beforeBez0, bezier[1] - bezier[0]) * (bezier[2] - beforeBez0);
                bezier[2] = bezier[0] + new Vector2(temp.x, temp.y);
                temp = Quaternion.FromToRotation(beforeBez1 - beforeBez0, bezier[1] - bezier[0]) * (bezier[3] - beforeBez1);
                bezier[3] = bezier[1] + new Vector2(temp.x, temp.y);
            }
            
        }
        Vector2 SnapCenterPosToGrid(Vector2 pos)
        {
            Vector2 ret = (new Vector2(Mathf.Round(pos.x / grid) * grid, Mathf.Round(pos.y / grid) * grid)); 
            ret.x = Mathf.Round(ret.x * 10000) / 10000;
            ret.y = Mathf.Round(ret.y * 10000) / 10000;
            return ret;
        }
        Vector2 SnapMousePointToGrid(Vector2 pos)
        {
            return SnapCenterPosToGrid(GetOutlinePosFromGuiPos(pos));
        }
        void DrawNormal(Vector2 pos,Vector2 norm)
        {
            norm = norm.normalized * zoom * 20;
            Handles.color = gizmoColor;
            Handles.DrawLine(pos, pos+norm);
        }
        Vector2 GetOutlinePosFromGuiPos(Vector2 position)
        {
            Vector2 ret = (position - editorRect.center) / (zoom * 100);
            ret.y = -ret.y;
            ret.x = Mathf.Round(ret.x * 10000) / 10000;
            ret.y = Mathf.Round(ret.y * 10000) / 10000;
            return ret;
        }
        Vector2 GetGuiPosFromOutlinPos(Vector2 pos)
        {
            Vector2 ret = editorRect.center + zoom * 100 * new Vector2(pos.x, -pos.y);
            ret.x = Mathf.Round(ret.x * 10000) / 10000;
            ret.y = Mathf.Round(ret.y * 10000) / 10000;
            return ret;
        }
    }
}