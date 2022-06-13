using System.Collections.Generic;
using UnityEditor;
using UnityEditorInternal;
using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    public class ArrowPathEditor : Editor
    {
        protected static bool renderPath = true;
        protected static bool renderControls = true;
        protected static bool updatePath = true; 
        protected const float handleSize = 0.1f;
        protected const float pickSize = 0.08f;
        public virtual void OnSceneGUI(SceneView sceneView)
        {
            if (renderControls)
            {
                RenderControls();
            }
            if (renderPath)
            {
                RenderPath();
            }
        }
        public virtual void ObstacleSettings()
        {
            EditorGUILayout.PropertyField(serializedObject.FindProperty("obstacleCheck"));
            if (((ArrowPath)target).obstacleCheck)
            {
                EditorGUILayout.PropertyField(serializedObject.FindProperty("obstacleCheckLayer"));
                EditorGUILayout.PropertyField(serializedObject.FindProperty("distanceFromObstacle"));
                EditorGUILayout.PropertyField(serializedObject.FindProperty("obstacleCheckRayLength"));
            }
        }
        public virtual void PathSettings()
        {
            GUILayout.BeginHorizontal(EditorStyles.toolbar);
            EditorGUILayout.LabelField("Path Options", EditorStyles.boldLabel);
            if (GUILayout.Button("Actions", EditorStyles.toolbarDropDown))
            {
                GenericMenu toolsMenu = new GenericMenu();
                toolsMenu.AddItem(new GUIContent("Update path"), updatePath, () =>
                {
                    updatePath = !updatePath;
                });
                toolsMenu.AddItem(new GUIContent("Render path gizmos"), renderPath, () =>
                {
                    renderPath = !renderPath;
                });
                toolsMenu.AddItem(new GUIContent("Render control gizmos"), renderControls, () =>
                {
                    renderControls = !renderControls;
                });
                toolsMenu.DropDown(new Rect(Event.current.mousePosition.x - 100, Event.current.mousePosition.y, 0, 16));
                GUIUtility.ExitGUI();
            }
            GUILayout.EndHorizontal();
        }
        public virtual void RenderControls()
        {

        }
        public virtual void RenderPath()
        {
            if(updatePath)((ArrowPath)target).CalculatePath();
            List<Vector3> path = ((ArrowPath) target).GetCalculatedPath();
            List<Quaternion> rotation = ((ArrowPath)target).GetCalculatedRotation();
            Vector3 p = Vector3.zero;
            Quaternion r = Quaternion.identity;
            if (((ArrowPath)target).local)
            {
                p = ((ArrowPath)target).transform.position;
                p = ((ArrowPath)target).transform.rotation * -p + p;
            }
            for (int i = 0; i < path.Count-1; i++)
            {
                Handles.color = Color.blue;
                Handles.DrawLine( path[i] + p, p + path[i] + rotation[i]*Vector3.up,1);
                Handles.DrawLine( path[i] + p, p + path[i+1] );
            }
        }
    }
}