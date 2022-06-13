using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum Type
{
    Actual,
    Arrow,
    Notification
}

/// <summary>
// Every object in view must be an entity
/// </summary>
public class Entity : MonoBehaviour
{
    public string id; // The ID/name of the entity
    public Type entityType; // The type of the entity
    public List<string> timestamps; // The time stamp of occurences of the entity

    public string label;
    public List<Vector3> boundingBox;
}
