using System;
using System.Collections;
using System.Collections.Generic;
using DilmerGames.Core.Singletons;
using UnityEngine;

public class EntityManager : Singleton<EntityManager>
{
    Dictionary<string, Entity> registry = new Dictionary<string, Entity>();

    public String PrintDict()
    {
        String output = registry.Count+"\n";
        foreach (KeyValuePair<string, Entity> kvp in registry)
        {
            output += string.Format("Key = {0}, Value = {1}\n", kvp.Key, kvp.Value);
        }
        return output;
    }

    public bool contains(string id) => registry.ContainsKey(id);

    public ArrowEntity CreateArrowEntity(string id)
    {
        if (contains(id)) Remove(id);
        registry.Add(id, new GameObject(id).AddComponent<ArrowEntity>());
        return registry[id] as ArrowEntity;
    }

    public DetectedEntity AddDetectedEntity(string id, string text)
    {
        DetectedEntity detectedEntity = new GameObject(text).AddComponent<DetectedEntity>();
        registry.Add(id, detectedEntity);
        return detectedEntity;
    }

    public void Remove(string id)
    {
        if (!registry.ContainsKey(id))
        {
            Debug.Log(id + " - id was not present.");
            return;
        }

        if (registry[id] != null)
        {
            Destroy(registry[id].gameObject);
        }
        registry.Remove(id);
    }

    public Entity Get(string id) => registry[id];


}
