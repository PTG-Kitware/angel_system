using DilmerGames.Core.Singletons;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum SoundType
{
    notification = 0,
    confirmation = 1,
    bell = 2,
    taskDone=3,
    moveStart=4,
    moveEnd=5,
}

public class AudioManager : Singleton<AudioManager>
{
    private Dictionary<SoundType, AudioSource> typeToSound;

    private List<string> soundTypeToPathMapping = new List<string>()
    {
        StringResources.notificationSound_path,
        StringResources.confirmationSound_path,
        StringResources.bellsound_path,
        StringResources.nextTaskSound_path,
        StringResources.moveStart_path,
        StringResources.moveEnd_path,
    };
    
    private void InitIfNeeded()
    {
        typeToSound = new Dictionary<SoundType, AudioSource>();

        //Load sound resources
        for (int i = 0; i < soundTypeToPathMapping.Count; i++)
        {
            AudioSource sound = new GameObject(soundTypeToPathMapping[i]).AddComponent<AudioSource>();
            sound.clip = Resources.Load(soundTypeToPathMapping[i]) as AudioClip;
            sound.transform.parent = transform;
            typeToSound.Add((SoundType)i, sound);
        }
    }

    public void PlaySound(Vector3 pos, SoundType type) => StartCoroutine(Play(pos, type));

    public void PlaySound(Vector3 pos, AudioClip clip) => StartCoroutine(Play(pos, clip));

    private IEnumerator Play(Vector3 pos, SoundType type)
    {
        if (typeToSound == null) InitIfNeeded();
        
        typeToSound[type].transform.position = pos;

        yield return new WaitForEndOfFrame();

        typeToSound[type].Play();

        while(typeToSound[type].isPlaying)
        {
           yield return new WaitForEndOfFrame();
        }

        typeToSound[type].transform.position = Vector3.zero;
    }

    private IEnumerator Play(Vector3 pos, AudioClip clip)
    {
        if (typeToSound == null) InitIfNeeded();
        
        GameObject temp_audio = new GameObject("temp_audio");
        temp_audio.transform.position = pos;

        AudioSource audioSource = temp_audio.AddComponent<AudioSource>();
        audioSource.clip = clip;

        yield return new WaitForEndOfFrame();

        audioSource.Play();

        while (audioSource.isPlaying)
        {
            yield return new WaitForEndOfFrame();
        }

        yield return new WaitForEndOfFrame();

        Destroy(temp_audio);
    }
}
