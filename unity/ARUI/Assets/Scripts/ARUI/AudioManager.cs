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
}

public class AudioManager : Singleton<AudioManager>
{
    private Dictionary<SoundType, AudioSource> typeToSound;

    private void InitIfNeeded()
    {
        typeToSound = new Dictionary<SoundType, AudioSource>();

        //Load sound resources
        AudioSource notificationsound = new GameObject("notification_sound").AddComponent<AudioSource>();
        notificationsound.clip = Resources.Load(StringResources.notificationSound_path) as AudioClip;
        notificationsound.transform.parent = transform;
        typeToSound.Add(SoundType.notification, notificationsound);

        AudioSource confirmationSound = new GameObject("confirmation_sound").AddComponent<AudioSource>();
        confirmationSound.clip = Resources.Load(StringResources.confirmationSound_path) as AudioClip;
        confirmationSound.transform.parent = transform;
        typeToSound.Add(SoundType.confirmation, confirmationSound);

        AudioSource bellSound = new GameObject("bell_sound").AddComponent<AudioSource>();
        bellSound.clip = Resources.Load(StringResources.bellsound_path) as AudioClip;
        bellSound.transform.parent = transform;
        typeToSound.Add(SoundType.bell, bellSound);
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
