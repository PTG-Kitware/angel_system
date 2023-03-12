using DilmerGames.Core.Singletons;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Audio;
using Microsoft.MixedReality.Toolkit.Input;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum SoundType
{
    notification = 0,
    confirmation = 1,
    bell = 2,
    taskDone = 3,
    moveStart = 4,
    moveEnd = 5,
    select = 6,
    warning = 7,
}

/// <summary>
/// Plays localized audio feedback at run-time
/// </summary>
public class AudioManager : Singleton<AudioManager>, IMixedRealitySpeechHandler
{
    private TextToSpeech _tTos;                                          /// <TTS from MRTK
    private Dictionary<SoundType, AudioSource> typeToSound;             /// <maps soundtype to audio file
    private Dictionary<SoundType, string> soundTypeToPathMapping = new Dictionary<SoundType, string>()
    {
        { SoundType.notification,StringResources.notificationSound_path},
        { SoundType.confirmation, StringResources.confirmationSound_path},
        { SoundType.bell,StringResources.bellsound_path},
        { SoundType.taskDone,StringResources.nextTaskSound_path},
        { SoundType.moveStart,StringResources.moveStart_path},
        { SoundType.moveEnd,StringResources.moveEnd_path},
        { SoundType.select,StringResources.selectSound_path},
        { SoundType.warning,StringResources.warningSound_path}
    };

    private List<AudioSource> _currentlyPlayingSound = null;             /// <Reference to all sounds that are currently playing
    private AudioSource _currentlyPlayingText = null;                    /// <REference to the tts sound that is playing. (only one possible)

    private bool _isMute = false;                                        /// <if true, task instructions or dialogue system audio feedback is not played. BUT system sound is.
    public bool IsMute { get { return _isMute; }
    }

    public void Awake() => CoreServices.InputSystem?.RegisterHandler<IMixedRealitySpeechHandler>(this);

    private void Start()
    {
        GameObject tmp = new GameObject("TextToSpeechSource");
        tmp.transform.parent = transform;
        tmp.transform.position = transform.position;
        _tTos = tmp.gameObject.AddComponent<TextToSpeech>();

        _currentlyPlayingSound = new List<AudioSource>();
    }

    /// <summary>
    /// Speech-To-Text for the task
    /// </summary>
    /// <param name="text"></param>
    public void PlayText(string text)
    {
        if (!_isMute)
            StartCoroutine(PlayTextLocalized(Orb.Instance.transform.position, text));
    }

    /// <summary>
    /// Plays a sound effect from a certain position
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="type"></param>
    public void PlaySound(Vector3 pos, SoundType type) => StartCoroutine(PlaySoundLocalized(pos, type));

    /// <summary>
    /// Mute audio feedback for task guidance
    /// </summary>
    /// <param name="mute"></param>
    public void MuteAudio(bool mute)
    {
        if (mute && _currentlyPlayingText != null)
        {
            _tTos.StopSpeaking();
            _currentlyPlayingText.Stop();
            _currentlyPlayingText = null;
        }

        _isMute = mute;
    }

    /// <summary>
    /// Immediately stops the audio instructions
    /// </summary>
    public void ImmediatelyStopSpeaking()
    {
        if (_currentlyPlayingText!=null)
            _currentlyPlayingText.Stop();

        if (_tTos) 
            _tTos.StopSpeaking();

        _currentlyPlayingText = null;
    }


    /// <summary>
    /// Initialize the sound library
    /// </summary>
    private void InitIfNeeded()
    {
        typeToSound = new Dictionary<SoundType, AudioSource>();

        //Load sound resources
        foreach (SoundType type in soundTypeToPathMapping.Keys)
        {
            AudioSource sound = new GameObject(soundTypeToPathMapping[type]).AddComponent<AudioSource>();
            sound.clip = Resources.Load(soundTypeToPathMapping[type]) as AudioClip;
            sound.transform.parent = transform;
            typeToSound.Add(type, sound);
        }
    }

    /// <summary>
    /// Moves the audio
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="type"></param>
    /// <returns></returns>
    private IEnumerator PlaySoundLocalized(Vector3 pos, SoundType type)
    {
        if (typeToSound == null) InitIfNeeded();

        GameObject tempCopy = Instantiate(typeToSound[type].gameObject);
        AudioSource tempCopyAudio = tempCopy.GetComponent<AudioSource>();

        tempCopyAudio.transform.position = pos;

        yield return new WaitForEndOfFrame();

        tempCopyAudio.Play();
        _currentlyPlayingSound.Add(tempCopyAudio);

        while (tempCopyAudio.isPlaying)
            yield return new WaitForEndOfFrame();

        _currentlyPlayingSound.Remove(tempCopyAudio);
        Destroy(tempCopyAudio.gameObject);
    }

    /// <summary>
    /// Transforms the given text to audio using MRTK's TTS and plays it for the user. Assumes that the audio is not mute.
    /// If there is already one TTS file playing, it will be interrupted.
    /// </summary>
    /// <param name="pos">the position the audio should be played</param>
    /// <param name="text">The text that should be spoken by the TTS</param>
    /// <returns></returns>
    private IEnumerator PlayTextLocalized(Vector3 pos, String text)
    {
        if (_currentlyPlayingText!= null)
        {
            _tTos.StopSpeaking();
            _currentlyPlayingText.Stop();
        }
            
        yield return new WaitForEndOfFrame();

        _tTos.gameObject.transform.position = pos;

        yield return new WaitForEndOfFrame();

        var msg = string.Format(text, _tTos.Voice.ToString());
        _tTos.StartSpeaking(text);
        _currentlyPlayingText = _tTos.AudioSource;

        while (_tTos.IsSpeaking())
            yield return new WaitForEndOfFrame();

        _currentlyPlayingText = null;
    }

    /// <summary>
    /// Handles Speech input event from MRTK 
    /// </summary>
    /// <param name="eventData"></param>
    public void OnSpeechKeywordRecognized(SpeechEventData eventData)
    {
        if (eventData.Command.Keyword.ToLower().Equals("stop"))
            ImmediatelyStopSpeaking();

        Debug.Log("Stop speaking.");
    }
}
