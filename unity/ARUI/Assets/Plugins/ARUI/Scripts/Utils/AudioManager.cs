using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Audio;
using Microsoft.MixedReality.Toolkit.Input;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

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
    voiceConfirmation = 8,
    actionConfirmation = 9,
}

/// <summary>
/// Plays localized audio feedback at run-time
/// Used for playing sound effects (e.g., notifications or warnings) and
/// for text to speech for guidance. 
/// If the user says 'stop', the sound effect is immediately stopped for this playing instance
/// </summary>
public class AudioManager : Singleton<AudioManager>, IMixedRealitySpeechHandler
{
    ///** MRTK build-in text to speech (ONLY WORKS IN BUILD)
    private TextToSpeech _tTos;                                          /// <TTS from MRTK
    private Dictionary<SoundType, AudioSource> _typeToSound;             /// <maps soundtype to audio file
    private Dictionary<SoundType, string> _soundTypeToPathMapping = new Dictionary<SoundType, string>()
    {
        { SoundType.notification,StringResources.NotificationSound_path},
        { SoundType.confirmation, StringResources.BtnConfirmationSound_path},
        { SoundType.taskDone,StringResources.NextTaskSound_path},
        { SoundType.moveStart,StringResources.MoveStart_path},
        { SoundType.moveEnd,StringResources.MoveEnd_path},
        { SoundType.select,StringResources.SelectSound_path},
        { SoundType.warning,StringResources.WarningSound_path},
        { SoundType.voiceConfirmation,StringResources.VoiceConfirmation_path},
        { SoundType.actionConfirmation, StringResources.ActionConfirmation_path }
    };

    private List<AudioSource> _currentlyPlayingSound = null;             /// <Reference to all sounds that are currently playing
    private AudioSource _currentlyPlayingText = null;                    /// <REference to the tts sound that is playing. (only one possible)
   
    ///** Orb is talking feedback
    private float _updateTime = 0f;
    private float _updateDelay = 0.04f;
    private float[] _spectrumData = new float[64];

    ///** Mute audio feedback for task guidance
    private bool _isMute = false;                                        /// <if true, task instructions or dialogue system audio feedback is not played. BUT system sound is.
    public bool IsMute => _isMute;

    ///** Voice commands callback storage
    private Dictionary<string, UnityAction> _keywordToActionMapping = new Dictionary<string, UnityAction>();

    public void Awake() => CoreServices.InputSystem?.RegisterHandler<IMixedRealitySpeechHandler>(this);

    private void Start()
    {
        GameObject _tTosGO = new GameObject("***ARUI-TextToSpeechSource");
        _tTosGO.transform.parent = transform;
        _tTosGO.transform.position = transform.position;
        _tTos = _tTosGO.AddComponent<TextToSpeech>();

        _currentlyPlayingSound = new List<AudioSource>();

        RegisterKeyword("stop", () => UserSaidStopAction());
    }

    /// <summary>
    /// Speech-To-Text for the task. Plays the text at the orb's position 
    /// and stops any other currently playing text instructions
    /// NOTE: THIS ONLY WORKS IN BUILD (NOT HOLOGRAPHIC REMOTING)
    /// </summary>
    /// <param name="text">The text that is turned into audion and played</param>
    public void PlayAndShowDialogue(string utterance, string answer, float timeout = 30)
    {
        if (utterance==null)
        {
            utterance = string.Empty;
        }

        if (!_isMute)
            StartCoroutine(PlayTextDialogue(Orb.Instance.orbTransform.position, utterance, answer, timeout));
    }

    /// <summary>
    /// Speech-To-Text for the task. Plays the text at the orb's position 
    /// and stops any other currently playing text instructions
    /// NOTE: THIS ONLY WORKS IN BUILD (NOT HOLOGRAPHIC REMOTING)
    /// </summary>
    /// <param name="text">The text that is turned into audion and played</param>
    public void PlayAndShowMessage(string message, float timeout = 30)
    {
        if (!_isMute)
            StartCoroutine(PlayTextDialogue(Orb.Instance.orbTransform.position, "", message, timeout));
    }

    /// <summary>
    /// Speech-To-Text for the task. Plays the text at the orb's position 
    /// and stops any other currently playing text instructions
    /// NOTE: THIS ONLY WORKS IN BUILD (NOT HOLOGRAPHIC REMOTING)
    /// </summary>
    /// <param name="text">The text that is turned into audion and played</param>
    public void PlayMessage(string text, float timeout = 30)
    {
        if (!_isMute)
            StartCoroutine(PlayTextLocalized(Orb.Instance.orbTransform.position, text));
    }

    public void PlayTextIfNotPlaying(string text)
    {
        if (!_isMute && _currentlyPlayingText != null && _currentlyPlayingText == false && _tTos.AudioSource.isPlaying == false)
        {
            StartCoroutine(PlayTextLocalized(Orb.Instance.orbTransform.position, text));
        }
    }

    /// <summary>
    /// Plays a sound effect from a given position
    /// </summary>
    /// <param name="pos">Sound effect is played form this position</param>
    /// <param name="type">Type of sound effect that should be played</param>
    public void PlaySound(Vector3 pos, SoundType type) => StartCoroutine(PlaySoundLocalized(pos, type));

    /// <summary>
    /// Mute audio feedback for task guidance, but NOT sound effects
    /// </summary>
    /// <param name="mute"></param>
    public void MuteAudio(bool mute)
    {
        if (mute && _currentlyPlayingText != null)
        {
            _tTos.AudioSource.Stop();
            _tTos.StopSpeaking();
            _currentlyPlayingText.Stop();
        }

        _isMute = mute;
    }

    /// <summary>
    /// Initialize the sound effect library
    /// </summary>
    private void InitIfNeeded()
    {
        _typeToSound = new Dictionary<SoundType, AudioSource>();

        //Load sound resources
        foreach (SoundType type in _soundTypeToPathMapping.Keys)
        {
            AudioSource sound = new GameObject(_soundTypeToPathMapping[type]).AddComponent<AudioSource>();
            sound.gameObject.name = "***ARUI-"+ _soundTypeToPathMapping[type];
            sound.clip = Resources.Load(_soundTypeToPathMapping[type]) as AudioClip;
            sound.transform.parent = transform;
            sound.spatialize = true;
            sound.maxDistance = 10f;
            sound.spatialBlend = 1;
            sound.loop = false;
            sound.playOnAwake = false;
            _typeToSound.Add(type, sound);
        }
    }

    /// <summary>
    /// Moves an audio source at the given position pos and plays it.
    /// </summary>
    /// <param name="pos">Sound effect is played form this position</param>
    /// <param name="type">Type of sound effect that should be played</param>
    /// <returns></returns>
    private IEnumerator PlaySoundLocalized(Vector3 pos, SoundType type)
    {
        if (_typeToSound == null) InitIfNeeded();

        GameObject tempCopy = Instantiate(_typeToSound[type].gameObject);
        AudioSource tempCopyAudio = tempCopy.GetComponent<AudioSource>();

        tempCopyAudio.transform.position = pos;

        yield return new WaitForEndOfFrame();

        tempCopyAudio.Play();
        _currentlyPlayingSound.Add(tempCopyAudio);

        while (tempCopyAudio.isPlaying)
        {
            yield return new WaitForEndOfFrame();
        }

        _currentlyPlayingSound.Remove(tempCopyAudio);
        Destroy(tempCopyAudio.gameObject);
    }

    /// <summary>
    /// Transforms the given text to audio using MRTK's TTS and plays it for the user. Assumes that the audio is not mute.
    /// If there is already one TTS file playing, it will be interrupted.
    /// The orb's mouth will show visual feedback while the TTS is playing.
    /// </summary>
    /// <param name="pos">the position the audio should be played</param>
    /// <param name="text">The text that should be spoken by the TTS</param>
    /// <returns></returns>
    private IEnumerator PlayTextLocalized(Vector3 pos, String text)
    {
        if (_currentlyPlayingText != null)
        {
            _tTos.AudioSource.Stop();
            _tTos.StopSpeaking();
            _currentlyPlayingText.Stop();
        }

        yield return new WaitForEndOfFrame();

        _tTos.gameObject.transform.position = pos;

        yield return new WaitForEndOfFrame();

        string cappedText = Utils.GetCappedText(text, 50);
        AngelARUI.Instance.DebugLogMessage("Orb says: " + cappedText, true);
        _tTos.StartSpeaking(cappedText);
        _currentlyPlayingText = _tTos.AudioSource;

        yield return new WaitForEndOfFrame();

        while (!_tTos.AudioSource.isPlaying)
            yield return new WaitForEndOfFrame();

        while (_tTos.AudioSource.isPlaying)
        {
            if (_updateTime > Time.time)
                yield return new WaitForEndOfFrame();

            _tTos.AudioSource.GetSpectrumData(_spectrumData, 0, FFTWindow.BlackmanHarris);
            _updateTime = Time.time + _updateDelay;

            var barHeight = Mathf.Clamp(_spectrumData[1], 0.001f, 1f);
            Orb.Instance.MouthScale = barHeight;

            yield return new WaitForEndOfFrame();
        }

        yield return new WaitForEndOfFrame();

        Orb.Instance.MouthScale = 0;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="utterance"></param>
    /// <param name="answer"></param>
    /// <returns></returns>
    private IEnumerator PlayTextDialogue(Vector3 pos, String utterance, string answer, float timeout)
    {
        if (_currentlyPlayingText!= null)
        {
            _tTos.AudioSource.Stop();
            _tTos.StopSpeaking();
            _currentlyPlayingText.Stop();
            Orb.Instance.SetDialogueActive(false);
        }
            
        yield return new WaitForEndOfFrame();

        _tTos.gameObject.transform.position = pos;
        Orb.Instance.SetDialogueActive(true);
        Orb.Instance.SetDialogueText(utterance, answer, timeout);

        yield return new WaitForEndOfFrame();

        string cappedText = Utils.GetCappedText(answer, 50);
        AngelARUI.Instance.DebugLogMessage("Orb says: " + cappedText, true);
        _tTos.StartSpeaking(cappedText);
        _currentlyPlayingText = _tTos.AudioSource;

        yield return new WaitForEndOfFrame();

        while (!_tTos.AudioSource.isPlaying)
            yield return new WaitForEndOfFrame();

        while (_tTos.AudioSource.isPlaying)
        {
            if (_updateTime > Time.time)
                yield return new WaitForEndOfFrame();

            _tTos.AudioSource.GetSpectrumData(_spectrumData, 0, FFTWindow.BlackmanHarris);
            _updateTime = Time.time + _updateDelay;

            var barHeight = Mathf.Clamp(_spectrumData[1], 0.001f, 1f);
            Orb.Instance.MouthScale = barHeight;

            yield return new WaitForEndOfFrame();
        } 
        
        yield return new WaitForEndOfFrame();

        Orb.Instance.MouthScale = 0; 
    }

    #region Keyword Detection

    /// <summary>
    /// This function is called if user said 'stop'
    /// Stops any ongoing TTS at the orb
    /// </summary>
    private void UserSaidStopAction()
    {
        if (_currentlyPlayingText != null)
            _currentlyPlayingText.Stop();

        if (_tTos)
            _tTos.StopSpeaking();

        AngelARUI.Instance.DebugLogMessage("Orb stopped speaking", true);
    }

    /// <summary>
    /// Handles Speech input event from MRTK, for now we only listen to the 
    /// keyword 'stop', so the orb stops talking immediately.
    /// </summary>
    /// <param name="eventData"></param>
    public void OnSpeechKeywordRecognized(SpeechEventData eventData)
    {
        AngelARUI.Instance.DebugLogMessage("Detected keyword: " + eventData.Command.Keyword.ToLower(), true);
        foreach (string keyword in _keywordToActionMapping.Keys)
        {
            if (eventData.Command.Keyword.ToLower().Equals(keyword.ToLower()))
            {
                _keywordToActionMapping[keyword].Invoke();
                PlaySound(Orb.Instance.orbTransform.position, SoundType.voiceConfirmation);
            }
        }
    }

    public bool RegisterKeyword(string keyword, UnityAction keyWordDetectedCallBack)
    {
        if (keyword != null && keyword.Length >= 2)
        {
            _keywordToActionMapping.Add(keyword, keyWordDetectedCallBack);
            AngelARUI.Instance.DebugLogMessage("Successfully registered keyword '" + keyword + "'.", true);
            return true;
        }

        AngelARUI.Instance.DebugLogMessage("Keyword '" + keyword + "'is already registered or not long enough. (>=2)", true);
        return false;
    }

    #endregion
}
