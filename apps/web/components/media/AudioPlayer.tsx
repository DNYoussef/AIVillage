import React, { useRef, useState, useEffect } from 'react';
import { MediaContent } from '../../types';
import './AudioPlayer.css';

interface AudioPlayerProps {
  content: MediaContent;
  isPreview?: boolean;
  fogEnabled?: boolean;
  onError?: (error: string) => void;
}

export const AudioPlayer: React.FC<AudioPlayerProps> = ({
  content,
  isPreview = false,
  fogEnabled = true,
  onError,
}) => {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleTimeUpdate = () => setCurrentTime(audio.currentTime);
    const handleLoadedMetadata = () => setDuration(audio.duration);
    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleEnded = () => setIsPlaying(false);
    const handleError = () => {
      setHasError(true);
      onError?.('Failed to load audio');
    };

    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('error', handleError);

    return () => {
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('error', handleError);
    };
  }, [onError]);

  const togglePlayPause = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    if (!audio) return;

    const newTime = parseFloat(e.target.value);
    audio.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    const newVolume = parseFloat(e.target.value);

    if (audio) {
      audio.volume = newVolume;
    }
    setVolume(newVolume);
  };

  const skipTime = (seconds: number) => {
    const audio = audioRef.current;
    if (!audio) return;

    audio.currentTime = Math.max(0, Math.min(duration, audio.currentTime + seconds));
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const getProgress = () => {
    return duration > 0 ? (currentTime / duration) * 100 : 0;
  };

  if (hasError) {
    return (
      <div className="audio-error">
        <div className="error-icon">🎧</div>
        <p>Failed to load audio</p>
        <button onClick={() => window.location.reload()} className="retry-btn">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className={`audio-player ${isPreview ? 'preview' : ''}`}>
      <audio
        ref={audioRef}
        src={content.url}
        preload="metadata"
      />

      <div className="audio-controls">
        {!isPreview && (
          <button
            onClick={() => skipTime(-10)}
            className="skip-btn"
            title="Skip back 10s"
          >
            ⏮️
          </button>
        )}

        <button
          onClick={togglePlayPause}
          className="play-pause-btn"
          title={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? '⏸️' : '▶️'}
        </button>

        {!isPreview && (
          <button
            onClick={() => skipTime(10)}
            className="skip-btn"
            title="Skip forward 10s"
          >
            ⏭️
          </button>
        )}

        <div className="time-display">
          {formatTime(currentTime)}
          {duration > 0 && ` / ${formatTime(duration)}`}
        </div>
      </div>

      {!isPreview && (
        <div className="audio-progress">
          <input
            type="range"
            min={0}
            max={duration || 0}
            value={currentTime}
            onChange={handleSeek}
            className="progress-bar"
          />
          <div
            className="progress-fill"
            style={{ width: `${getProgress()}%` }}
          />
        </div>
      )}

      {isPreview && (
        <div className="preview-progress">
          <div
            className="preview-progress-fill"
            style={{ width: `${getProgress()}%` }}
          />
        </div>
      )}

      {!isPreview && (
        <div className="volume-control">
          <button className="volume-btn">
            {volume === 0 ? '🔇' : volume < 0.5 ? '🔉' : '🔊'}
          </button>
          <input
            type="range"
            min={0}
            max={1}
            step={0.1}
            value={volume}
            onChange={handleVolumeChange}
            className="volume-bar"
          />
        </div>
      )}

      {!isPreview && content.metadata && (
        <div className="audio-metadata">
          <span className="metadata-item">
            {content.metadata.format}
          </span>
          {content.metadata.size && (
            <span className="metadata-item">
              {(content.metadata.size / 1024 / 1024).toFixed(1)} MB
            </span>
          )}
          {content.metadata.duration && (
            <span className="metadata-item">
              {formatTime(content.metadata.duration)}
            </span>
          )}
          {fogEnabled && (
            <span className="metadata-item fog-optimized">
              ☁️ Fog Streaming
            </span>
          )}
        </div>
      )}
    </div>
  );
};
