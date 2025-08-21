import React, { useRef, useState, useEffect } from 'react';
import { MediaContent } from '../../types';
import './VideoPlayer.css';

interface VideoPlayerProps {
  content: MediaContent;
  isPreview?: boolean;
  fogEnabled?: boolean;
  onError?: (error: string) => void;
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({
  content,
  isPreview = false,
  fogEnabled = true,
  onError,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleTimeUpdate = () => setCurrentTime(video.currentTime);
    const handleLoadedMetadata = () => setDuration(video.duration);
    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleError = () => {
      setHasError(true);
      onError?.('Failed to load video');
    };

    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);
    video.addEventListener('error', handleError);

    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
      video.removeEventListener('error', handleError);
    };
  }, [onError]);

  const togglePlayPause = () => {
    const video = videoRef.current;
    if (!video) return;

    if (isPlaying) {
      video.pause();
    } else {
      video.play();
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current;
    if (!video) return;

    const newTime = parseFloat(e.target.value);
    video.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current;
    const newVolume = parseFloat(e.target.value);

    if (video) {
      video.volume = newVolume;
    }
    setVolume(newVolume);
  };

  const toggleFullscreen = () => {
    const video = videoRef.current;
    if (!video) return;

    if (!isFullscreen) {
      video.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  if (hasError) {
    return (
      <div className="video-error">
        <div className="error-icon">🎥</div>
        <p>Failed to load video</p>
        <button onClick={() => window.location.reload()} className="retry-btn">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className={`video-player ${isPreview ? 'preview' : ''}`}>
      <div className="video-container">
        <video
          ref={videoRef}
          src={content.url}
          poster={content.thumbnail}
          className="video-element"
          preload="metadata"
          onClick={togglePlayPause}
        />

        {!isPreview && (
          <div className="video-controls">
            <button
              onClick={togglePlayPause}
              className="play-pause-btn"
              title={isPlaying ? 'Pause' : 'Play'}
            >
              {isPlaying ? '⏸️' : '▶️'}
            </button>

            <div className="time-display">
              {formatTime(currentTime)} / {formatTime(duration)}
            </div>

            <input
              type="range"
              min={0}
              max={duration || 0}
              value={currentTime}
              onChange={handleSeek}
              className="seek-bar"
            />

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

            <button
              onClick={toggleFullscreen}
              className="fullscreen-btn"
              title="Fullscreen"
            >
              🔲
            </button>
          </div>
        )}

        {isPreview && (
          <div className="preview-overlay">
            <button className="preview-play-btn" onClick={togglePlayPause}>
              ▶️
            </button>
            {content.metadata?.duration && (
              <div className="video-duration">
                {formatTime(content.metadata.duration)}
              </div>
            )}
          </div>
        )}
      </div>

      {!isPreview && content.metadata && (
        <div className="video-metadata">
          {content.metadata.dimensions && (
            <span className="metadata-item">
              {content.metadata.dimensions.width} × {content.metadata.dimensions.height}
            </span>
          )}
          <span className="metadata-item">
            {content.metadata.format}
          </span>
          {content.metadata.size && (
            <span className="metadata-item">
              {(content.metadata.size / 1024 / 1024).toFixed(1)} MB
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
