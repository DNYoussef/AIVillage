import React, { useState, useRef, useEffect } from 'react';
import { MediaContent } from '../../types';
import { useMediaService } from '../../hooks/useMediaService';
import { ImageViewer } from './ImageViewer';
import { VideoPlayer } from './VideoPlayer';
import { TextViewer } from './TextViewer';
import { AudioPlayer } from './AudioPlayer';
import { MediaControls } from './MediaControls';
import { FullscreenView } from './FullscreenView';
import './MediaDisplayEngine.css';

interface MediaDisplayEngineProps {
  contentId?: string;
  initialContent?: MediaContent[];
  fogEnabled?: boolean;
  onContentLoad?: (content: MediaContent) => void;
  onError?: (error: string) => void;
}

export const MediaDisplayEngine: React.FC<MediaDisplayEngineProps> = ({
  contentId,
  initialContent = [],
  fogEnabled = true,
  onContentLoad,
  onError
}) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [displayMode, setDisplayMode] = useState<'single' | 'grid' | 'carousel'>('single');
  const [filterType, setFilterType] = useState<'all' | 'image' | 'video' | 'text' | 'audio'>('all');

  const containerRef = useRef<HTMLDivElement>(null);

  const {
    mediaContent,
    isLoading,
    error,
    loadMedia,
    downloadMedia,
    streamingStats,
    fogStats
  } = useMediaService(fogEnabled);

  const filteredContent = mediaContent.filter(content =>
    filterType === 'all' || content.type === filterType
  );

  const currentContent = filteredContent[currentIndex];

  useEffect(() => {
    if (contentId) {
      loadMedia(contentId);
    } else if (initialContent.length > 0) {
      // Use initial content if provided
    }
  }, [contentId, loadMedia]);

  useEffect(() => {
    if (currentContent && onContentLoad) {
      onContentLoad(currentContent);
    }
  }, [currentContent, onContentLoad]);

  useEffect(() => {
    if (error && onError) {
      onError(error);
    }
  }, [error, onError]);

  const handleNext = () => {
    if (currentIndex < filteredContent.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  };

  const handleFullscreen = () => {
    setIsFullscreen(true);
  };

  const handleExitFullscreen = () => {
    setIsFullscreen(false);
  };

  const renderMediaContent = (content: MediaContent, isPreview: boolean = false) => {
    const commonProps = {
      content,
      isPreview,
      fogEnabled,
      onError: (err: string) => onError?.(err)
    };

    switch (content.type) {
      case 'image':
        return <ImageViewer {...commonProps} />;
      case 'video':
        return <VideoPlayer {...commonProps} />;
      case 'text':
        return <TextViewer {...commonProps} />;
      case 'audio':
        return <AudioPlayer {...commonProps} />;
      default:
        return <div className="unsupported-media">Unsupported media type</div>;
    }
  };

  const renderGridView = () => (
    <div className="media-grid">
      {filteredContent.map((content, index) => (
        <div
          key={content.id}
          className={`media-grid-item ${index === currentIndex ? 'active' : ''}`}
          onClick={() => setCurrentIndex(index)}
        >
          {renderMediaContent(content, true)}
          <div className="grid-item-overlay">
            <span className="media-type-badge">{content.type}</span>
            {content.metadata.duration && (
              <span className="duration-badge">
                {Math.floor(content.metadata.duration / 60)}:
                {(content.metadata.duration % 60).toString().padStart(2, '0')}
              </span>
            )}
          </div>
        </div>
      ))}
    </div>
  );

  const renderCarouselView = () => (
    <div className="media-carousel">
      <div className="carousel-track">
        {filteredContent.map((content, index) => (
          <div
            key={content.id}
            className={`carousel-item ${index === currentIndex ? 'active' : ''}`}
            style={{
              transform: `translateX(${(index - currentIndex) * 100}%)`
            }}
          >
            {renderMediaContent(content)}
          </div>
        ))}
      </div>
      <div className="carousel-indicators">
        {filteredContent.map((_, index) => (
          <button
            key={index}
            className={`indicator ${index === currentIndex ? 'active' : ''}`}
            onClick={() => setCurrentIndex(index)}
            aria-label={`Go to slide ${index + 1}`}
          />
        ))}
      </div>
    </div>
  );

  if (isLoading) {
    return (
      <div className="media-display-loading">
        <div className="loading-spinner"></div>
        <p>Loading media content...</p>
        {fogEnabled && (
          <div className="fog-loading-info">
            <p>Optimizing delivery through fog network</p>
            <div className="fog-nodes-status">
              Nodes: {fogStats.activeNodes} | Latency: {fogStats.averageLatency}ms
            </div>
          </div>
        )}
      </div>
    );
  }

  if (error) {
    return (
      <div className="media-display-error">
        <div className="error-icon">⚠️</div>
        <h3>Media Load Error</h3>
        <p>{error}</p>
        <button onClick={() => loadMedia(contentId)} className="retry-btn">
          Retry
        </button>
      </div>
    );
  }

  if (filteredContent.length === 0) {
    return (
      <div className="no-media-content">
        <div className="no-content-icon">📁</div>
        <h3>No Media Content</h3>
        <p>No media files match the current filter.</p>
      </div>
    );
  }

  return (
    <div className="media-display-engine" ref={containerRef}>
      <div className="media-controls-header">
        <MediaControls
          displayMode={displayMode}
          onDisplayModeChange={setDisplayMode}
          filterType={filterType}
          onFilterChange={setFilterType}
          canNavigate={filteredContent.length > 1}
          onPrevious={handlePrevious}
          onNext={handleNext}
          onFullscreen={handleFullscreen}
          onDownload={() => downloadMedia(currentContent.id)}
          currentIndex={currentIndex}
          totalCount={filteredContent.length}
        />

        {fogEnabled && (
          <div className="fog-status">
            <span className="fog-indicator">☁️</span>
            <span className="fog-text">Fog Optimized</span>
            <div className="fog-tooltip">
              <div>Active Nodes: {fogStats.activeNodes}</div>
              <div>Cache Hit Rate: {fogStats.cacheHitRate}%</div>
              <div>Bandwidth Saved: {fogStats.bandwidthSaved}</div>
            </div>
          </div>
        )}
      </div>

      <div className="media-content-area">
        {displayMode === 'single' && currentContent && (
          <div className="single-media-view">
            {renderMediaContent(currentContent)}
          </div>
        )}

        {displayMode === 'grid' && renderGridView()}

        {displayMode === 'carousel' && renderCarouselView()}
      </div>

      {isFullscreen && currentContent && (
        <FullscreenView
          content={currentContent}
          onClose={handleExitFullscreen}
          onNext={filteredContent.length > 1 ? handleNext : undefined}
          onPrevious={filteredContent.length > 1 ? handlePrevious : undefined}
        />
      )}

      <div className="media-info-footer">
        {currentContent && (
          <div className="current-media-info">
            <span className="media-title">{currentContent.id}</span>
            <span className="media-details">
              {currentContent.type} •
              {currentContent.metadata.format} •
              {currentContent.metadata.size && ` ${(currentContent.metadata.size / 1024 / 1024).toFixed(1)}MB`}
            </span>
            {streamingStats.isStreaming && (
              <div className="streaming-info">
                <span className="streaming-indicator">🔴 Live</span>
                <span>Quality: {streamingStats.quality}</span>
                <span>Buffer: {streamingStats.bufferHealth}%</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
