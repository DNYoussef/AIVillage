import React, { useEffect } from 'react';
import { MediaContent } from '../../types';
import { ImageViewer } from './ImageViewer';
import { VideoPlayer } from './VideoPlayer';
import { TextViewer } from './TextViewer';
import { AudioPlayer } from './AudioPlayer';
import './FullscreenView.css';

interface FullscreenViewProps {
  content: MediaContent;
  onClose: () => void;
  onNext?: () => void;
  onPrevious?: () => void;
}

export const FullscreenView: React.FC<FullscreenViewProps> = ({
  content,
  onClose,
  onNext,
  onPrevious,
}) => {
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'Escape':
          onClose();
          break;
        case 'ArrowLeft':
          onPrevious?.();
          break;
        case 'ArrowRight':
          onNext?.();
          break;
      }
    };

    document.addEventListener('keydown', handleKeyPress);
    document.body.style.overflow = 'hidden';

    return () => {
      document.removeEventListener('keydown', handleKeyPress);
      document.body.style.overflow = 'auto';
    };
  }, [onClose, onNext, onPrevious]);

  const renderMediaContent = () => {
    const commonProps = {
      content,
      isPreview: false,
      fogEnabled: true,
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

  return (
    <div className="fullscreen-view">
      <div className="fullscreen-overlay" onClick={onClose} />

      <div className="fullscreen-container">
        <div className="fullscreen-controls">
          {onPrevious && (
            <button
              onClick={onPrevious}
              className="nav-btn previous"
              title="Previous (Left Arrow)"
            >
              ◀️
            </button>
          )}

          <button
            onClick={onClose}
            className="close-btn"
            title="Close (Escape)"
          >
            ✕
          </button>

          {onNext && (
            <button
              onClick={onNext}
              className="nav-btn next"
              title="Next (Right Arrow)"
            >
              ▶️
            </button>
          )}
        </div>

        <div className="fullscreen-content">
          {renderMediaContent()}
        </div>

        <div className="fullscreen-info">
          <div className="content-title">{content.id}</div>
          <div className="content-details">
            <span className="content-type">{content.type.toUpperCase()}</span>
            {content.metadata.format && (
              <span className="content-format"> • {content.metadata.format}</span>
            )}
            {content.metadata.size && (
              <span className="content-size">
                 • {(content.metadata.size / 1024 / 1024).toFixed(1)} MB
              </span>
            )}
            {content.metadata.dimensions && (
              <span className="content-dimensions">
                 • {content.metadata.dimensions.width} × {content.metadata.dimensions.height}
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
