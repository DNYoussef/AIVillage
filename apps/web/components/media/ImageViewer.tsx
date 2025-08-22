import React, { useState } from 'react';
import { MediaContent } from '../../types';
import './ImageViewer.css';

interface ImageViewerProps {
  content: MediaContent;
  isPreview?: boolean;
  fogEnabled?: boolean;
  onError?: (error: string) => void;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({
  content,
  isPreview = false,
  fogEnabled = true,
  onError,
}) => {
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const [isZoomed, setIsZoomed] = useState(false);

  const handleImageLoad = () => {
    setIsLoading(false);
    setHasError(false);
  };

  const handleImageError = () => {
    setIsLoading(false);
    setHasError(true);
    onError?.('Failed to load image');
  };

  const toggleZoom = () => {
    if (!isPreview) {
      setIsZoomed(!isZoomed);
    }
  };

  const imageUrl = isPreview && content.thumbnail ? content.thumbnail : content.url;

  return (
    <div className={`image-viewer ${isPreview ? 'preview' : ''} ${isZoomed ? 'zoomed' : ''}`}>
      {isLoading && (
        <div className="image-loading">
          <div className="loading-spinner"></div>
          <p>Loading image...</p>
        </div>
      )}

      {hasError && (
        <div className="image-error">
          <div className="error-icon">🖼️</div>
          <p>Failed to load image</p>
          <button onClick={() => window.location.reload()} className="retry-btn">
            Retry
          </button>
        </div>
      )}

      {imageUrl && !hasError && (
        <>
          <img
            src={imageUrl}
            alt={`Media content ${content.id}`}
            onLoad={handleImageLoad}
            onError={handleImageError}
            onClick={toggleZoom}
            className={`image-content ${isZoomed ? 'zoomed' : ''} ${!isPreview && !isZoomed ? 'zoomable' : ''}`}
            style={{
              display: isLoading ? 'none' : 'block',
              maxWidth: isPreview ? '100%' : isZoomed ? 'none' : '100%',
              maxHeight: isPreview ? '100%' : isZoomed ? 'none' : '80vh',
              cursor: !isPreview ? 'zoom-in' : 'default',
            }}
          />

          {!isPreview && (
            <div className="image-controls">
              <button
                onClick={toggleZoom}
                className="zoom-btn"
                title={isZoomed ? 'Zoom out' : 'Zoom in'}
              >
                {isZoomed ? '🔍' : '🔍'}
              </button>

              {imageUrl && (
                <a
                  href={imageUrl}
                  download
                  className="download-btn"
                  title="Download image"
                >
                  ⬇️
                </a>
              )}
            </div>
          )}

          {!isPreview && content.metadata && (
            <div className="image-metadata">
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
                  ☁️ Fog Optimized
                </span>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
};
