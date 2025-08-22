import React, { useState } from 'react';
import { MediaContent } from '../../types';
import './TextViewer.css';

interface TextViewerProps {
  content: MediaContent;
  isPreview?: boolean;
  fogEnabled?: boolean;
  onError?: (error: string) => void;
}

export const TextViewer: React.FC<TextViewerProps> = ({
  content,
  isPreview = false,
  fogEnabled = true,
  onError,
}) => {
  const [fontSize, setFontSize] = useState(14);
  const [lineHeight, setLineHeight] = useState(1.6);
  const [isExpanded, setIsExpanded] = useState(!isPreview);

  const handleFontSizeChange = (delta: number) => {
    setFontSize(prev => Math.max(8, Math.min(24, prev + delta)));
  };

  const handleLineHeightChange = (delta: number) => {
    setLineHeight(prev => Math.max(1.0, Math.min(2.5, prev + delta)));
  };

  const toggleExpanded = () => {
    setIsExpanded(!isExpanded);
  };

  const textContent = content.content || 'No text content available';
  const displayContent = isPreview && !isExpanded
    ? textContent.length > 200
      ? `${textContent.substring(0, 200)}...`
      : textContent
    : textContent;

  return (
    <div className={`text-viewer ${isPreview ? 'preview' : ''}`}>
      {!isPreview && (
        <div className="text-controls">
          <div className="font-controls">
            <button
              onClick={() => handleFontSizeChange(-1)}
              className="font-btn"
              title="Decrease font size"
            >
              A-
            </button>
            <span className="font-size-display">{fontSize}px</span>
            <button
              onClick={() => handleFontSizeChange(1)}
              className="font-btn"
              title="Increase font size"
            >
              A+
            </button>
          </div>

          <div className="line-height-controls">
            <button
              onClick={() => handleLineHeightChange(-0.1)}
              className="line-height-btn"
              title="Decrease line height"
            >
              ↔️-
            </button>
            <span className="line-height-display">{lineHeight.toFixed(1)}</span>
            <button
              onClick={() => handleLineHeightChange(0.1)}
              className="line-height-btn"
              title="Increase line height"
            >
              ↔️+
            </button>
          </div>

          <button
            onClick={() => navigator.clipboard.writeText(textContent)}
            className="copy-btn"
            title="Copy text"
          >
            📋 Copy
          </button>
        </div>
      )}

      <div
        className="text-content"
        style={{
          fontSize: `${fontSize}px`,
          lineHeight: lineHeight,
          maxHeight: isPreview && !isExpanded ? '150px' : 'none',
          overflow: isPreview && !isExpanded ? 'hidden' : 'auto',
        }}
      >
        <pre className="text-display">{displayContent}</pre>
      </div>

      {isPreview && textContent.length > 200 && (
        <button
          onClick={toggleExpanded}
          className="expand-btn"
        >
          {isExpanded ? 'Show less' : 'Show more'}
        </button>
      )}

      {!isPreview && content.metadata && (
        <div className="text-metadata">
          <span className="metadata-item">
            {content.metadata.format || 'Plain Text'}
          </span>
          {content.metadata.size && (
            <span className="metadata-item">
              {content.metadata.size} bytes
            </span>
          )}
          <span className="metadata-item">
            {textContent.length} characters
          </span>
          {fogEnabled && (
            <span className="metadata-item fog-optimized">
              ☁️ Fog Cached
            </span>
          )}
        </div>
      )}
    </div>
  );
};
