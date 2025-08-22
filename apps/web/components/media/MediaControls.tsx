import React from 'react';
import './MediaControls.css';

interface MediaControlsProps {
  displayMode: 'single' | 'grid' | 'carousel';
  onDisplayModeChange: (mode: 'single' | 'grid' | 'carousel') => void;
  filterType: 'all' | 'image' | 'video' | 'text' | 'audio';
  onFilterChange: (type: 'all' | 'image' | 'video' | 'text' | 'audio') => void;
  canNavigate: boolean;
  onPrevious: () => void;
  onNext: () => void;
  onFullscreen: () => void;
  onDownload: () => void;
  currentIndex: number;
  totalCount: number;
}

export const MediaControls: React.FC<MediaControlsProps> = ({
  displayMode,
  onDisplayModeChange,
  filterType,
  onFilterChange,
  canNavigate,
  onPrevious,
  onNext,
  onFullscreen,
  onDownload,
  currentIndex,
  totalCount,
}) => {
  return (
    <div className="media-controls">
      <div className="display-mode-controls">
        <button
          onClick={() => onDisplayModeChange('single')}
          className={`mode-btn ${displayMode === 'single' ? 'active' : ''}`}
          title="Single view"
        >
          🔲 Single
        </button>
        <button
          onClick={() => onDisplayModeChange('grid')}
          className={`mode-btn ${displayMode === 'grid' ? 'active' : ''}`}
          title="Grid view"
        >
          ⬜ Grid
        </button>
        <button
          onClick={() => onDisplayModeChange('carousel')}
          className={`mode-btn ${displayMode === 'carousel' ? 'active' : ''}`}
          title="Carousel view"
        >
          🎠 Carousel
        </button>
      </div>

      <div className="filter-controls">
        <select
          value={filterType}
          onChange={(e) => onFilterChange(e.target.value as any)}
          className="filter-select"
        >
          <option value="all">All Media</option>
          <option value="image">Images</option>
          <option value="video">Videos</option>
          <option value="audio">Audio</option>
          <option value="text">Text</option>
        </select>
      </div>

      {displayMode === 'single' && canNavigate && (
        <div className="navigation-controls">
          <button
            onClick={onPrevious}
            disabled={currentIndex === 0}
            className="nav-btn"
            title="Previous"
          >
            ◀️
          </button>

          <span className="position-indicator">
            {currentIndex + 1} / {totalCount}
          </span>

          <button
            onClick={onNext}
            disabled={currentIndex >= totalCount - 1}
            className="nav-btn"
            title="Next"
          >
            ▶️
          </button>
        </div>
      )}

      <div className="action-controls">
        <button
          onClick={onFullscreen}
          className="action-btn"
          title="Fullscreen"
        >
          🔲 Fullscreen
        </button>

        <button
          onClick={onDownload}
          className="action-btn"
          title="Download"
        >
          ⬇️ Download
        </button>
      </div>
    </div>
  );
};
