// Loading Spinner Component - Versatile loading indicator with multiple styles
import React from 'react';
import './LoadingSpinner.css';

interface LoadingSpinnerProps {
  size?: 'small' | 'medium' | 'large';
  variant?: 'spinner' | 'dots' | 'pulse' | 'bars';
  color?: 'primary' | 'secondary' | 'white' | 'custom';
  message?: string;
  className?: string;
  customColor?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'medium',
  variant = 'spinner',
  color = 'primary',
  message,
  className = '',
  customColor
}) => {
  const spinnerClasses = [
    'loading-spinner',
    `loading-spinner--${size}`,
    `loading-spinner--${variant}`,
    `loading-spinner--${color}`,
    className
  ].filter(Boolean).join(' ');

  const customStyles = customColor ? { '--custom-color': customColor } as React.CSSProperties : {};

  const renderSpinner = () => {
    switch (variant) {
      case 'dots':
        return (
          <div className="spinner-dots">
            <div className="spinner-dot"></div>
            <div className="spinner-dot"></div>
            <div className="spinner-dot"></div>
          </div>
        );

      case 'pulse':
        return <div className="spinner-pulse"></div>;

      case 'bars':
        return (
          <div className="spinner-bars">
            <div className="spinner-bar"></div>
            <div className="spinner-bar"></div>
            <div className="spinner-bar"></div>
            <div className="spinner-bar"></div>
          </div>
        );

      case 'spinner':
      default:
        return <div className="spinner-circle"></div>;
    }
  };

  return (
    <div className={spinnerClasses} style={customStyles} role="status" aria-live="polite">
      <div className="loading-spinner-content">
        {renderSpinner()}
        {message && (
          <div className="loading-message" aria-label={message}>
            {message}
          </div>
        )}
      </div>
      <span className="sr-only">Loading...</span>
    </div>
  );
};

// Convenience components for common use cases
export const SmallSpinner: React.FC<Omit<LoadingSpinnerProps, 'size'>> = (props) => (
  <LoadingSpinner {...props} size="small" />
);

export const LargeSpinner: React.FC<Omit<LoadingSpinnerProps, 'size'>> = (props) => (
  <LoadingSpinner {...props} size="large" />
);

export const ButtonSpinner: React.FC<Omit<LoadingSpinnerProps, 'size' | 'variant'>> = (props) => (
  <LoadingSpinner {...props} size="small" variant="spinner" />
);

export default LoadingSpinner;
