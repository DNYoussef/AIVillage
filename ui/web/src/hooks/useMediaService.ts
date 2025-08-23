// Media Service Hook - Handle multimedia content in P2P messaging
import { useState, useCallback, useRef } from 'react';
import { MediaContent } from '../types';

interface MediaServiceState {
  mediaItems: MediaContent[];
  isProcessing: boolean;
  uploadProgress: number;
  compressionSettings: {
    quality: number;
    maxSize: number;
    format: 'original' | 'compressed' | 'adaptive';
  };
}

export interface MediaServiceHook {
  mediaState: MediaServiceState;
  processMedia: (file: File) => Promise<MediaContent | null>;
  generateThumbnail: (media: MediaContent) => Promise<string>;
  compressMedia: (media: MediaContent, quality?: number) => Promise<MediaContent>;
  shareMedia: (mediaId: string, peerId: string) => Promise<boolean>;
  deleteMedia: (mediaId: string) => void;
  updateCompressionSettings: (settings: Partial<MediaServiceState['compressionSettings']>) => void;
}

export const useMediaService = (): MediaServiceHook => {
  const [mediaState, setMediaState] = useState<MediaServiceState>({
    mediaItems: [],
    isProcessing: false,
    uploadProgress: 0,
    compressionSettings: {
      quality: 0.8,
      maxSize: 5 * 1024 * 1024, // 5MB
      format: 'adaptive'
    }
  });

  const compressionWorker = useRef<Worker | null>(null);
  const mediaCache = useRef<Map<string, Blob>>(new Map());

  // Initialize compression worker for efficient media processing
  const initializeWorker = useCallback(() => {
    if (!compressionWorker.current) {
      // Create web worker for media compression
      const workerScript = `
        self.onmessage = function(e) {
          const { imageData, quality, format } = e.data;
          // Simulate compression processing
          setTimeout(() => {
            self.postMessage({
              success: true,
              compressedData: imageData,
              compressionRatio: 0.6
            });
          }, 1000);
        };
      `;

      const blob = new Blob([workerScript], { type: 'application/javascript' });
      compressionWorker.current = new Worker(URL.createObjectURL(blob));
    }
  }, []);

  const processMedia = useCallback(async (file: File): Promise<MediaContent | null> => {
    if (!file) return null;

    setMediaState(prev => ({ ...prev, isProcessing: true, uploadProgress: 0 }));

    try {
      // Validate file type and size
      const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'video/mp4', 'video/webm', 'audio/mp3', 'audio/wav'];
      if (!allowedTypes.includes(file.type)) {
        throw new Error('Unsupported file type');
      }

      if (file.size > mediaState.compressionSettings.maxSize) {
        console.warn('File exceeds maximum size, will be compressed');
      }

      // Generate unique media ID
      const mediaId = `media-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

      // Create media content object
      const mediaContent: MediaContent = {
        id: mediaId,
        type: file.type.startsWith('image/') ? 'image' :
              file.type.startsWith('video/') ? 'video' :
              file.type.startsWith('audio/') ? 'audio' : 'text',
        metadata: {
          size: file.size,
          format: file.type,
          dimensions: await getMediaDimensions(file)
        }
      };

      // Process based on media type
      if (mediaContent.type === 'image') {
        const processedImage = await processImage(file, mediaContent);
        mediaContent.url = processedImage.url;
        mediaContent.thumbnail = await generateThumbnail(processedImage);
      } else if (mediaContent.type === 'video') {
        const processedVideo = await processVideo(file, mediaContent);
        mediaContent.url = processedVideo.url;
        mediaContent.thumbnail = await generateVideoThumbnail(file);
      } else if (mediaContent.type === 'audio') {
        const processedAudio = await processAudio(file, mediaContent);
        mediaContent.url = processedAudio.url;
      }

      // Store in cache for efficient access
      const fileBlob = new Blob([file], { type: file.type });
      mediaCache.current.set(mediaId, fileBlob);

      // Update state
      setMediaState(prev => ({
        ...prev,
        mediaItems: [...prev.mediaItems, mediaContent],
        isProcessing: false,
        uploadProgress: 100
      }));

      return mediaContent;
    } catch (error) {
      console.error('Media processing failed:', error);
      setMediaState(prev => ({ ...prev, isProcessing: false, uploadProgress: 0 }));
      return null;
    }
  }, [mediaState.compressionSettings]);

  const generateThumbnail = useCallback(async (media: MediaContent): Promise<string> => {
    if (media.type === 'image' && media.url) {
      return generateImageThumbnail(media.url);
    } else if (media.type === 'video' && media.url) {
      return generateVideoThumbnail(media.url);
    }
    return '';
  }, []);

  const compressMedia = useCallback(async (media: MediaContent, quality = 0.8): Promise<MediaContent> => {
    initializeWorker();

    return new Promise((resolve) => {
      if (!compressionWorker.current) {
        resolve(media);
        return;
      }

      compressionWorker.current.onmessage = (e) => {
        const { success, compressedData, compressionRatio } = e.data;
        if (success) {
          const compressedMedia: MediaContent = {
            ...media,
            metadata: {
              ...media.metadata,
              size: Math.floor(media.metadata.size! * compressionRatio)
            }
          };
          resolve(compressedMedia);
        } else {
          resolve(media);
        }
      };

      compressionWorker.current.postMessage({
        imageData: media.url,
        quality,
        format: mediaState.compressionSettings.format
      });
    });
  }, [mediaState.compressionSettings.format]);

  const shareMedia = useCallback(async (mediaId: string, peerId: string): Promise<boolean> => {
    try {
      const media = mediaState.mediaItems.find(m => m.id === mediaId);
      if (!media) return false;

      // In production, this would send media over P2P connection
      console.log(`Sharing media ${mediaId} with peer ${peerId}`);

      // Simulate network transfer
      await new Promise(resolve => setTimeout(resolve, 1000));

      return true;
    } catch (error) {
      console.error('Failed to share media:', error);
      return false;
    }
  }, [mediaState.mediaItems]);

  const deleteMedia = useCallback((mediaId: string): void => {
    setMediaState(prev => ({
      ...prev,
      mediaItems: prev.mediaItems.filter(m => m.id !== mediaId)
    }));

    // Clean up cache
    mediaCache.current.delete(mediaId);

    // Revoke object URL to free memory
    const media = mediaState.mediaItems.find(m => m.id === mediaId);
    if (media?.url) {
      URL.revokeObjectURL(media.url);
    }
  }, [mediaState.mediaItems]);

  const updateCompressionSettings = useCallback((settings: Partial<MediaServiceState['compressionSettings']>): void => {
    setMediaState(prev => ({
      ...prev,
      compressionSettings: {
        ...prev.compressionSettings,
        ...settings
      }
    }));
  }, []);

  // Helper functions
  const getMediaDimensions = async (file: File): Promise<{ width: number; height: number } | undefined> => {
    if (file.type.startsWith('image/')) {
      return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
          resolve({ width: img.width, height: img.height });
        };
        img.onerror = () => resolve(undefined);
        img.src = URL.createObjectURL(file);
      });
    }
    return undefined;
  };

  const processImage = async (file: File, media: MediaContent): Promise<MediaContent> => {
    const url = URL.createObjectURL(file);
    return { ...media, url };
  };

  const processVideo = async (file: File, media: MediaContent): Promise<MediaContent> => {
    const url = URL.createObjectURL(file);
    return { ...media, url };
  };

  const processAudio = async (file: File, media: MediaContent): Promise<MediaContent> => {
    const url = URL.createObjectURL(file);
    return { ...media, url };
  };

  const generateImageThumbnail = async (imageUrl: string): Promise<string> => {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d')!;
      const img = new Image();

      img.onload = () => {
        canvas.width = 150;
        canvas.height = 150;
        ctx.drawImage(img, 0, 0, 150, 150);
        resolve(canvas.toDataURL('image/jpeg', 0.7));
      };

      img.src = imageUrl;
    });
  };

  const generateVideoThumbnail = async (videoUrl: string | File): Promise<string> => {
    return new Promise((resolve) => {
      const video = document.createElement('video');
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d')!;

      video.onloadedmetadata = () => {
        video.currentTime = 1; // Get frame at 1 second
      };

      video.onseeked = () => {
        canvas.width = 150;
        canvas.height = 150;
        ctx.drawImage(video, 0, 0, 150, 150);
        resolve(canvas.toDataURL('image/jpeg', 0.7));
      };

      if (typeof videoUrl === 'string') {
        video.src = videoUrl;
      } else {
        video.src = URL.createObjectURL(videoUrl);
      }
    });
  };

  return {
    mediaState,
    processMedia,
    generateThumbnail,
    compressMedia,
    shareMedia,
    deleteMedia,
    updateCompressionSettings
  };
};
