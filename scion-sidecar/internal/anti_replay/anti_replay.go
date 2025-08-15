// Anti-Replay Manager - Sliding window anti-replay protection for SCION packets
package anti_replay

import (
	"context"
	"encoding/binary"
	"fmt"
	"sync"
	"time"

	"github.com/aivillage/scion-sidecar/internal/metrics"
	log "github.com/sirupsen/logrus"

	// Using BadgerDB for persistence (lightweight alternative to RocksDB)
	"github.com/dgraph-io/badger/v3"
)

// Config holds anti-replay configuration
type Config struct {
	WindowSize   int           // Replay window size in bits (typically 1024)
	DBPath       string        // Database path for persistence
	CleanupTTL   time.Duration // TTL for cleaning up old windows
	CleanupInterval time.Duration // How often to run cleanup
	SyncInterval    time.Duration // How often to sync to disk
}

// AntiReplayManager manages sliding window anti-replay protection
type AntiReplayManager struct {
	config  *Config
	ctx     context.Context
	cancel  context.CancelFunc
	metrics *metrics.MetricsCollector

	// Database for persistence
	db *badger.DB

	// In-memory window cache for performance
	windows     map[string]*SequenceWindow // peer_id -> window
	windowMutex sync.RWMutex

	// Cleanup and sync
	cleanupTicker *time.Ticker
	syncTicker    *time.Ticker

	// Statistics
	stats struct {
		sync.RWMutex
		TotalValidated     uint64
		ReplayBlocks       uint64
		FutureRejections   uint64
		ExpiredRejections  uint64
		WindowSlides       uint64
		WindowUpdates      uint64
		PersistenceErrors  uint64
		CleanupOperations  uint64
		AverageValidationTimeUs uint64
	}
}

// SequenceWindow represents a sliding window for sequence number validation
type SequenceWindow struct {
	PeerID           string    // Peer identifier
	WindowBase       uint64    // Base sequence number of window
	WindowSize       uint32    // Window size in bits
	WindowBitmap     []byte    // Bitmap of received sequences
	HighestReceived  uint64    // Highest sequence number received
	LastUpdate       time.Time // Last window update time
	Stats            WindowStats // Window statistics
	dirty            bool      // Whether window needs persistence
	mutex            sync.RWMutex // Per-window lock
}

// WindowStats tracks statistics for a sequence window
type WindowStats struct {
	TotalProcessed   uint64
	ValidAccepted    uint64
	InvalidRejected  uint64
	WindowSlides     uint64
	BitmapUpdates    uint64
}

// ValidationResult represents the result of sequence validation
type ValidationResult struct {
	Valid           bool
	RejectionReason string
	WindowState     *SequenceWindow
	ValidationTimeUs uint64
}

// NewAntiReplayManager creates a new anti-replay manager
func NewAntiReplayManager(config *Config, metricsCollector *metrics.MetricsCollector) (*AntiReplayManager, error) {
	// Set default values
	if config.WindowSize == 0 {
		config.WindowSize = 1024 // 1024-bit window
	}
	if config.CleanupTTL == 0 {
		config.CleanupTTL = 24 * time.Hour
	}
	if config.CleanupInterval == 0 {
		config.CleanupInterval = 1 * time.Hour
	}
	if config.SyncInterval == 0 {
		config.SyncInterval = 5 * time.Minute
	}

	ctx, cancel := context.WithCancel(context.Background())

	manager := &AntiReplayManager{
		config:        config,
		ctx:           ctx,
		cancel:        cancel,
		metrics:       metricsCollector,
		windows:       make(map[string]*SequenceWindow),
		cleanupTicker: time.NewTicker(config.CleanupInterval),
		syncTicker:    time.NewTicker(config.SyncInterval),
	}

	// Initialize database
	if err := manager.initDatabase(); err != nil {
		cancel()
		return nil, fmt.Errorf("failed to initialize database: %w", err)
	}

	// Load existing windows from database
	if err := manager.loadWindows(); err != nil {
		cancel()
		return nil, fmt.Errorf("failed to load windows: %w", err)
	}

	// Start background tasks
	go manager.cleanupWorker()
	go manager.syncWorker()

	log.WithFields(log.Fields{
		"window_size":      config.WindowSize,
		"db_path":         config.DBPath,
		"cleanup_ttl":     config.CleanupTTL,
		"cleanup_interval": config.CleanupInterval,
		"sync_interval":   config.SyncInterval,
	}).Info("Anti-replay manager initialized")

	return manager, nil
}

// initDatabase initializes the BadgerDB database
func (arm *AntiReplayManager) initDatabase() error {
	opts := badger.DefaultOptions(arm.config.DBPath)
	opts.Logger = &badgerLogger{} // Custom logger to reduce verbosity

	db, err := badger.Open(opts)
	if err != nil {
		return fmt.Errorf("failed to open BadgerDB at %s: %w", arm.config.DBPath, err)
	}

	arm.db = db
	log.WithField("db_path", arm.config.DBPath).Info("BadgerDB initialized for anti-replay persistence")
	return nil
}

// loadWindows loads existing sequence windows from database
func (arm *AntiReplayManager) loadWindows() error {
	log.Info("Loading sequence windows from database")

	loadCount := 0
	err := arm.db.View(func(txn *badger.Txn) error {
		opts := badger.DefaultIteratorOptions
		opts.PrefetchSize = 10

		it := txn.NewIterator(opts)
		defer it.Close()

		for it.Rewind(); it.Valid(); it.Next() {
			item := it.Item()
			key := item.Key()

			err := item.Value(func(val []byte) error {
				window, err := arm.deserializeWindow(key, val)
				if err != nil {
					log.WithError(err).WithField("key", string(key)).Error("Failed to deserialize window")
					return nil // Continue loading other windows
				}

				arm.windowMutex.Lock()
				arm.windows[window.PeerID] = window
				arm.windowMutex.Unlock()

				loadCount++
				return nil
			})

			if err != nil {
				log.WithError(err).Error("Error processing window item")
			}
		}

		return nil
	})

	if err != nil {
		return fmt.Errorf("failed to load windows from database: %w", err)
	}

	log.WithField("windows_loaded", loadCount).Info("Sequence windows loaded from database")
	return nil
}

// ValidateSequence validates a sequence number against the anti-replay window
func (arm *AntiReplayManager) ValidateSequence(peerID string, sequenceNumber uint64, timestampNs int64, updateWindow bool) *ValidationResult {
	start := time.Now()
	arm.recordValidationAttempt()

	// Get or create window for peer
	window := arm.getOrCreateWindow(peerID)

	window.mutex.Lock()
	defer window.mutex.Unlock()

	// Validate sequence number
	result := &ValidationResult{
		ValidationTimeUs: 0, // Will be set at the end
	}

	// Check if sequence is too old (before window)
	if sequenceNumber < window.WindowBase {
		result.Valid = false
		result.RejectionReason = "expired"
		arm.recordExpiredRejection()
		window.Stats.InvalidRejected++
		log.WithFields(log.Fields{
			"peer_id":         peerID,
			"sequence":        sequenceNumber,
			"window_base":     window.WindowBase,
			"rejection":       "expired",
		}).Debug("Sequence number rejected: too old")
	} else if sequenceNumber >= window.WindowBase+uint64(window.WindowSize) {
		// Sequence is ahead of window - slide window if not too far ahead
		maxFuture := uint64(window.WindowSize * 2) // Allow up to 2x window size ahead
		if sequenceNumber >= window.WindowBase+maxFuture {
			result.Valid = false
			result.RejectionReason = "future"
			arm.recordFutureRejection()
			window.Stats.InvalidRejected++
			log.WithFields(log.Fields{
				"peer_id":         peerID,
				"sequence":        sequenceNumber,
				"window_base":     window.WindowBase,
				"max_future":      window.WindowBase + maxFuture,
				"rejection":       "future",
			}).Debug("Sequence number rejected: too far in future")
		} else {
			// Valid future sequence - slide window and accept
			if updateWindow {
				arm.slideWindow(window, sequenceNumber)
				arm.setBit(window, sequenceNumber)
				window.Stats.ValidAccepted++
			}
			result.Valid = true
			log.WithFields(log.Fields{
				"peer_id":          peerID,
				"sequence":         sequenceNumber,
				"old_window_base":  window.WindowBase,
				"window_slid":      updateWindow,
			}).Debug("Sequence accepted with window slide")
		}
	} else {
		// Sequence is within current window
		bitIndex := sequenceNumber - window.WindowBase
		if arm.getBit(window, sequenceNumber) {
			// Already received - replay attack
			result.Valid = false
			result.RejectionReason = "replay"
			arm.recordReplayBlock()
			window.Stats.InvalidRejected++
			log.WithFields(log.Fields{
				"peer_id":     peerID,
				"sequence":    sequenceNumber,
				"bit_index":   bitIndex,
				"rejection":   "replay",
			}).Debug("Sequence number rejected: replay detected")
		} else {
			// Valid new sequence within window
			if updateWindow {
				arm.setBit(window, sequenceNumber)
				window.Stats.ValidAccepted++
			}
			result.Valid = true
			log.WithFields(log.Fields{
				"peer_id":    peerID,
				"sequence":   sequenceNumber,
				"bit_index":  bitIndex,
				"updated":    updateWindow,
			}).Debug("Sequence accepted within window")
		}
	}

	// Update window metadata
	if updateWindow && result.Valid {
		if sequenceNumber > window.HighestReceived {
			window.HighestReceived = sequenceNumber
		}
		window.LastUpdate = time.Now()
		window.Stats.TotalProcessed++
		window.dirty = true // Mark for persistence
	} else {
		window.Stats.TotalProcessed++
	}

	// Record timing and metrics
	validationTime := time.Since(start)
	result.ValidationTimeUs = uint64(validationTime.Microseconds())
	result.WindowState = arm.copyWindowState(window)

	arm.recordValidationTime(validationTime)
	arm.recordWindowUpdate()

	return result
}

// getOrCreateWindow gets an existing window or creates a new one for a peer
func (arm *AntiReplayManager) getOrCreateWindow(peerID string) *SequenceWindow {
	arm.windowMutex.RLock()
	if window, exists := arm.windows[peerID]; exists {
		arm.windowMutex.RUnlock()
		return window
	}
	arm.windowMutex.RUnlock()

	// Need to create new window
	arm.windowMutex.Lock()
	defer arm.windowMutex.Unlock()

	// Check again in case another goroutine created it
	if window, exists := arm.windows[peerID]; exists {
		return window
	}

	// Create new window
	bitmapSize := (arm.config.WindowSize + 7) / 8 // Round up to nearest byte
	window := &SequenceWindow{
		PeerID:          peerID,
		WindowBase:      0,
		WindowSize:      uint32(arm.config.WindowSize),
		WindowBitmap:    make([]byte, bitmapSize),
		HighestReceived: 0,
		LastUpdate:      time.Now(),
		Stats:           WindowStats{},
		dirty:           true,
	}

	arm.windows[peerID] = window

	log.WithFields(log.Fields{
		"peer_id":     peerID,
		"window_size": arm.config.WindowSize,
		"bitmap_size": bitmapSize,
	}).Info("Created new sequence window")

	return window
}

// slideWindow slides the sequence window to accommodate a new sequence number
func (arm *AntiReplayManager) slideWindow(window *SequenceWindow, newSequence uint64) {
	if newSequence < window.WindowBase+uint64(window.WindowSize) {
		return // No need to slide
	}

	oldBase := window.WindowBase
	slideAmount := newSequence - (window.WindowBase + uint64(window.WindowSize) - 1)

	// Slide window forward
	window.WindowBase += slideAmount
	window.Stats.WindowSlides++
	arm.recordWindowSlide()

	// Clear bitmap bits that are now outside the window
	if slideAmount >= uint64(window.WindowSize) {
		// Complete slide - clear entire bitmap
		for i := range window.WindowBitmap {
			window.WindowBitmap[i] = 0
		}
	} else {
		// Partial slide - shift bitmap
		arm.shiftBitmap(window.WindowBitmap, int(slideAmount))
	}

	window.dirty = true

	log.WithFields(log.Fields{
		"peer_id":       window.PeerID,
		"old_base":      oldBase,
		"new_base":      window.WindowBase,
		"slide_amount":  slideAmount,
		"new_sequence":  newSequence,
	}).Debug("Slid sequence window")
}

// shiftBitmap shifts bitmap left by specified number of bits
func (arm *AntiReplayManager) shiftBitmap(bitmap []byte, bits int) {
	if bits <= 0 || bits >= len(bitmap)*8 {
		// Clear entire bitmap
		for i := range bitmap {
			bitmap[i] = 0
		}
		return
	}

	byteShift := bits / 8
	bitShift := bits % 8

	// Shift by whole bytes first
	if byteShift > 0 {
		copy(bitmap, bitmap[byteShift:])
		// Clear the tail
		for i := len(bitmap) - byteShift; i < len(bitmap); i++ {
			bitmap[i] = 0
		}
	}

	// Shift remaining bits
	if bitShift > 0 {
		carry := byte(0)
		for i := len(bitmap) - 1; i >= 0; i-- {
			newCarry := bitmap[i] >> (8 - bitShift)
			bitmap[i] = (bitmap[i] << bitShift) | carry
			carry = newCarry
		}
	}
}

// getBit gets the bit for a sequence number in the window
func (arm *AntiReplayManager) getBit(window *SequenceWindow, sequence uint64) bool {
	if sequence < window.WindowBase || sequence >= window.WindowBase+uint64(window.WindowSize) {
		return false
	}

	bitIndex := sequence - window.WindowBase
	byteIndex := bitIndex / 8
	bitPos := bitIndex % 8

	if int(byteIndex) >= len(window.WindowBitmap) {
		return false
	}

	return (window.WindowBitmap[byteIndex] & (1 << bitPos)) != 0
}

// setBit sets the bit for a sequence number in the window
func (arm *AntiReplayManager) setBit(window *SequenceWindow, sequence uint64) {
	if sequence < window.WindowBase || sequence >= window.WindowBase+uint64(window.WindowSize) {
		return
	}

	bitIndex := sequence - window.WindowBase
	byteIndex := bitIndex / 8
	bitPos := bitIndex % 8

	if int(byteIndex) >= len(window.WindowBitmap) {
		return
	}

	window.WindowBitmap[byteIndex] |= (1 << bitPos)
	window.Stats.BitmapUpdates++
	window.dirty = true
}

// copyWindowState creates a copy of window state for external use
func (arm *AntiReplayManager) copyWindowState(window *SequenceWindow) *SequenceWindow {
	bitmapCopy := make([]byte, len(window.WindowBitmap))
	copy(bitmapCopy, window.WindowBitmap)

	return &SequenceWindow{
		PeerID:          window.PeerID,
		WindowBase:      window.WindowBase,
		WindowSize:      window.WindowSize,
		WindowBitmap:    bitmapCopy,
		HighestReceived: window.HighestReceived,
		LastUpdate:      window.LastUpdate,
		Stats:           window.Stats, // Copy by value
	}
}

// cleanupWorker runs periodic cleanup of old windows
func (arm *AntiReplayManager) cleanupWorker() {
	log.Info("Starting anti-replay cleanup worker")
	defer log.Info("Anti-replay cleanup worker stopped")

	for {
		select {
		case <-arm.ctx.Done():
			return
		case <-arm.cleanupTicker.C:
			arm.performCleanup()
		}
	}
}

// performCleanup removes old unused sequence windows
func (arm *AntiReplayManager) performCleanup() {
	arm.recordCleanupOperation()

	cutoff := time.Now().Add(-arm.config.CleanupTTL)
	toDelete := make([]string, 0)

	arm.windowMutex.RLock()
	for peerID, window := range arm.windows {
		window.mutex.RLock()
		if window.LastUpdate.Before(cutoff) {
			toDelete = append(toDelete, peerID)
		}
		window.mutex.RUnlock()
	}
	arm.windowMutex.RUnlock()

	if len(toDelete) == 0 {
		return
	}

	log.WithField("windows_to_delete", len(toDelete)).Info("Cleaning up old sequence windows")

	// Delete from memory and database
	for _, peerID := range toDelete {
		arm.windowMutex.Lock()
		delete(arm.windows, peerID)
		arm.windowMutex.Unlock()

		// Delete from database
		err := arm.db.Update(func(txn *badger.Txn) error {
			return txn.Delete([]byte(peerID))
		})
		if err != nil {
			log.WithError(err).WithField("peer_id", peerID).Error("Failed to delete window from database")
			arm.recordPersistenceError()
		}
	}
}

// syncWorker periodically syncs dirty windows to database
func (arm *AntiReplayManager) syncWorker() {
	log.Info("Starting anti-replay sync worker")
	defer log.Info("Anti-replay sync worker stopped")

	for {
		select {
		case <-arm.ctx.Done():
			return
		case <-arm.syncTicker.C:
			arm.syncDirtyWindows()
		}
	}
}

// syncDirtyWindows syncs dirty windows to database
func (arm *AntiReplayManager) syncDirtyWindows() {
	dirtyWindows := make([]*SequenceWindow, 0)

	arm.windowMutex.RLock()
	for _, window := range arm.windows {
		window.mutex.RLock()
		if window.dirty {
			dirtyWindows = append(dirtyWindows, window)
		}
		window.mutex.RUnlock()
	}
	arm.windowMutex.RUnlock()

	if len(dirtyWindows) == 0 {
		return
	}

	log.WithField("dirty_windows", len(dirtyWindows)).Debug("Syncing dirty windows to database")

	// Batch write to database
	err := arm.db.Update(func(txn *badger.Txn) error {
		for _, window := range dirtyWindows {
			window.mutex.Lock()
			if window.dirty {
				data, err := arm.serializeWindow(window)
				if err != nil {
					window.mutex.Unlock()
					return fmt.Errorf("failed to serialize window for %s: %w", window.PeerID, err)
				}

				if err := txn.Set([]byte(window.PeerID), data); err != nil {
					window.mutex.Unlock()
					return fmt.Errorf("failed to write window for %s: %w", window.PeerID, err)
				}

				window.dirty = false
			}
			window.mutex.Unlock()
		}
		return nil
	})

	if err != nil {
		log.WithError(err).Error("Failed to sync dirty windows")
		arm.recordPersistenceError()
	}
}

// serializeWindow serializes a sequence window for database storage
func (arm *AntiReplayManager) serializeWindow(window *SequenceWindow) ([]byte, error) {
	// Simple binary format:
	// 8 bytes: WindowBase
	// 4 bytes: WindowSize
	// 8 bytes: HighestReceived
	// 8 bytes: LastUpdate (Unix nano)
	// 8 bytes: Stats.TotalProcessed
	// 8 bytes: Stats.ValidAccepted
	// 8 bytes: Stats.InvalidRejected
	// 8 bytes: Stats.WindowSlides
	// 8 bytes: Stats.BitmapUpdates
	// N bytes: WindowBitmap

	headerSize := 8 + 4 + 8 + 8 + 8 + 8 + 8 + 8 + 8 // 72 bytes
	data := make([]byte, headerSize+len(window.WindowBitmap))

	offset := 0
	binary.BigEndian.PutUint64(data[offset:], window.WindowBase)
	offset += 8
	binary.BigEndian.PutUint32(data[offset:], window.WindowSize)
	offset += 4
	binary.BigEndian.PutUint64(data[offset:], window.HighestReceived)
	offset += 8
	binary.BigEndian.PutUint64(data[offset:], uint64(window.LastUpdate.UnixNano()))
	offset += 8
	binary.BigEndian.PutUint64(data[offset:], window.Stats.TotalProcessed)
	offset += 8
	binary.BigEndian.PutUint64(data[offset:], window.Stats.ValidAccepted)
	offset += 8
	binary.BigEndian.PutUint64(data[offset:], window.Stats.InvalidRejected)
	offset += 8
	binary.BigEndian.PutUint64(data[offset:], window.Stats.WindowSlides)
	offset += 8
	binary.BigEndian.PutUint64(data[offset:], window.Stats.BitmapUpdates)
	offset += 8

	copy(data[offset:], window.WindowBitmap)

	return data, nil
}

// deserializeWindow deserializes a sequence window from database storage
func (arm *AntiReplayManager) deserializeWindow(key, data []byte) (*SequenceWindow, error) {
	if len(data) < 72 { // Minimum header size
		return nil, fmt.Errorf("data too short for window header")
	}

	peerID := string(key)
	offset := 0

	windowBase := binary.BigEndian.Uint64(data[offset:])
	offset += 8
	windowSize := binary.BigEndian.Uint32(data[offset:])
	offset += 4
	highestReceived := binary.BigEndian.Uint64(data[offset:])
	offset += 8
	lastUpdateNano := binary.BigEndian.Uint64(data[offset:])
	offset += 8
	totalProcessed := binary.BigEndian.Uint64(data[offset:])
	offset += 8
	validAccepted := binary.BigEndian.Uint64(data[offset:])
	offset += 8
	invalidRejected := binary.BigEndian.Uint64(data[offset:])
	offset += 8
	windowSlides := binary.BigEndian.Uint64(data[offset:])
	offset += 8
	bitmapUpdates := binary.BigEndian.Uint64(data[offset:])
	offset += 8

	bitmapData := make([]byte, len(data)-offset)
	copy(bitmapData, data[offset:])

	window := &SequenceWindow{
		PeerID:          peerID,
		WindowBase:      windowBase,
		WindowSize:      windowSize,
		WindowBitmap:    bitmapData,
		HighestReceived: highestReceived,
		LastUpdate:      time.Unix(0, int64(lastUpdateNano)),
		Stats: WindowStats{
			TotalProcessed:  totalProcessed,
			ValidAccepted:   validAccepted,
			InvalidRejected: invalidRejected,
			WindowSlides:    windowSlides,
			BitmapUpdates:   bitmapUpdates,
		},
		dirty: false,
	}

	return window, nil
}

// GetStats returns anti-replay manager statistics
func (arm *AntiReplayManager) GetStats() AntiReplayStats {
	arm.stats.RLock()
	defer arm.stats.RUnlock()

	arm.windowMutex.RLock()
	windowCount := uint64(len(arm.windows))
	arm.windowMutex.RUnlock()

	falsePositiveRate := 0.0
	if arm.stats.TotalValidated > 0 {
		// False positives are legitimate packets incorrectly rejected
		// This is hard to measure accurately without ground truth
		falsePositiveRate = float64(arm.stats.ExpiredRejections) / float64(arm.stats.TotalValidated) * 0.01 // Estimate
	}

	return AntiReplayStats{
		TotalValidated:         arm.stats.TotalValidated,
		ReplaysBlocked:         arm.stats.ReplayBlocks,
		FutureRejected:         arm.stats.FutureRejections,
		ExpiredRejected:        arm.stats.ExpiredRejections,
		FalsePositiveRate:      falsePositiveRate,
		WindowUpdates:          arm.stats.WindowUpdates,
		AverageValidationTimeUs: arm.stats.AverageValidationTimeUs,
		PersistenceErrors:      arm.stats.PersistenceErrors,
		ActiveWindows:          windowCount,
	}
}

// Stop gracefully stops the anti-replay manager
func (arm *AntiReplayManager) Stop() error {
	log.Info("Stopping anti-replay manager")

	arm.cancel()
	arm.cleanupTicker.Stop()
	arm.syncTicker.Stop()

	// Final sync of dirty windows
	arm.syncDirtyWindows()

	// Close database
	if arm.db != nil {
		if err := arm.db.Close(); err != nil {
			log.WithError(err).Error("Failed to close anti-replay database")
		}
	}

	log.Info("Anti-replay manager stopped")
	return nil
}

// Statistics recording methods

func (arm *AntiReplayManager) recordValidationAttempt() {
	arm.stats.Lock()
	arm.stats.TotalValidated++
	arm.stats.Unlock()
}

func (arm *AntiReplayManager) recordReplayBlock() {
	arm.stats.Lock()
	arm.stats.ReplayBlocks++
	arm.stats.Unlock()
}

func (arm *AntiReplayManager) recordFutureRejection() {
	arm.stats.Lock()
	arm.stats.FutureRejections++
	arm.stats.Unlock()
}

func (arm *AntiReplayManager) recordExpiredRejection() {
	arm.stats.Lock()
	arm.stats.ExpiredRejections++
	arm.stats.Unlock()
}

func (arm *AntiReplayManager) recordWindowSlide() {
	arm.stats.Lock()
	arm.stats.WindowSlides++
	arm.stats.Unlock()
}

func (arm *AntiReplayManager) recordWindowUpdate() {
	arm.stats.Lock()
	arm.stats.WindowUpdates++
	arm.stats.Unlock()
}

func (arm *AntiReplayManager) recordPersistenceError() {
	arm.stats.Lock()
	arm.stats.PersistenceErrors++
	arm.stats.Unlock()
}

func (arm *AntiReplayManager) recordCleanupOperation() {
	arm.stats.Lock()
	arm.stats.CleanupOperations++
	arm.stats.Unlock()
}

func (arm *AntiReplayManager) recordValidationTime(duration time.Duration) {
	arm.stats.Lock()
	// Update exponentially weighted moving average
	newTime := uint64(duration.Microseconds())
	if arm.stats.AverageValidationTimeUs == 0 {
		arm.stats.AverageValidationTimeUs = newTime
	} else {
		alpha := 0.1 // Smoothing factor
		arm.stats.AverageValidationTimeUs = uint64(float64(arm.stats.AverageValidationTimeUs)*(1-alpha) + float64(newTime)*alpha)
	}
	arm.stats.Unlock()
}

// AntiReplayStats represents anti-replay statistics
type AntiReplayStats struct {
	TotalValidated          uint64
	ReplaysBlocked          uint64
	FutureRejected          uint64
	ExpiredRejected         uint64
	FalsePositiveRate       float64
	WindowUpdates           uint64
	AverageValidationTimeUs uint64
	PersistenceErrors       uint64
	ActiveWindows           uint64
}

// Custom BadgerDB logger to reduce verbosity
type badgerLogger struct{}

func (bl *badgerLogger) Errorf(format string, args ...interface{})   { log.Errorf("BadgerDB: "+format, args...) }
func (bl *badgerLogger) Warningf(format string, args ...interface{}) { log.Warnf("BadgerDB: "+format, args...) }
func (bl *badgerLogger) Infof(format string, args ...interface{})    { log.Debugf("BadgerDB: "+format, args...) } // Reduce to debug
func (bl *badgerLogger) Debugf(format string, args ...interface{})   { /* Ignore debug messages */ }
