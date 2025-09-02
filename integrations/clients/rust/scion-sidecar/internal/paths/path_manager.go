// Path Manager - SCION path discovery, caching, and selection
package paths

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/aivillage/scion-sidecar/internal/metrics"
	"github.com/scionproto/scion/pkg/addr"
	"github.com/scionproto/scion/pkg/daemon"
	"github.com/scionproto/scion/pkg/snet"
	log "github.com/sirupsen/logrus"
)

// Config holds path manager configuration
type Config struct {
	LocalIA         string        // Local ISD-AS
	SCIONDAddr      string        // SCION daemon address
	RefreshInterval time.Duration // Path refresh interval
	CacheSize       int           // Maximum cached paths
	MaxPathAge      time.Duration // Maximum path age before refresh
	QueryTimeout    time.Duration // Path query timeout
}

// PathManager handles SCION path discovery and management
type PathManager struct {
	config  *Config
	ctx     context.Context
	cancel  context.CancelFunc
	metrics *metrics.MetricsCollector

	// SCION components
	localIA addr.IA
	daemon  daemon.Connector

	// Path cache and management
	pathCache  map[string]*CachedPathSet // dst_ia -> paths
	pathMutex  sync.RWMutex
	refreshTicker *time.Ticker

	// Path quality tracking
	qualityTracker map[string]*PathQuality // path_id -> quality
	qualityMutex   sync.RWMutex

	// Statistics
	stats struct {
		sync.RWMutex
		PathsDiscovered  uint64
		PathsUsed        uint64
		PathQueries      uint64
		CacheHits        uint64
		CacheMisses      uint64
		RefreshOperations uint64
		Errors           uint64
	}
}

// CachedPathSet represents a set of paths to a destination
type CachedPathSet struct {
	DstIA       addr.IA
	Paths       []*PathInfo
	LastRefresh time.Time
	QueryCount  uint64
	UsageCount  uint64
}

// PathInfo contains information about a SCION path
type PathInfo struct {
	ID           string               // Unique path identifier
	Path         snet.Path           // SCION path object
	SrcIA        addr.IA             // Source IA
	DstIA        addr.IA             // Destination IA
	Hops         []HopInfo           // Path hop information
	Quality      *PathQuality        // Quality metrics
	LastUsed     time.Time           // Last usage timestamp
	UsageCount   uint64              // Total usage count
	Expiry       time.Time           // Path expiration time
	SelectionScore float64           // Score for path selection
}

// HopInfo represents information about a path hop
type HopInfo struct {
	IA          addr.IA
	Interface   uint16
	HopType     string
	MTU         uint16
	Latency     time.Duration
}

// PathQuality tracks quality metrics for a path
type PathQuality struct {
	PathID           string
	RTT_EWMA         time.Duration  // Exponentially weighted moving average RTT
	Jitter           time.Duration  // RTT jitter (p95-p50)
	LossRate         float64        // Packet loss rate (0.0-1.0)
	Bandwidth        uint64         // Estimated bandwidth (bytes/sec)
	Stability        float64        // Path stability score (0.0-1.0)
	LastMeasurement  time.Time      // Last measurement timestamp
	MeasurementCount uint64         // Number of measurements
	FailureCount     uint64         // Number of failures
}

// NewPathManager creates a new path manager
func NewPathManager(ctx context.Context, config *Config, metricsCollector *metrics.MetricsCollector) (*PathManager, error) {
	// Set default values
	if config.QueryTimeout == 0 {
		config.QueryTimeout = 10 * time.Second
	}
	if config.RefreshInterval == 0 {
		config.RefreshInterval = 30 * time.Second
	}
	if config.CacheSize == 0 {
		config.CacheSize = 1000
	}
	if config.MaxPathAge == 0 {
		config.MaxPathAge = 5 * time.Minute
	}

	// Parse local IA
	localIA, err := addr.ParseIA(config.LocalIA)
	if err != nil {
		return nil, fmt.Errorf("failed to parse local IA %s: %w", config.LocalIA, err)
	}

	ctx, cancel := context.WithCancel(ctx)

	manager := &PathManager{
		config:         config,
		ctx:            ctx,
		cancel:         cancel,
		metrics:        metricsCollector,
		localIA:        localIA,
		pathCache:      make(map[string]*CachedPathSet),
		qualityTracker: make(map[string]*PathQuality),
		refreshTicker:  time.NewTicker(config.RefreshInterval),
	}

	// Connect to SCION daemon
	if err := manager.connectDaemon(); err != nil {
		cancel()
		return nil, fmt.Errorf("failed to connect to SCION daemon: %w", err)
	}

	// Start background refresh
	go manager.backgroundRefresh()

	log.WithFields(log.Fields{
		"local_ia":         config.LocalIA,
		"refresh_interval": config.RefreshInterval,
		"cache_size":       config.CacheSize,
		"max_path_age":     config.MaxPathAge,
	}).Info("Path manager initialized")

	return manager, nil
}

// connectDaemon connects to the SCION daemon
func (pm *PathManager) connectDaemon() error {
	daemonConn, err := daemon.NewService(pm.config.SCIONDAddr).Connect(pm.ctx)
	if err != nil {
		return fmt.Errorf("failed to connect to daemon at %s: %w", pm.config.SCIONDAddr, err)
	}

	pm.daemon = daemonConn
	log.WithField("sciond_addr", pm.config.SCIONDAddr).Info("Connected to SCION daemon")
	return nil
}

// QueryPaths queries paths to the specified destination
func (pm *PathManager) QueryPaths(dstIA string) ([]*PathInfo, error) {
	start := time.Now()
	pm.recordPathQuery()

	// Parse destination IA
	dstIAAddr, err := addr.ParseIA(dstIA)
	if err != nil {
		pm.recordError()
		return nil, fmt.Errorf("failed to parse destination IA %s: %w", dstIA, err)
	}

	// Check cache first
	pm.pathMutex.RLock()
	if cachedSet, exists := pm.pathCache[dstIA]; exists && !pm.isStale(cachedSet) {
		pm.pathMutex.RUnlock()
		pm.recordCacheHit()

		// Update usage statistics
		pm.pathMutex.Lock()
		cachedSet.QueryCount++
		pm.pathMutex.Unlock()

		log.WithFields(log.Fields{
			"dst_ia":    dstIA,
			"paths":     len(cachedSet.Paths),
			"duration":  time.Since(start),
			"source":    "cache",
		}).Debug("Returned cached paths")

		return cachedSet.Paths, nil
	}
	pm.pathMutex.RUnlock()
	pm.recordCacheMiss()

	// Query fresh paths from daemon
	paths, err := pm.queryFreshPaths(dstIAAddr)
	if err != nil {
		pm.recordError()
		return nil, fmt.Errorf("failed to query fresh paths: %w", err)
	}

	// Convert and cache paths
	pathInfos := pm.convertPaths(paths, dstIAAddr)
	pm.cachePaths(dstIA, pathInfos)

	// Record metrics
	pm.recordPathsDiscovered(len(pathInfos))
	pm.metrics.RecordPathQuery(time.Since(start), len(pathInfos), dstIA)

	log.WithFields(log.Fields{
		"dst_ia":    dstIA,
		"paths":     len(pathInfos),
		"duration":  time.Since(start),
		"source":    "fresh",
	}).Debug("Returned fresh paths")

	return pathInfos, nil
}

// queryFreshPaths queries fresh paths from the SCION daemon
func (pm *PathManager) queryFreshPaths(dstIA addr.IA) ([]snet.Path, error) {
	ctx, cancel := context.WithTimeout(pm.ctx, pm.config.QueryTimeout)
	defer cancel()

	log.WithField("dst_ia", dstIA).Debug("Querying fresh paths from SCION daemon")

	// Query paths from daemon
	pathReply, err := pm.daemon.Paths(ctx, dstIA, pm.localIA, daemon.PathReqFlags{})
	if err != nil {
		return nil, fmt.Errorf("daemon path query failed: %w", err)
	}

	if len(pathReply) == 0 {
		log.WithField("dst_ia", dstIA).Warn("No paths found to destination")
		return nil, fmt.Errorf("no paths available to %s", dstIA)
	}

	log.WithFields(log.Fields{
		"dst_ia": dstIA,
		"paths":  len(pathReply),
	}).Debug("Received paths from SCION daemon")

	return pathReply, nil
}

// convertPaths converts SCION paths to PathInfo objects
func (pm *PathManager) convertPaths(paths []snet.Path, dstIA addr.IA) []*PathInfo {
	pathInfos := make([]*PathInfo, 0, len(paths))

	for i, path := range paths {
		pathInfo := &PathInfo{
			ID:             pm.generatePathID(path, i),
			Path:           path,
			SrcIA:          pm.localIA,
			DstIA:          dstIA,
			Hops:           pm.extractHops(path),
			LastUsed:       time.Time{},
			UsageCount:     0,
			Expiry:         time.Now().Add(pm.config.MaxPathAge),
			SelectionScore: 0.5, // Default neutral score
		}

		// Initialize or get quality tracking
		pm.qualityMutex.Lock()
		if quality, exists := pm.qualityTracker[pathInfo.ID]; exists {
			pathInfo.Quality = quality
		} else {
			pathInfo.Quality = &PathQuality{
				PathID:           pathInfo.ID,
				RTT_EWMA:         0,
				Jitter:           0,
				LossRate:         0.0,
				Bandwidth:        0,
				Stability:        1.0, // Start with high stability
				LastMeasurement:  time.Time{},
				MeasurementCount: 0,
				FailureCount:     0,
			}
			pm.qualityTracker[pathInfo.ID] = pathInfo.Quality
		}
		pm.qualityMutex.Unlock()

		// Calculate initial selection score
		pathInfo.SelectionScore = pm.calculateSelectionScore(pathInfo)

		pathInfos = append(pathInfos, pathInfo)
	}

	// Sort paths by selection score (highest first)
	sort.Slice(pathInfos, func(i, j int) bool {
		return pathInfos[i].SelectionScore > pathInfos[j].SelectionScore
	})

	return pathInfos
}

// generatePathID generates a unique identifier for a path
func (pm *PathManager) generatePathID(path snet.Path, index int) string {
	// Use path fingerprint if available, otherwise use index-based ID
	if pathMeta := path.Metadata(); pathMeta != nil {
		// Use path interfaces for unique identification since String() is not available in v0.10.0
		if interfaces := pathMeta.Interfaces; len(interfaces) > 0 {
			return fmt.Sprintf("path_%v_%d", interfaces, index)
		}
		return fmt.Sprintf("path_meta_%d", index)
	}
	return fmt.Sprintf("path_%s_to_%s_%d", pm.localIA, path.Destination(), index)
}

// extractHops extracts hop information from a SCION path
func (pm *PathManager) extractHops(path snet.Path) []HopInfo {
	var hops []HopInfo

	// Extract hop information from path metadata
	if pathMeta := path.Metadata(); pathMeta != nil {
		// TODO: Extract detailed hop information from path metadata
		// For now, return basic information
		hops = append(hops, HopInfo{
			IA:        path.Destination(),
			Interface: 0, // Would need to extract from path
			HopType:   "unknown",
			MTU:       1500, // Default MTU
			Latency:   0,
		})
	}

	return hops
}

// calculateSelectionScore calculates a selection score for path ranking
func (pm *PathManager) calculateSelectionScore(pathInfo *PathInfo) float64 {
	score := 0.5 // Base score

	// Factor in path quality if available
	if pathInfo.Quality != nil {
		quality := pathInfo.Quality

		// RTT factor (lower is better)
		if quality.RTT_EWMA > 0 {
			rttScore := 1.0 - (float64(quality.RTT_EWMA.Milliseconds()) / 1000.0)
			score += 0.3 * max(0.0, min(1.0, rttScore))
		}

		// Loss rate factor (lower is better)
		lossScore := 1.0 - quality.LossRate
		score += 0.3 * max(0.0, min(1.0, lossScore))

		// Stability factor
		score += 0.2 * quality.Stability

		// Recency factor
		if !quality.LastMeasurement.IsZero() {
			age := time.Since(quality.LastMeasurement)
			recencyScore := 1.0 - (float64(age.Minutes()) / 60.0) // 1 hour decay
			score += 0.2 * max(0.0, min(1.0, recencyScore))
		}
	}

	// Path length factor (shorter is generally better)
	hopCount := len(pathInfo.Hops)
	if hopCount > 0 {
		hopScore := 1.0 - (float64(hopCount-1) / 10.0) // Penalty for long paths
		score += 0.1 * max(0.0, min(1.0, hopScore))
	}

	return max(0.0, min(1.0, score))
}

// cachePaths caches the paths for a destination
func (pm *PathManager) cachePaths(dstIA string, pathInfos []*PathInfo) {
	pm.pathMutex.Lock()
	defer pm.pathMutex.Unlock()

	// Check cache size limit
	if len(pm.pathCache) >= pm.config.CacheSize {
		pm.evictOldestCache()
	}

	pm.pathCache[dstIA] = &CachedPathSet{
		DstIA:       pathInfos[0].DstIA,
		Paths:       pathInfos,
		LastRefresh: time.Now(),
		QueryCount:  1,
		UsageCount:  0,
	}
}

// isStale checks if cached paths are stale
func (pm *PathManager) isStale(cachedSet *CachedPathSet) bool {
	return time.Since(cachedSet.LastRefresh) > pm.config.MaxPathAge
}

// evictOldestCache evicts the oldest cache entry
func (pm *PathManager) evictOldestCache() {
	var oldestKey string
	var oldestTime time.Time

	for key, pathSet := range pm.pathCache {
		if oldestTime.IsZero() || pathSet.LastRefresh.Before(oldestTime) {
			oldestTime = pathSet.LastRefresh
			oldestKey = key
		}
	}

	if oldestKey != "" {
		delete(pm.pathCache, oldestKey)
		log.WithField("evicted_dst", oldestKey).Debug("Evicted oldest path cache entry")
	}
}

// backgroundRefresh performs background path refresh
func (pm *PathManager) backgroundRefresh() {
	log.Info("Starting background path refresh")
	defer log.Info("Background path refresh stopped")

	for {
		select {
		case <-pm.ctx.Done():
			return
		case <-pm.refreshTicker.C:
			pm.performBackgroundRefresh()
		}
	}
}

// performBackgroundRefresh refreshes stale cached paths
func (pm *PathManager) performBackgroundRefresh() {
	pm.recordRefreshOperation()

	pm.pathMutex.RLock()
	staleDestinations := make([]string, 0)

	for dstIA, pathSet := range pm.pathCache {
		if pm.isStale(pathSet) {
			staleDestinations = append(staleDestinations, dstIA)
		}
	}
	pm.pathMutex.RUnlock()

	if len(staleDestinations) == 0 {
		return
	}

	log.WithField("stale_destinations", len(staleDestinations)).Debug("Refreshing stale path cache entries")

	for _, dstIA := range staleDestinations {
		// Refresh paths asynchronously
		go func(dst string) {
			if _, err := pm.QueryPaths(dst); err != nil {
				log.WithError(err).WithField("dst_ia", dst).Error("Failed to refresh paths")
			}
		}(dstIA)
	}
}

// SelectBestPath selects the best path to a destination
func (pm *PathManager) SelectBestPath(dstIA string, preferences *PathPreferences) (*PathInfo, error) {
	paths, err := pm.QueryPaths(dstIA)
	if err != nil {
		return nil, err
	}

	if len(paths) == 0 {
		return nil, fmt.Errorf("no paths available to %s", dstIA)
	}

	// Apply preferences and re-score if needed
	if preferences != nil {
		paths = pm.applyPreferences(paths, preferences)
	}

	// Return the best path (first in sorted list)
	bestPath := paths[0]

	// Update usage statistics
	pm.pathMutex.Lock()
	if cachedSet, exists := pm.pathCache[dstIA]; exists {
		cachedSet.UsageCount++
		bestPath.LastUsed = time.Now()
		bestPath.UsageCount++
	}
	pm.pathMutex.Unlock()

	pm.recordPathUsed()

	return bestPath, nil
}

// applyPreferences applies path selection preferences
func (pm *PathManager) applyPreferences(paths []*PathInfo, prefs *PathPreferences) []*PathInfo {
	// Filter and re-score paths based on preferences
	filtered := make([]*PathInfo, 0, len(paths))

	for _, path := range paths {
		// Apply filters
		if pm.matchesPreferences(path, prefs) {
			// Recalculate score with preferences
			path.SelectionScore = pm.calculatePreferenceScore(path, prefs)
			filtered = append(filtered, path)
		}
	}

	// Re-sort by preference-adjusted score
	sort.Slice(filtered, func(i, j int) bool {
		return filtered[i].SelectionScore > filtered[j].SelectionScore
	})

	return filtered
}

// matchesPreferences checks if a path matches selection preferences
func (pm *PathManager) matchesPreferences(path *PathInfo, prefs *PathPreferences) bool {
	if prefs == nil {
		return true
	}

	// Check RTT constraint
	if prefs.MaxRTTMs > 0 && path.Quality != nil {
		if path.Quality.RTT_EWMA.Milliseconds() > int64(prefs.MaxRTTMs) {
			return false
		}
	}

	// Check bandwidth constraint
	if prefs.MinBandwidthMbps > 0 && path.Quality != nil {
		mbps := float64(path.Quality.Bandwidth) / (1024 * 1024)
		if mbps < prefs.MinBandwidthMbps {
			return false
		}
	}

	// Check hop count constraint
	if prefs.MaxHops > 0 && len(path.Hops) > int(prefs.MaxHops) {
		return false
	}

	// Check avoid/prefer AS lists
	// TODO: Implement AS filtering based on path hops

	return true
}

// calculatePreferenceScore calculates score with preferences applied
func (pm *PathManager) calculatePreferenceScore(path *PathInfo, prefs *PathPreferences) float64 {
	score := pm.calculateSelectionScore(path)

	if prefs == nil {
		return score
	}

	// Apply preference bonuses/penalties
	if prefs.PreferLowLatency && path.Quality != nil && path.Quality.RTT_EWMA > 0 {
		latencyBonus := 1.0 - (float64(path.Quality.RTT_EWMA.Milliseconds()) / 1000.0)
		score += 0.1 * max(0.0, min(1.0, latencyBonus))
	}

	if prefs.PreferHighBandwidth && path.Quality != nil && path.Quality.Bandwidth > 0 {
		bandwidthBonus := float64(path.Quality.Bandwidth) / (100 * 1024 * 1024) // Normalize to 100Mbps
		score += 0.1 * max(0.0, min(1.0, bandwidthBonus))
	}

	if prefs.PreferStable && path.Quality != nil {
		score += 0.1 * path.Quality.Stability
	}

	return max(0.0, min(1.0, score))
}

// UpdatePathQuality updates quality metrics for a path
func (pm *PathManager) UpdatePathQuality(pathID string, rtt time.Duration, success bool) {
	pm.qualityMutex.Lock()
	defer pm.qualityMutex.Unlock()

	quality, exists := pm.qualityTracker[pathID]
	if !exists {
		quality = &PathQuality{
			PathID:    pathID,
			Stability: 1.0,
		}
		pm.qualityTracker[pathID] = quality
	}

	// Update RTT EWMA (exponentially weighted moving average)
	if quality.RTT_EWMA == 0 {
		quality.RTT_EWMA = rtt
	} else {
		alpha := 0.125 // Standard TCP alpha
		quality.RTT_EWMA = time.Duration(float64(quality.RTT_EWMA)*(1-alpha) + float64(rtt)*alpha)
	}

	// Update loss rate
	quality.MeasurementCount++
	if !success {
		quality.FailureCount++
	}
	quality.LossRate = float64(quality.FailureCount) / float64(quality.MeasurementCount)

	// Update stability (penalize failures)
	if !success {
		quality.Stability *= 0.9 // Reduce stability on failure
	} else {
		quality.Stability = min(1.0, quality.Stability*1.01) // Slowly increase on success
	}

	quality.LastMeasurement = time.Now()

	log.WithFields(log.Fields{
		"path_id":     pathID,
		"rtt_ewma":    quality.RTT_EWMA,
		"loss_rate":   quality.LossRate,
		"stability":   quality.Stability,
		"success":     success,
	}).Debug("Updated path quality")
}

// GetStats returns path manager statistics
func (pm *PathManager) GetStats() PathManagerStats {
	pm.stats.RLock()
	defer pm.stats.RUnlock()

	pm.pathMutex.RLock()
	cacheSize := len(pm.pathCache)
	pm.pathMutex.RUnlock()

	return PathManagerStats{
		PathsDiscovered:   pm.stats.PathsDiscovered,
		PathsUsed:         pm.stats.PathsUsed,
		PathQueries:       pm.stats.PathQueries,
		CacheHits:         pm.stats.CacheHits,
		CacheMisses:       pm.stats.CacheMisses,
		RefreshOperations: pm.stats.RefreshOperations,
		Errors:            pm.stats.Errors,
		CacheSize:         uint64(cacheSize),
		CacheCapacity:     uint64(pm.config.CacheSize),
	}
}

// Stop gracefully stops the path manager
func (pm *PathManager) Stop() error {
	log.Info("Stopping path manager")

	pm.cancel()
	pm.refreshTicker.Stop()

	if pm.daemon != nil {
		if err := pm.daemon.Close(); err != nil {
			log.WithError(err).Error("Failed to close daemon connection")
		}
	}

	log.Info("Path manager stopped")
	return nil
}

// Helper functions and statistics recording

func (pm *PathManager) recordPathQuery() {
	pm.stats.Lock()
	pm.stats.PathQueries++
	pm.stats.Unlock()
}

func (pm *PathManager) recordCacheHit() {
	pm.stats.Lock()
	pm.stats.CacheHits++
	pm.stats.Unlock()
}

func (pm *PathManager) recordCacheMiss() {
	pm.stats.Lock()
	pm.stats.CacheMisses++
	pm.stats.Unlock()
}

func (pm *PathManager) recordPathsDiscovered(count int) {
	pm.stats.Lock()
	pm.stats.PathsDiscovered += uint64(count)
	pm.stats.Unlock()
}

func (pm *PathManager) recordPathUsed() {
	pm.stats.Lock()
	pm.stats.PathsUsed++
	pm.stats.Unlock()
}

func (pm *PathManager) recordRefreshOperation() {
	pm.stats.Lock()
	pm.stats.RefreshOperations++
	pm.stats.Unlock()
}

func (pm *PathManager) recordError() {
	pm.stats.Lock()
	pm.stats.Errors++
	pm.stats.Unlock()
}

// Utility functions
func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

// PathPreferences represents path selection preferences
type PathPreferences struct {
	PreferLowLatency     bool
	PreferHighBandwidth  bool
	PreferStable         bool
	AvoidASes            []string
	PreferASes           []string
	MaxRTTMs             uint32
	MinBandwidthMbps     float64
	MaxHops              uint32
}

// PathManagerStats represents path manager statistics
type PathManagerStats struct {
	PathsDiscovered   uint64
	PathsUsed         uint64
	PathQueries       uint64
	CacheHits         uint64
	CacheMisses       uint64
	RefreshOperations uint64
	Errors            uint64
	CacheSize         uint64
	CacheCapacity     uint64
}
