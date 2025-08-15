// Anti-Replay Manager Tests
package anti_replay

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/aivillage/scion-sidecar/internal/metrics"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAntiReplayManager_BasicValidation(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "antireplay_test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	config := &Config{
		WindowSize:      64,
		DBPath:         tempDir + "/test.db",
		CleanupTTL:     1 * time.Hour,
		CleanupInterval: 1 * time.Minute,
		SyncInterval:   10 * time.Second,
	}

	metrics := metrics.NewMetricsCollector()
	manager, err := NewAntiReplayManager(config, metrics)
	require.NoError(t, err)
	defer manager.Stop()

	peerID := "test_peer"

	// Test cases for sequence validation
	tests := []struct {
		name            string
		sequence        uint64
		timestampNs     int64
		updateWindow    bool
		expectedValid   bool
		expectedReason  string
	}{
		{
			name:           "first sequence accepted",
			sequence:       1,
			timestampNs:    time.Now().UnixNano(),
			updateWindow:   true,
			expectedValid:  true,
			expectedReason: "",
		},
		{
			name:           "replay rejected",
			sequence:       1,
			timestampNs:    time.Now().UnixNano(),
			updateWindow:   true,
			expectedValid:  false,
			expectedReason: "replay",
		},
		{
			name:           "future sequence accepted",
			sequence:       10,
			timestampNs:    time.Now().UnixNano(),
			updateWindow:   true,
			expectedValid:  true,
			expectedReason: "",
		},
		{
			name:           "expired sequence rejected",
			sequence:       0,
			timestampNs:    time.Now().UnixNano(),
			updateWindow:   true,
			expectedValid:  false,
			expectedReason: "expired",
		},
		{
			name:           "far future sequence rejected",
			sequence:       1000,
			timestampNs:    time.Now().UnixNano(),
			updateWindow:   true,
			expectedValid:  false,
			expectedReason: "future",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := manager.ValidateSequence(peerID, tt.sequence, tt.timestampNs, tt.updateWindow)

			assert.Equal(t, tt.expectedValid, result.Valid, "validation result")
			if !tt.expectedValid {
				assert.Equal(t, tt.expectedReason, result.RejectionReason, "rejection reason")
			}
			assert.Greater(t, result.ValidationTimeUs, uint64(0), "validation time should be recorded")
			assert.NotNil(t, result.WindowState, "window state should be returned")
		})
	}
}

func TestAntiReplayManager_WindowSliding(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "antireplay_test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	config := &Config{
		WindowSize: 64,
		DBPath:    tempDir + "/test.db",
	}

	metrics := metrics.NewMetricsCollector()
	manager, err := NewAntiReplayManager(config, metrics)
	require.NoError(t, err)
	defer manager.Stop()

	peerID := "slide_test_peer"
	timestamp := time.Now().UnixNano()

	// Accept first sequence
	result1 := manager.ValidateSequence(peerID, 1, timestamp, true)
	assert.True(t, result1.Valid)

	// Accept sequence that causes window slide
	slideSequence := uint64(1 + config.WindowSize + 1)
	result2 := manager.ValidateSequence(peerID, slideSequence, timestamp, true)
	assert.True(t, result2.Valid)

	// Verify window base has moved
	assert.Greater(t, result2.WindowState.WindowBase, uint64(0))

	// Previous sequence should now be expired
	result3 := manager.ValidateSequence(peerID, 1, timestamp, true)
	assert.False(t, result3.Valid)
	assert.Equal(t, "expired", result3.RejectionReason)
}

func TestAntiReplayManager_Persistence(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "antireplay_test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	config := &Config{
		WindowSize:   64,
		DBPath:      tempDir + "/test.db",
		SyncInterval: 100 * time.Millisecond, // Fast sync for testing
	}

	metrics := metrics.NewMetricsCollector()

	peerID := "persist_test_peer"
	timestamp := time.Now().UnixNano()

	// First manager instance
	{
		manager1, err := NewAntiReplayManager(config, metrics)
		require.NoError(t, err)

		// Accept some sequences
		result1 := manager1.ValidateSequence(peerID, 1, timestamp, true)
		assert.True(t, result1.Valid)

		result2 := manager1.ValidateSequence(peerID, 3, timestamp, true)
		assert.True(t, result2.Valid)

		result3 := manager1.ValidateSequence(peerID, 5, timestamp, true)
		assert.True(t, result3.Valid)

		// Wait for sync
		time.Sleep(200 * time.Millisecond)

		manager1.Stop()
	}

	// Second manager instance - should load persisted state
	{
		manager2, err := NewAntiReplayManager(config, metrics)
		require.NoError(t, err)
		defer manager2.Stop()

		// Previously accepted sequences should be rejected as replays
		result1 := manager2.ValidateSequence(peerID, 1, timestamp, true)
		assert.False(t, result1.Valid)
		assert.Equal(t, "replay", result1.RejectionReason)

		result2 := manager2.ValidateSequence(peerID, 3, timestamp, true)
		assert.False(t, result2.Valid)
		assert.Equal(t, "replay", result2.RejectionReason)

		result3 := manager2.ValidateSequence(peerID, 5, timestamp, true)
		assert.False(t, result3.Valid)
		assert.Equal(t, "replay", result3.RejectionReason)

		// New sequence should be accepted
		result4 := manager2.ValidateSequence(peerID, 7, timestamp, true)
		assert.True(t, result4.Valid)
	}
}

func TestAntiReplayManager_ConcurrentAccess(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "antireplay_test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	config := &Config{
		WindowSize: 1024, // Large window for concurrent test
		DBPath:    tempDir + "/test.db",
	}

	metrics := metrics.NewMetricsCollector()
	manager, err := NewAntiReplayManager(config, metrics)
	require.NoError(t, err)
	defer manager.Stop()

	peerID := "concurrent_test_peer"
	timestamp := time.Now().UnixNano()

	// Run concurrent validations
	numGoroutines := 10
	sequencesPerGoroutine := 100
	done := make(chan bool, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(offset int) {
			defer func() { done <- true }()

			for j := 0; j < sequencesPerGoroutine; j++ {
				sequence := uint64(offset*sequencesPerGoroutine + j + 1)
				result := manager.ValidateSequence(peerID, sequence, timestamp, true)

				// All sequences should be accepted (no duplicates across goroutines)
				assert.True(t, result.Valid, "sequence %d should be valid", sequence)
			}
		}(i)
	}

	// Wait for all goroutines to complete
	for i := 0; i < numGoroutines; i++ {
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatal("Test timed out waiting for concurrent validations")
		}
	}

	// Verify total processed count
	stats := manager.GetStats()
	expectedTotal := uint64(numGoroutines * sequencesPerGoroutine)
	assert.Equal(t, expectedTotal, stats.TotalValidated, "total validations")
	assert.Equal(t, uint64(0), stats.ReplaysBlocked, "no replays should be detected")
}

func TestAntiReplayManager_StatisticsAccuracy(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "antireplay_test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	config := &Config{
		WindowSize: 64,
		DBPath:    tempDir + "/test.db",
	}

	metrics := metrics.NewMetricsCollector()
	manager, err := NewAntiReplayManager(config, metrics)
	require.NoError(t, err)
	defer manager.Stop()

	peerID := "stats_test_peer"
	timestamp := time.Now().UnixNano()

	// Test various validation outcomes

	// 1. Valid new sequence
	result1 := manager.ValidateSequence(peerID, 1, timestamp, true)
	assert.True(t, result1.Valid)

	// 2. Replay attack
	result2 := manager.ValidateSequence(peerID, 1, timestamp, true)
	assert.False(t, result2.Valid)
	assert.Equal(t, "replay", result2.RejectionReason)

	// 3. Expired sequence
	result3 := manager.ValidateSequence(peerID, 0, timestamp, true)
	assert.False(t, result3.Valid)
	assert.Equal(t, "expired", result3.RejectionReason)

	// 4. Far future sequence
	result4 := manager.ValidateSequence(peerID, 10000, timestamp, true)
	assert.False(t, result4.Valid)
	assert.Equal(t, "future", result4.RejectionReason)

	// 5. Valid future sequence (causes slide)
	result5 := manager.ValidateSequence(peerID, 100, timestamp, true)
	assert.True(t, result5.Valid)

	// Check statistics
	stats := manager.GetStats()
	assert.Equal(t, uint64(5), stats.TotalValidated, "total validations")
	assert.Equal(t, uint64(1), stats.ReplaysBlocked, "replay blocks")
	assert.Equal(t, uint64(1), stats.ExpiredRejected, "expired rejections")
	assert.Equal(t, uint64(1), stats.FutureRejected, "future rejections")
	assert.Greater(t, stats.AverageValidationTimeUs, uint64(0), "average validation time")
}

func TestAntiReplayManager_WindowEviction(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "antireplay_test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	config := &Config{
		WindowSize:      64,
		DBPath:         tempDir + "/test.db",
		CleanupTTL:     100 * time.Millisecond, // Fast cleanup for testing
		CleanupInterval: 50 * time.Millisecond,
	}

	metrics := metrics.NewMetricsCollector()
	manager, err := NewAntiReplayManager(config, metrics)
	require.NoError(t, err)
	defer manager.Stop()

	peerID := "evict_test_peer"
	timestamp := time.Now().UnixNano()

	// Create a window
	result1 := manager.ValidateSequence(peerID, 1, timestamp, true)
	assert.True(t, result1.Valid)

	// Wait for cleanup to occur
	time.Sleep(200 * time.Millisecond)

	// Window should be evicted, so next validation should create new window
	result2 := manager.ValidateSequence(peerID, 1, timestamp, true)
	assert.True(t, result2.Valid, "sequence should be accepted after window eviction")
}

// Benchmark tests for performance validation

func BenchmarkAntiReplayManager_ValidateSequence(b *testing.B) {
	tempDir, err := os.MkdirTemp("", "antireplay_bench")
	require.NoError(b, err)
	defer os.RemoveAll(tempDir)

	config := &Config{
		WindowSize: 1024,
		DBPath:    tempDir + "/bench.db",
	}

	metrics := metrics.NewMetricsCollector()
	manager, err := NewAntiReplayManager(config, metrics)
	require.NoError(b, err)
	defer manager.Stop()

	peerID := "bench_peer"
	timestamp := time.Now().UnixNano()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		sequence := uint64(i + 1)
		result := manager.ValidateSequence(peerID, sequence, timestamp, true)
		if !result.Valid {
			b.Fatalf("Validation failed for sequence %d: %s", sequence, result.RejectionReason)
		}
	}
}

func BenchmarkAntiReplayManager_ReplayDetection(b *testing.B) {
	tempDir, err := os.MkdirTemp("", "antireplay_bench")
	require.NoError(b, err)
	defer os.RemoveAll(tempDir)

	config := &Config{
		WindowSize: 1024,
		DBPath:    tempDir + "/bench.db",
	}

	metrics := metrics.NewMetricsCollector()
	manager, err := NewAntiReplayManager(config, metrics)
	require.NoError(b, err)
	defer manager.Stop()

	peerID := "bench_peer"
	timestamp := time.Now().UnixNano()

	// Pre-populate window with some sequences
	for i := 1; i <= 100; i++ {
		manager.ValidateSequence(peerID, uint64(i), timestamp, true)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Try to replay a sequence from the middle of the window
		sequence := uint64((i % 100) + 1)
		result := manager.ValidateSequence(peerID, sequence, timestamp, false)
		if result.Valid {
			b.Fatalf("Replay should have been detected for sequence %d", sequence)
		}
	}
}
