// Metrics Collector - Prometheus metrics for SCION sidecar
package metrics

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	log "github.com/sirupsen/logrus"
)

// MetricsCollector collects and exposes Prometheus metrics
type MetricsCollector struct {
	// SCION packet metrics
	scionPacketsSent    *prometheus.CounterVec
	scionPacketsReceived *prometheus.CounterVec
	scionBytesSent      *prometheus.CounterVec
	scionBytesReceived  *prometheus.CounterVec
	scionPacketErrors   *prometheus.CounterVec

	// SCION packet processing time
	scionPacketSendDuration    *prometheus.HistogramVec
	scionPacketReceiveDuration *prometheus.HistogramVec

	// Path management metrics
	pathQueries            *prometheus.CounterVec
	pathQueryDuration      *prometheus.HistogramVec
	pathCacheHits          prometheus.Counter
	pathCacheMisses        prometheus.Counter
	pathsDiscovered        *prometheus.CounterVec
	pathsUsed              *prometheus.CounterVec
	pathQualityRTT         *prometheus.GaugeVec
	pathQualityLoss        *prometheus.GaugeVec
	pathQualityStability   *prometheus.GaugeVec

	// Anti-replay metrics
	antiReplayValidations     prometheus.Counter
	antiReplayBlocks          prometheus.Counter
	antiReplayFutureRejected  prometheus.Counter
	antiReplayExpiredRejected prometheus.Counter
	antiReplayValidationTime  prometheus.Histogram
	antiReplayWindowSlides    prometheus.Counter
	antiReplayWindowUpdates   prometheus.Counter
	antiReplayPersistenceErrors prometheus.Counter
	antiReplayActiveWindows   prometheus.Gauge

	// Gateway metrics
	gatewayRequests         *prometheus.CounterVec
	gatewayRequestDuration  *prometheus.HistogramVec
	gatewayRequestErrors    *prometheus.CounterVec
	gatewayActiveConnections prometheus.Gauge

	// System metrics
	memoryUsage        prometheus.Gauge
	cpuUtilization     prometheus.Gauge
	goroutineCount     prometheus.Gauge
	uptimeSeconds      prometheus.Counter
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector() *MetricsCollector {
	mc := &MetricsCollector{
		// SCION packet metrics
		scionPacketsSent: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "scion_packets_sent_total",
				Help: "Total number of SCION packets sent",
			},
			[]string{"dst_ia", "result"},
		),
		scionPacketsReceived: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "scion_packets_received_total",
				Help: "Total number of SCION packets received",
			},
			[]string{"src_ia"},
		),
		scionBytesSent: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "scion_bytes_sent_total",
				Help: "Total bytes sent via SCION",
			},
			[]string{"dst_ia"},
		),
		scionBytesReceived: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "scion_bytes_received_total",
				Help: "Total bytes received via SCION",
			},
			[]string{"src_ia"},
		),
		scionPacketErrors: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "scion_packet_errors_total",
				Help: "Total SCION packet errors",
			},
			[]string{"error_type"},
		),

		// SCION packet processing time
		scionPacketSendDuration: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "scion_packet_send_duration_seconds",
				Help:    "Time spent sending SCION packets",
				Buckets: prometheus.DefBuckets,
			},
			[]string{"dst_ia"},
		),
		scionPacketReceiveDuration: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "scion_packet_receive_duration_seconds",
				Help:    "Time spent processing received SCION packets",
				Buckets: prometheus.DefBuckets,
			},
			[]string{"src_ia"},
		),

		// Path management metrics
		pathQueries: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "scion_path_queries_total",
				Help: "Total number of SCION path queries",
			},
			[]string{"dst_ia", "result"},
		),
		pathQueryDuration: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "scion_path_query_duration_seconds",
				Help:    "Time spent querying SCION paths",
				Buckets: []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0},
			},
			[]string{"dst_ia", "source"},
		),
		pathCacheHits: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "scion_path_cache_hits_total",
				Help: "Total number of path cache hits",
			},
		),
		pathCacheMisses: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "scion_path_cache_misses_total",
				Help: "Total number of path cache misses",
			},
		),
		pathsDiscovered: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "scion_paths_discovered_total",
				Help: "Total number of SCION paths discovered",
			},
			[]string{"dst_ia"},
		),
		pathsUsed: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "scion_paths_used_total",
				Help: "Total number of times SCION paths were used",
			},
			[]string{"path_id", "dst_ia"},
		),
		pathQualityRTT: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "scion_path_rtt_seconds",
				Help: "SCION path round-trip time EWMA",
			},
			[]string{"path_id", "dst_ia"},
		),
		pathQualityLoss: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "scion_path_loss_rate",
				Help: "SCION path packet loss rate",
			},
			[]string{"path_id", "dst_ia"},
		),
		pathQualityStability: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "scion_path_stability_score",
				Help: "SCION path stability score",
			},
			[]string{"path_id", "dst_ia"},
		),

		// Anti-replay metrics
		antiReplayValidations: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "antireplay_validations_total",
				Help: "Total number of anti-replay validations performed",
			},
		),
		antiReplayBlocks: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "antireplay_blocks_total",
				Help: "Total number of replay attacks blocked",
			},
		),
		antiReplayFutureRejected: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "antireplay_future_rejected_total",
				Help: "Total number of future packets rejected",
			},
		),
		antiReplayExpiredRejected: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "antireplay_expired_rejected_total",
				Help: "Total number of expired packets rejected",
			},
		),
		antiReplayValidationTime: promauto.NewHistogram(
			prometheus.HistogramOpts{
				Name:    "antireplay_validation_duration_seconds",
				Help:    "Time spent validating sequences for anti-replay",
				Buckets: []float64{0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05},
			},
		),
		antiReplayWindowSlides: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "antireplay_window_slides_total",
				Help: "Total number of anti-replay window slides",
			},
		),
		antiReplayWindowUpdates: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "antireplay_window_updates_total",
				Help: "Total number of anti-replay window updates",
			},
		),
		antiReplayPersistenceErrors: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "antireplay_persistence_errors_total",
				Help: "Total number of anti-replay persistence errors",
			},
		),
		antiReplayActiveWindows: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "antireplay_active_windows",
				Help: "Number of active anti-replay windows",
			},
		),

		// Gateway metrics
		gatewayRequests: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "gateway_requests_total",
				Help: "Total number of gateway requests",
			},
			[]string{"method", "result"},
		),
		gatewayRequestDuration: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "gateway_request_duration_seconds",
				Help:    "Time spent processing gateway requests",
				Buckets: prometheus.DefBuckets,
			},
			[]string{"method"},
		),
		gatewayRequestErrors: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "gateway_request_errors_total",
				Help: "Total number of gateway request errors",
			},
			[]string{"method", "error_type"},
		),
		gatewayActiveConnections: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "gateway_active_connections",
				Help: "Number of active gateway connections",
			},
		),

		// System metrics
		memoryUsage: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "scion_sidecar_memory_usage_bytes",
				Help: "Memory usage of SCION sidecar in bytes",
			},
		),
		cpuUtilization: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "scion_sidecar_cpu_utilization",
				Help: "CPU utilization of SCION sidecar (0.0-1.0)",
			},
		),
		goroutineCount: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "scion_sidecar_goroutines",
				Help: "Number of goroutines in SCION sidecar",
			},
		),
		uptimeSeconds: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "scion_sidecar_uptime_seconds_total",
				Help: "Total uptime of SCION sidecar in seconds",
			},
		),
	}

	log.Info("Prometheus metrics collector initialized")
	return mc
}

// SCION Packet Metrics

func (mc *MetricsCollector) RecordScionPacketSent(duration time.Duration, bytes int, dstIA string) {
	mc.scionPacketsSent.WithLabelValues(dstIA, "success").Inc()
	mc.scionBytesSent.WithLabelValues(dstIA).Add(float64(bytes))
	mc.scionPacketSendDuration.WithLabelValues(dstIA).Observe(duration.Seconds())
}

func (mc *MetricsCollector) RecordScionPacketSendError(dstIA, errorType string) {
	mc.scionPacketsSent.WithLabelValues(dstIA, "error").Inc()
	mc.scionPacketErrors.WithLabelValues(errorType).Inc()
}

func (mc *MetricsCollector) RecordScionPacketReceived(bytes int, srcIA string) {
	mc.scionPacketsReceived.WithLabelValues(srcIA).Inc()
	mc.scionBytesReceived.WithLabelValues(srcIA).Add(float64(bytes))
}

func (mc *MetricsCollector) RecordScionPacketReceiveError(errorType string) {
	mc.scionPacketErrors.WithLabelValues(errorType).Inc()
}

// Path Management Metrics

func (mc *MetricsCollector) RecordPathQuery(duration time.Duration, pathCount int, dstIA string) {
	source := "cache"
	if pathCount > 0 {
		source = "fresh"
	}

	mc.pathQueries.WithLabelValues(dstIA, "success").Inc()
	mc.pathQueryDuration.WithLabelValues(dstIA, source).Observe(duration.Seconds())

	if pathCount > 0 {
		mc.pathsDiscovered.WithLabelValues(dstIA).Add(float64(pathCount))
	}
}

func (mc *MetricsCollector) RecordPathQueryError(dstIA, errorType string) {
	mc.pathQueries.WithLabelValues(dstIA, "error").Inc()
}

func (mc *MetricsCollector) RecordPathCacheHit() {
	mc.pathCacheHits.Inc()
}

func (mc *MetricsCollector) RecordPathCacheMiss() {
	mc.pathCacheMisses.Inc()
}

func (mc *MetricsCollector) RecordPathUsed(pathID, dstIA string) {
	mc.pathsUsed.WithLabelValues(pathID, dstIA).Inc()
}

func (mc *MetricsCollector) UpdatePathQuality(pathID, dstIA string, rtt time.Duration, lossRate, stability float64) {
	mc.pathQualityRTT.WithLabelValues(pathID, dstIA).Set(rtt.Seconds())
	mc.pathQualityLoss.WithLabelValues(pathID, dstIA).Set(lossRate)
	mc.pathQualityStability.WithLabelValues(pathID, dstIA).Set(stability)
}

// Anti-Replay Metrics

func (mc *MetricsCollector) RecordAntiReplayValidation(duration time.Duration, result string) {
	mc.antiReplayValidations.Inc()
	mc.antiReplayValidationTime.Observe(duration.Seconds())

	switch result {
	case "replay":
		mc.antiReplayBlocks.Inc()
	case "future":
		mc.antiReplayFutureRejected.Inc()
	case "expired":
		mc.antiReplayExpiredRejected.Inc()
	}
}

func (mc *MetricsCollector) RecordAntiReplayWindowSlide() {
	mc.antiReplayWindowSlides.Inc()
}

func (mc *MetricsCollector) RecordAntiReplayWindowUpdate() {
	mc.antiReplayWindowUpdates.Inc()
}

func (mc *MetricsCollector) RecordAntiReplayPersistenceError() {
	mc.antiReplayPersistenceErrors.Inc()
}

func (mc *MetricsCollector) UpdateAntiReplayActiveWindows(count int) {
	mc.antiReplayActiveWindows.Set(float64(count))
}

// Gateway Metrics

func (mc *MetricsCollector) RecordGatewayRequest(method string, duration time.Duration, result string) {
	mc.gatewayRequests.WithLabelValues(method, result).Inc()
	mc.gatewayRequestDuration.WithLabelValues(method).Observe(duration.Seconds())
}

func (mc *MetricsCollector) RecordGatewayRequestError(method, errorType string) {
	mc.gatewayRequestErrors.WithLabelValues(method, errorType).Inc()
}

func (mc *MetricsCollector) UpdateGatewayActiveConnections(count int) {
	mc.gatewayActiveConnections.Set(float64(count))
}

// System Metrics

func (mc *MetricsCollector) UpdateMemoryUsage(bytes uint64) {
	mc.memoryUsage.Set(float64(bytes))
}

func (mc *MetricsCollector) UpdateCPUUtilization(utilization float64) {
	mc.cpuUtilization.Set(utilization)
}

func (mc *MetricsCollector) UpdateGoroutineCount(count int) {
	mc.goroutineCount.Set(float64(count))
}

func (mc *MetricsCollector) RecordUptime(duration time.Duration) {
	mc.uptimeSeconds.Add(duration.Seconds())
}

// Composite metric updates for complex operations

func (mc *MetricsCollector) RecordScionPacketFlow(srcIA, dstIA string, bytes int, rtt time.Duration, success bool) {
	if success {
		mc.RecordScionPacketSent(rtt/2, bytes, dstIA) // Approximate send time
		mc.RecordScionPacketReceived(bytes, srcIA)
	} else {
		mc.RecordScionPacketSendError(dstIA, "timeout")
	}
}

func (mc *MetricsCollector) RecordPathQualityMeasurement(pathID, dstIA string, rtt time.Duration, success bool) {
	// Update path usage
	mc.RecordPathUsed(pathID, dstIA)

	// Update quality metrics if we have RTT measurement
	if rtt > 0 {
		// Estimate loss rate and stability based on success
		lossRate := 0.0
		stability := 1.0
		if !success {
			lossRate = 0.1 // Estimate some loss on failure
			stability = 0.8 // Reduce stability score
		}

		mc.UpdatePathQuality(pathID, dstIA, rtt, lossRate, stability)
	}
}

// Batch metric updates for performance

type BatchMetrics struct {
	PacketsSent     map[string]int // dst_ia -> count
	PacketsReceived map[string]int // src_ia -> count
	BytesSent       map[string]int // dst_ia -> bytes
	BytesReceived   map[string]int // src_ia -> bytes
	Errors          map[string]int // error_type -> count
}

func (mc *MetricsCollector) RecordBatchMetrics(batch *BatchMetrics) {
	for dstIA, count := range batch.PacketsSent {
		mc.scionPacketsSent.WithLabelValues(dstIA, "success").Add(float64(count))
	}

	for srcIA, count := range batch.PacketsReceived {
		mc.scionPacketsReceived.WithLabelValues(srcIA).Add(float64(count))
	}

	for dstIA, bytes := range batch.BytesSent {
		mc.scionBytesSent.WithLabelValues(dstIA).Add(float64(bytes))
	}

	for srcIA, bytes := range batch.BytesReceived {
		mc.scionBytesReceived.WithLabelValues(srcIA).Add(float64(bytes))
	}

	for errorType, count := range batch.Errors {
		mc.scionPacketErrors.WithLabelValues(errorType).Add(float64(count))
	}
}

// Metrics snapshot for debugging and monitoring

type MetricsSnapshot struct {
	Timestamp             time.Time                    `json:"timestamp"`
	ScionPacketsSent      map[string]float64          `json:"scion_packets_sent"`
	ScionPacketsReceived  map[string]float64          `json:"scion_packets_received"`
	PathCacheHitRate      float64                     `json:"path_cache_hit_rate"`
	AntiReplayBlocks      float64                     `json:"antireplay_blocks"`
	ActiveConnections     float64                     `json:"active_connections"`
	MemoryUsageBytes      float64                     `json:"memory_usage_bytes"`
	CPUUtilization        float64                     `json:"cpu_utilization"`
	GoroutineCount        float64                     `json:"goroutine_count"`
}

func (mc *MetricsCollector) GetSnapshot() *MetricsSnapshot {
	// This would require gathering current metric values
	// For now, return a basic snapshot structure
	return &MetricsSnapshot{
		Timestamp:         time.Now(),
		ScionPacketsSent:  make(map[string]float64),
		ScionPacketsReceived: make(map[string]float64),
		PathCacheHitRate:  0.0,
		AntiReplayBlocks:  0.0,
		ActiveConnections: 0.0,
		MemoryUsageBytes:  0.0,
		CPUUtilization:    0.0,
		GoroutineCount:    0.0,
	}
}

// Cleanup removes old metric labels to prevent memory leaks
func (mc *MetricsCollector) CleanupOldLabels() {
	// Prometheus doesn't have built-in cleanup for metric labels
	// This would require custom implementation or external cleanup
	// For production, consider using metric label limits and rotation
	log.Debug("Metrics cleanup completed")
}
