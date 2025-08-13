// Gateway Service - gRPC service implementation for Betanet Gateway
package gateway

import (
	"context"
	"fmt"
	"net"
	"time"

	"github.com/aivillage/scion-sidecar/internal/anti_replay"
	"github.com/aivillage/scion-sidecar/internal/metrics"
	"github.com/aivillage/scion-sidecar/internal/paths"
	"github.com/aivillage/scion-sidecar/internal/scionio"
	log "github.com/sirupsen/logrus"
)

// Service implements the BetanetGateway gRPC service
type Service struct {
	UnimplementedBetanetGatewayServer
	
	scionIO     *scionio.ScionIOHandler
	pathManager *paths.PathManager
	antiReplay  *anti_replay.AntiReplayManager
	metrics     *metrics.MetricsCollector
}

// NewService creates a new gateway service
func NewService(
	scionIO *scionio.ScionIOHandler,
	pathManager *paths.PathManager,
	antiReplay *anti_replay.AntiReplayManager,
	metricsCollector *metrics.MetricsCollector,
) *Service {
	return &Service{
		scionIO:     scionIO,
		pathManager: pathManager,
		antiReplay:  antiReplay,
		metrics:     metricsCollector,
	}
}

// SendScionPacket sends a SCION packet to the specified destination
func (s *Service) SendScionPacket(ctx context.Context, req *SendScionPacketRequest) (*SendScionPacketResponse, error) {
	start := time.Now()
	
	log.WithFields(log.Fields{
		"dst_ia":         req.EgressIa,
		"sequence":       req.SequenceNumber,
		"packet_size":    len(req.RawPacket),
		"priority":       req.Priority,
		"correlation_id": req.CorrelationId,
	}).Debug("Processing SendScionPacket request")
	
	// Validate request
	if req.EgressIa == "" {
		s.metrics.RecordGatewayRequestError("SendScionPacket", "invalid_ia")
		return &SendScionPacketResponse{
			Success: false,
			Error:   "egress_ia is required",
		}, nil
	}
	
	if len(req.RawPacket) == 0 {
		s.metrics.RecordGatewayRequestError("SendScionPacket", "empty_packet")
		return &SendScionPacketResponse{
			Success: false,
			Error:   "raw_packet is required",
		}, nil
	}
	
	// Validate sequence number with anti-replay
	validationResult := s.antiReplay.ValidateSequence(
		req.EgressIa,
		req.SequenceNumber,
		req.TimestampNs,
		false, // Don't update window on send
	)
	
	if !validationResult.Valid {
		s.metrics.RecordGatewayRequestError("SendScionPacket", "sequence_validation_failed")
		return &SendScionPacketResponse{
			Success: false,
			Error:   fmt.Sprintf("sequence validation failed: %s", validationResult.RejectionReason),
		}, nil
	}
	
	// Select best path
	var selectedPath *paths.PathInfo
	if req.Preferences != nil {
		prefs := convertPathPreferences(req.Preferences)
		path, err := s.pathManager.SelectBestPath(req.EgressIa, prefs)
		if err != nil {
			s.metrics.RecordGatewayRequestError("SendScionPacket", "path_selection_failed")
			return &SendScionPacketResponse{
				Success: false,
				Error:   fmt.Sprintf("path selection failed: %v", err),
			}, nil
		}
		selectedPath = path
	} else {
		path, err := s.pathManager.SelectBestPath(req.EgressIa, nil)
		if err != nil {
			s.metrics.RecordGatewayRequestError("SendScionPacket", "path_selection_failed")
			return &SendScionPacketResponse{
				Success: false,
				Error:   fmt.Sprintf("path selection failed: %v", err),
			}, nil
		}
		selectedPath = path
	}
	
	// Parse egress interface
	var dstAddr *net.UDPAddr
	if req.EgressIface != "" {
		// Parse interface address
		addr, err := net.ResolveUDPAddr("udp", req.EgressIface)
		if err != nil {
			s.metrics.RecordGatewayRequestError("SendScionPacket", "invalid_interface")
			return &SendScionPacketResponse{
				Success: false,
				Error:   fmt.Sprintf("invalid egress interface: %v", err),
			}, nil
		}
		dstAddr = addr
	} else {
		// Use default port
		dstAddr = &net.UDPAddr{IP: net.IPv4(0, 0, 0, 0), Port: 30041}
	}
	
	// Send packet via SCION
	err := s.scionIO.SendPacket(ctx, req.RawPacket, req.EgressIa, dstAddr)
	if err != nil {
		s.metrics.RecordGatewayRequestError("SendScionPacket", "send_failed")
		s.metrics.RecordScionPacketSendError(req.EgressIa, "send_error")
		
		// Update path quality on failure
		s.pathManager.UpdatePathQuality(selectedPath.ID, 0, false)
		
		return &SendScionPacketResponse{
			Success: false,
			Error:   fmt.Sprintf("failed to send packet: %v", err),
		}, nil
	}
	
	// Update sequence window on successful send
	s.antiReplay.ValidateSequence(
		req.EgressIa,
		req.SequenceNumber,
		req.TimestampNs,
		true, // Update window
	)
	
	// Calculate delivery estimate based on path quality
	deliveryEstimateMs := uint32(100) // Default 100ms
	if selectedPath.Quality != nil && selectedPath.Quality.RTT_EWMA > 0 {
		deliveryEstimateMs = uint32(selectedPath.Quality.RTT_EWMA.Milliseconds())
	}
	
	processingTime := time.Since(start)
	
	// Record metrics
	s.metrics.RecordGatewayRequest("SendScionPacket", processingTime, "success")
	s.metrics.RecordScionPacketSent(processingTime, len(req.RawPacket), req.EgressIa)
	
	// Update path quality on success
	s.pathManager.UpdatePathQuality(selectedPath.ID, processingTime, true)
	
	response := &SendScionPacketResponse{
		Success:             true,
		SelectedPath:        convertPathInfo(selectedPath),
		DeliveryEstimateMs:  deliveryEstimateMs,
		ProcessingTimeUs:    uint64(processingTime.Microseconds()),
	}
	
	log.WithFields(log.Fields{
		"dst_ia":              req.EgressIa,
		"sequence":            req.SequenceNumber,
		"processing_time_us":  response.ProcessingTimeUs,
		"delivery_estimate_ms": response.DeliveryEstimateMs,
		"selected_path_id":    selectedPath.ID,
	}).Debug("SendScionPacket completed successfully")
	
	return response, nil
}

// RecvScionPacket handles received SCION packet notifications
func (s *Service) RecvScionPacket(ctx context.Context, req *RecvScionPacketRequest) (*RecvScionPacketResponse, error) {
	start := time.Now()
	
	log.WithFields(log.Fields{
		"src_ia":      req.IngressIa,
		"sequence":    req.SequenceNumber,
		"packet_size": len(req.RawPacket),
	}).Debug("Processing RecvScionPacket request")
	
	// Validate sequence number with anti-replay
	validationResult := s.antiReplay.ValidateSequence(
		req.IngressIa,
		req.SequenceNumber,
		req.TimestampNs,
		true, // Update window on receive
	)
	
	status := "accepted"
	if !validationResult.Valid {
		status = fmt.Sprintf("rejected_%s", validationResult.RejectionReason)
		s.metrics.RecordGatewayRequestError("RecvScionPacket", validationResult.RejectionReason)
	} else {
		// Record successful packet reception
		s.metrics.RecordScionPacketReceived(len(req.RawPacket), req.IngressIa)
		
		// Update path quality if path info is provided
		if req.PathInfo != nil && req.Quality != nil {
			pathID := req.PathInfo.PathId
			rtt := time.Duration(req.Quality.RttEwmaMs * float32(time.Millisecond))
			s.pathManager.UpdatePathQuality(pathID, rtt, true)
		}
	}
	
	processingTime := time.Since(start)
	s.metrics.RecordGatewayRequest("RecvScionPacket", processingTime, "success")
	
	response := &RecvScionPacketResponse{
		Acknowledged: validationResult.Valid,
		Status:       status,
	}
	
	log.WithFields(log.Fields{
		"src_ia":         req.IngressIa,
		"sequence":       req.SequenceNumber,
		"acknowledged":   response.Acknowledged,
		"status":         response.Status,
		"processing_us":  processingTime.Microseconds(),
	}).Debug("RecvScionPacket completed")
	
	return response, nil
}

// RegisterPath registers interest in paths to a destination
func (s *Service) RegisterPath(ctx context.Context, req *RegisterPathRequest) (*RegisterPathResponse, error) {
	start := time.Now()
	
	log.WithFields(log.Fields{
		"dst_ia":       req.DstIa,
		"ttl_seconds":  req.TtlSeconds,
		"callback":     req.CallbackAddr,
	}).Debug("Processing RegisterPath request")
	
	// Query current paths
	pathInfos, err := s.pathManager.QueryPaths(req.DstIa)
	if err != nil {
		s.metrics.RecordGatewayRequestError("RegisterPath", "path_query_failed")
		return &RegisterPathResponse{
			Success: false,
		}, nil
	}
	
	// Convert path infos to protobuf
	pathMetas := make([]*PathMeta, len(pathInfos))
	for i, pathInfo := range pathInfos {
		pathMetas[i] = convertPathInfo(pathInfo)
	}
	
	// Generate registration ID
	registrationID := fmt.Sprintf("reg_%s_%d", req.DstIa, time.Now().UnixNano())
	
	// Calculate expiry time
	expiresAt := time.Now().Add(time.Duration(req.TtlSeconds) * time.Second).UnixNano()
	
	processingTime := time.Since(start)
	s.metrics.RecordGatewayRequest("RegisterPath", processingTime, "success")
	
	response := &RegisterPathResponse{
		Success:        true,
		RegistrationId: registrationID,
		AvailablePaths: pathMetas,
		ExpiresAt:      expiresAt,
	}
	
	log.WithFields(log.Fields{
		"dst_ia":          req.DstIa,
		"registration_id": registrationID,
		"available_paths": len(pathMetas),
		"expires_at":      time.Unix(0, expiresAt),
	}).Debug("RegisterPath completed")
	
	return response, nil
}

// QueryPaths queries available paths to a destination
func (s *Service) QueryPaths(ctx context.Context, req *QueryPathsRequest) (*QueryPathsResponse, error) {
	start := time.Now()
	
	log.WithFields(log.Fields{
		"dst_ia":          req.DstIa,
		"include_expired": req.IncludeExpired,
		"limit":          req.Limit,
	}).Debug("Processing QueryPaths request")
	
	// Query paths from path manager
	pathInfos, err := s.pathManager.QueryPaths(req.DstIa)
	if err != nil {
		s.metrics.RecordGatewayRequestError("QueryPaths", "path_query_failed")
		return &QueryPathsResponse{
			QueryTime: time.Now().UnixNano(),
		}, nil
	}
	
	// Apply limit if specified
	if req.Limit > 0 && len(pathInfos) > int(req.Limit) {
		pathInfos = pathInfos[:req.Limit]
	}
	
	// Filter expired paths if requested
	if !req.IncludeExpired {
		now := time.Now()
		filteredPaths := make([]*paths.PathInfo, 0, len(pathInfos))
		for _, pathInfo := range pathInfos {
			if pathInfo.Expiry.After(now) {
				filteredPaths = append(filteredPaths, pathInfo)
			}
		}
		pathInfos = filteredPaths
	}
	
	// Convert to protobuf
	pathMetas := make([]*PathMeta, len(pathInfos))
	for i, pathInfo := range pathInfos {
		pathMetas[i] = convertPathInfo(pathInfo)
	}
	
	processingTime := time.Since(start)
	s.metrics.RecordGatewayRequest("QueryPaths", processingTime, "success")
	
	response := &QueryPathsResponse{
		Paths:       pathMetas,
		QueryTime:   time.Now().UnixNano(),
		NextRefresh: time.Now().Add(30 * time.Second).UnixNano(), // Next refresh in 30s
		TotalCount:  uint32(len(pathMetas)),
	}
	
	log.WithFields(log.Fields{
		"dst_ia":       req.DstIa,
		"paths_found":  len(pathMetas),
		"processing_us": processingTime.Microseconds(),
	}).Debug("QueryPaths completed")
	
	return response, nil
}

// Health returns the health status of the gateway
func (s *Service) Health(ctx context.Context, req *HealthRequest) (*HealthResponse, error) {
	start := time.Now()
	
	// Perform health checks based on requested level
	status := HealthResponse_HEALTHY
	components := make(map[string]string)
	
	// Basic health check
	components["scion_io"] = "healthy"
	components["path_manager"] = "healthy"
	components["anti_replay"] = "healthy"
	components["metrics"] = "healthy"
	
	if req.CheckLevel >= 1 {
		// Connectivity check
		// Check SCION daemon connectivity
		if _, err := s.pathManager.QueryPaths("1-ff00:0:110"); err != nil {
			components["scion_daemon"] = fmt.Sprintf("error: %v", err)
			status = HealthResponse_DEGRADED
		} else {
			components["scion_daemon"] = "connected"
		}
	}
	
	if req.CheckLevel >= 2 {
		// Full health check
		ioStats := s.scionIO.GetStats()
		pathStats := s.pathManager.GetStats()
		antiReplayStats := s.antiReplay.GetStats()
		
		components["io_errors"] = fmt.Sprintf("%d", ioStats.Errors)
		components["path_errors"] = fmt.Sprintf("%d", pathStats.Errors)
		components["replay_errors"] = fmt.Sprintf("%d", antiReplayStats.PersistenceErrors)
		
		// Check error rates
		if ioStats.Errors > 100 || pathStats.Errors > 50 || antiReplayStats.PersistenceErrors > 10 {
			status = HealthResponse_DEGRADED
		}
	}
	
	processingTime := time.Since(start)
	s.metrics.RecordGatewayRequest("Health", processingTime, "success")
	
	response := &HealthResponse{
		Status:     status,
		Components: components,
		Timestamp:  time.Now().UnixNano(),
		UptimeSeconds: uint64(time.Since(start).Seconds()), // Placeholder
		ScionDaemonConnected: components["scion_daemon"] == "connected",
		ActivePaths: 0, // Would need to query actual active paths
	}
	
	return response, nil
}

// GetStats returns gateway statistics
func (s *Service) GetStats(ctx context.Context, req *StatsRequest) (*StatsResponse, error) {
	start := time.Now()
	
	log.WithFields(log.Fields{
		"period_seconds":         req.PeriodSeconds,
		"include_path_breakdown": req.IncludePathBreakdown,
		"include_anti_replay":    req.IncludeAntiReplay,
	}).Debug("Processing GetStats request")
	
	// Get statistics from all components
	ioStats := s.scionIO.GetStats()
	pathStats := s.pathManager.GetStats()
	antiReplayStats := s.antiReplay.GetStats()
	
	// Build gateway stats
	gatewayStats := &GatewayStats{
		TotalPackets:       ioStats.PacketsSent + ioStats.PacketsReceived,
		TotalBytes:        ioStats.BytesSent + ioStats.BytesReceived,
		ProcessingRatePps: 0, // Would need rate calculation
		ThroughputBps:     0, // Would need rate calculation
		ActiveConnections: 1, // Placeholder
		ErrorCount:        ioStats.Errors + pathStats.Errors,
		AvgProcessingTimeUs: 1000, // Placeholder
		MemoryUsageBytes:    0, // Would need runtime stats
		CpuUtilization:      0, // Would need system stats
	}
	
	response := &StatsResponse{
		Gateway:       gatewayStats,
		Timestamp:     time.Now().UnixNano(),
		PeriodSeconds: req.PeriodSeconds,
	}
	
	// Include path breakdown if requested
	if req.IncludePathBreakdown {
		// Would populate with actual path statistics
		response.Paths = []*PathStats{}
	}
	
	// Include anti-replay stats if requested
	if req.IncludeAntiReplay {
		response.AntiReplay = &AntiReplayStats{
			TotalValidated:         antiReplayStats.TotalValidated,
			ReplaysBlocked:         antiReplayStats.ReplaysBlocked,
			FutureRejected:         antiReplayStats.FutureRejected,
			ExpiredRejected:        antiReplayStats.ExpiredRejected,
			FalsePositiveRate:      antiReplayStats.FalsePositiveRate,
			WindowUpdates:          antiReplayStats.WindowUpdates,
			AverageValidationTimeUs: antiReplayStats.AverageValidationTimeUs,
			PersistenceErrors:      antiReplayStats.PersistenceErrors,
		}
	}
	
	processingTime := time.Since(start)
	s.metrics.RecordGatewayRequest("GetStats", processingTime, "success")
	
	log.WithFields(log.Fields{
		"total_packets":    response.Gateway.TotalPackets,
		"total_bytes":      response.Gateway.TotalBytes,
		"processing_us":    processingTime.Microseconds(),
	}).Debug("GetStats completed")
	
	return response, nil
}

// ValidateSequence validates a sequence number for anti-replay
func (s *Service) ValidateSequence(ctx context.Context, req *ValidateSequenceRequest) (*ValidateSequenceResponse, error) {
	start := time.Now()
	
	// Perform validation
	result := s.antiReplay.ValidateSequence(
		req.PeerId,
		req.SequenceNumber,
		req.TimestampNs,
		req.UpdateWindow,
	)
	
	processingTime := time.Since(start)
	s.metrics.RecordGatewayRequest("ValidateSequence", processingTime, "success")
	
	// Convert window state
	var windowState *SequenceWindow
	if result.WindowState != nil {
		windowState = &SequenceWindow{
			PeerId:          result.WindowState.PeerID,
			WindowBase:      result.WindowState.WindowBase,
			WindowSize:      result.WindowState.WindowSize,
			WindowBitmap:    result.WindowState.WindowBitmap,
			HighestReceived: result.WindowState.HighestReceived,
			LastUpdate:      result.WindowState.LastUpdate.UnixNano(),
			Stats: &WindowStats{
				TotalProcessed:  result.WindowState.Stats.TotalProcessed,
				ValidAccepted:   result.WindowState.Stats.ValidAccepted,
				InvalidRejected: result.WindowState.Stats.InvalidRejected,
				WindowSlides:    result.WindowState.Stats.WindowSlides,
				BitmapUpdates:   result.WindowState.Stats.BitmapUpdates,
			},
		}
	}
	
	response := &ValidateSequenceResponse{
		Valid:             result.Valid,
		RejectionReason:   result.RejectionReason,
		WindowState:       windowState,
		ValidationTimeUs:  result.ValidationTimeUs,
	}
	
	return response, nil
}

// Helper functions for type conversion

func convertPathPreferences(prefs *PathPreferences) *paths.PathPreferences {
	if prefs == nil {
		return nil
	}
	
	return &paths.PathPreferences{
		PreferLowLatency:     prefs.PreferLowLatency,
		PreferHighBandwidth:  prefs.PreferHighBandwidth,
		PreferStable:         prefs.PreferStable,
		AvoidASes:            prefs.AvoidAses,
		PreferASes:           prefs.PreferAses,
		MaxRTTMs:             prefs.MaxRttMs,
		MinBandwidthMbps:     float64(prefs.MinBandwidthMbps),
		MaxHops:              prefs.MaxHops,
	}
}

func convertPathInfo(pathInfo *paths.PathInfo) *PathMeta {
	if pathInfo == nil {
		return nil
	}
	
	// Convert hops
	hops := make([]*Hop, len(pathInfo.Hops))
	for i, hop := range pathInfo.Hops {
		hops[i] = &Hop{
			AsId:        hop.IA.String(),
			InterfaceId: uint32(hop.Interface),
			HopType:     hop.HopType,
			Mtu:         uint32(hop.MTU),
		}
	}
	
	// Convert quality
	var quality *PathQuality
	if pathInfo.Quality != nil {
		quality = &PathQuality{
			RttEwmaMs:        float32(pathInfo.Quality.RTT_EWMA.Milliseconds()),
			JitterMs:         float32(pathInfo.Quality.Jitter.Milliseconds()),
			LossRate:         float32(pathInfo.Quality.LossRate),
			BandwidthMbps:    float32(pathInfo.Quality.Bandwidth / (1024 * 1024)),
			StabilityScore:   float32(pathInfo.Quality.Stability),
			LastMeasured:     pathInfo.Quality.LastMeasurement.UnixNano(),
			MeasurementCount: pathInfo.Quality.MeasurementCount,
		}
	}
	
	// Convert usage stats
	usage := &PathUsageStats{
		PacketsSent:     pathInfo.UsageCount, // Approximate
		PacketsReceived: pathInfo.UsageCount, // Approximate
		BytesTransferred: pathInfo.UsageCount * 1024, // Estimate
		SuccessRate:     0.95, // Estimate
		FirstUsed:       pathInfo.LastUsed.UnixNano(),
		UsageDurationS:  0, // Would need to track
	}
	
	return &PathMeta{
		PathId:         pathInfo.ID,
		SrcIa:          pathInfo.SrcIA.String(),
		DstIa:          pathInfo.DstIA.String(),
		ExpiresAt:      pathInfo.Expiry.UnixNano(),
		Hops:           hops,
		Quality:        quality,
		SelectionScore: float32(pathInfo.SelectionScore),
		LastUsed:       pathInfo.LastUsed.UnixNano(),
		Usage:          usage,
	}
}