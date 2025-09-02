// SCION I/O Handler - Production packet processing using official SCION stack
package scionio

import (
	"context"
	"fmt"
	"net"
	"sync"
	"time"

	"github.com/aivillage/scion-sidecar/internal/metrics"
	"github.com/scionproto/scion/pkg/addr"
	"github.com/scionproto/scion/pkg/daemon"
	"github.com/scionproto/scion/pkg/snet"
	"github.com/scionproto/scion/pkg/sock/reliable"
	log "github.com/sirupsen/logrus"
)

// Config holds SCION I/O configuration
type Config struct {
	LocalIA          string        // Local ISD-AS (e.g., "1-ff00:0:110")
	DispatcherAddr   string        // SCION dispatcher address
	SCIONDAddr       string        // SCION daemon address
	PacketBufferSize int           // Maximum packet size
	WorkerPoolSize   int           // Number of worker goroutines
	ReadTimeout      time.Duration // Socket read timeout
	WriteTimeout     time.Duration // Socket write timeout
}

// ScionIOHandler manages SCION packet I/O using official SCION stack
type ScionIOHandler struct {
	config   *Config
	ctx      context.Context
	cancel   context.CancelFunc
	metrics  *metrics.MetricsCollector

	// SCION components
	localIA    addr.IA
	dispatcher reliable.Dispatcher
	daemon     daemon.Connector
	conn       snet.PacketConn

	// Worker management
	workerWG   sync.WaitGroup
	packetChan chan *snet.Packet

	// Statistics
	stats struct {
		sync.RWMutex
		PacketsSent     uint64
		PacketsReceived uint64
		BytesSent       uint64
		BytesReceived   uint64
		Errors          uint64
		LastActivity    time.Time
	}
}

// NewScionIOHandler creates a new SCION I/O handler
func NewScionIOHandler(ctx context.Context, config *Config, metricsCollector *metrics.MetricsCollector) (*ScionIOHandler, error) {
	// Set default values
	if config.ReadTimeout == 0 {
		config.ReadTimeout = 5 * time.Second
	}
	if config.WriteTimeout == 0 {
		config.WriteTimeout = 5 * time.Second
	}
	if config.PacketBufferSize == 0 {
		config.PacketBufferSize = 64 * 1024 // 64KB
	}
	if config.WorkerPoolSize == 0 {
		config.WorkerPoolSize = 10
	}

	// Parse local IA
	localIA, err := addr.ParseIA(config.LocalIA)
	if err != nil {
		return nil, fmt.Errorf("failed to parse local IA %s: %w", config.LocalIA, err)
	}

	ctx, cancel := context.WithCancel(ctx)

	handler := &ScionIOHandler{
		config:     config,
		ctx:        ctx,
		cancel:     cancel,
		metrics:    metricsCollector,
		localIA:    localIA,
		packetChan: make(chan *snet.Packet, config.WorkerPoolSize*2),
	}

	// Initialize SCION components
	if err := handler.initScionComponents(); err != nil {
		cancel()
		return nil, fmt.Errorf("failed to initialize SCION components: %w", err)
	}

	// Start worker goroutines
	handler.startWorkers()

	// Start packet receiver
	go handler.packetReceiver()

	log.WithFields(log.Fields{
		"local_ia":     config.LocalIA,
		"dispatcher":   config.DispatcherAddr,
		"sciond":       config.SCIONDAddr,
		"buffer_size":  config.PacketBufferSize,
		"workers":      config.WorkerPoolSize,
	}).Info("SCION I/O handler initialized")

	return handler, nil
}

// initScionComponents initializes SCION networking components
func (h *ScionIOHandler) initScionComponents() error {
	log.Info("Initializing SCION networking components")

	// Connect to SCION dispatcher
	dispatcher := reliable.NewDispatcher(h.config.DispatcherAddr)

	// Connect to SCION daemon
	daemonConn, err := daemon.NewService(h.config.SCIONDAddr).Connect(h.ctx)
	if err != nil {
		return fmt.Errorf("failed to connect to SCION daemon at %s: %w", h.config.SCIONDAddr, err)
	}

	// Create local address
	localAddr := &snet.UDPAddr{
		IA:   h.localIA,
		Host: &net.UDPAddr{IP: net.IPv4(0, 0, 0, 0), Port: 0}, // Bind to any port
	}

	// Create SCION packet connection using the v0.10.0 API
	// Initialize network with daemon connection
	network := &snet.SCIONNetwork{
		Dispatcher: h.dispatcher,
		Connector:  h.daemon,
	}
	
	// Create packet connection
	conn, err := network.Listen(h.ctx, "udp", localAddr)
	if err != nil {
		return fmt.Errorf("failed to create SCION connection: %w", err)
	}

	h.dispatcher = dispatcher
	h.daemon = daemonConn
	h.conn = conn

	log.WithFields(log.Fields{
		"local_addr": h.conn.LocalAddr(),
		"bound_port": getPortFromAddr(h.conn.LocalAddr()),
	}).Info("SCION networking components initialized")

	return nil
}

// startWorkers starts worker goroutines for packet processing
func (h *ScionIOHandler) startWorkers() {
	log.WithField("workers", h.config.WorkerPoolSize).Info("Starting SCION I/O workers")

	for i := 0; i < h.config.WorkerPoolSize; i++ {
		h.workerWG.Add(1)
		go h.packetWorker(i)
	}
}

// packetWorker processes packets from the channel
func (h *ScionIOHandler) packetWorker(workerID int) {
	defer h.workerWG.Done()

	logger := log.WithField("worker_id", workerID)
	logger.Debug("SCION packet worker started")

	for {
		select {
		case <-h.ctx.Done():
			logger.Debug("SCION packet worker stopping")
			return
		case packet := <-h.packetChan:
			if packet == nil {
				continue
			}

			// Process received packet
			if err := h.processReceivedPacket(packet); err != nil {
				logger.WithError(err).Error("Failed to process received packet")
				h.recordError()
			}
		}
	}
}

// packetReceiver continuously receives packets from SCION network
func (h *ScionIOHandler) packetReceiver() {
	log.Info("Starting SCION packet receiver")
	defer log.Info("SCION packet receiver stopped")

	buffer := make([]byte, h.config.PacketBufferSize)

	for {
		select {
		case <-h.ctx.Done():
			return
		default:
			// Set read deadline
			if err := h.conn.SetReadDeadline(time.Now().Add(h.config.ReadTimeout)); err != nil {
				log.WithError(err).Error("Failed to set read deadline")
				continue
			}

			// Read packet using v0.10.0 API
			packet := &snet.Packet{
				Bytes: buffer,
			}
			err := h.conn.ReadFrom(packet)
			if err != nil {
				if !isTimeoutError(err) && h.ctx.Err() == nil {
					log.WithError(err).Error("Failed to read SCION packet")
					h.recordError()
				}
				continue
			}

			// Queue packet for processing
			select {
			case h.packetChan <- packet:
				h.recordPacketReceived(len(packet.Bytes))
			default:
				log.Warn("Packet channel full, dropping packet")
				h.recordError()
			}
		}
	}
}

// SendPacket sends a SCION packet to the specified destination
func (h *ScionIOHandler) SendPacket(ctx context.Context, rawPacket []byte, dstIA string, dstAddr *net.UDPAddr) error {
	start := time.Now()

	// Parse destination IA
	dstIAAddr, err := addr.ParseIA(dstIA)
	if err != nil {
		return fmt.Errorf("failed to parse destination IA %s: %w", dstIA, err)
	}

	// Create destination address
	destination := &snet.UDPAddr{
		IA:   dstIAAddr,
		Host: dstAddr,
	}

	// Set write deadline
	deadline := time.Now().Add(h.config.WriteTimeout)
	if ctxDeadline, ok := ctx.Deadline(); ok && ctxDeadline.Before(deadline) {
		deadline = ctxDeadline
	}

	if err := h.conn.SetWriteDeadline(deadline); err != nil {
		return fmt.Errorf("failed to set write deadline: %w", err)
	}

	// Send packet using v0.10.0 API - WriteTo now expects a packet pointer
	packet := &snet.Packet{
		Bytes:       rawPacket,
		Destination: destination,
	}
	n, err := h.conn.WriteTo(packet, destination)
	if err != nil {
		h.recordError()
		return fmt.Errorf("failed to send SCION packet: %w", err)
	}

	// Record metrics
	h.recordPacketSent(n)

	// Record timing
	h.metrics.RecordScionPacketSent(time.Since(start), len(rawPacket), dstIA)

	log.WithFields(log.Fields{
		"dst_ia":    dstIA,
		"dst_addr":  dstAddr,
		"bytes":     n,
		"duration":  time.Since(start),
	}).Debug("SCION packet sent")

	return nil
}

// processReceivedPacket processes a received SCION packet
func (h *ScionIOHandler) processReceivedPacket(packet *snet.Packet) error {
	if packet == nil {
		return fmt.Errorf("received nil packet")
	}

	// Extract packet information
	srcIA := packet.Source.IA.String()
	srcAddr := packet.Source.Host
	packetData := packet.Bytes

	log.WithFields(log.Fields{
		"src_ia":   srcIA,
		"src_addr": srcAddr,
		"bytes":    len(packetData),
	}).Debug("Processing received SCION packet")

	// Record metrics
	h.metrics.RecordScionPacketReceived(len(packetData), srcIA)

	// Forward packet to Betanet Gateway via configured callback
	// For now, we just log the packet receipt

	return nil
}

// GetLocalAddr returns the local SCION address
func (h *ScionIOHandler) GetLocalAddr() net.Addr {
	if h.conn == nil {
		return nil
	}
	// LocalAddr is available on the connection
	if localAddr := h.conn.LocalAddr(); localAddr != nil {
		return localAddr
	}
	return nil
}

// GetStats returns current I/O statistics
func (h *ScionIOHandler) GetStats() IOStats {
	h.stats.RLock()
	defer h.stats.RUnlock()

	return IOStats{
		PacketsSent:     h.stats.PacketsSent,
		PacketsReceived: h.stats.PacketsReceived,
		BytesSent:       h.stats.BytesSent,
		BytesReceived:   h.stats.BytesReceived,
		Errors:          h.stats.Errors,
		LastActivity:    h.stats.LastActivity,
		Uptime:          time.Since(h.stats.LastActivity),
	}
}

// Stop gracefully stops the SCION I/O handler
func (h *ScionIOHandler) Stop() error {
	log.Info("Stopping SCION I/O handler")

	// Cancel context to stop all operations
	h.cancel()

	// Close packet channel
	close(h.packetChan)

	// Wait for workers to finish
	h.workerWG.Wait()

	// Close connections
	if h.conn != nil {
		if err := h.conn.Close(); err != nil {
			log.WithError(err).Error("Failed to close SCION connection")
		}
	}

	if h.daemon != nil {
		if err := h.daemon.Close(); err != nil {
			log.WithError(err).Error("Failed to close SCION daemon connection")
		}
	}

	log.Info("SCION I/O handler stopped")
	return nil
}

// Helper methods for statistics tracking

func (h *ScionIOHandler) recordPacketSent(bytes int) {
	h.stats.Lock()
	h.stats.PacketsSent++
	h.stats.BytesSent += uint64(bytes)
	h.stats.LastActivity = time.Now()
	h.stats.Unlock()
}

func (h *ScionIOHandler) recordPacketReceived(bytes int) {
	h.stats.Lock()
	h.stats.PacketsReceived++
	h.stats.BytesReceived += uint64(bytes)
	h.stats.LastActivity = time.Now()
	h.stats.Unlock()
}

func (h *ScionIOHandler) recordError() {
	h.stats.Lock()
	h.stats.Errors++
	h.stats.Unlock()
}

// isTimeoutError checks if error is a timeout error
func isTimeoutError(err error) bool {
	if netErr, ok := err.(net.Error); ok {
		return netErr.Timeout()
	}
	return false
}

// getPortFromAddr extracts port number from network address
func getPortFromAddr(addr net.Addr) int {
	if addr == nil {
		return 0
	}
	if udpAddr, ok := addr.(*snet.UDPAddr); ok {
		if udpAddr.Host != nil {
			if hostUDP, ok := udpAddr.Host.(*net.UDPAddr); ok {
				return hostUDP.Port
			}
		}
	}
	return 0
}

// IOStats represents I/O statistics
type IOStats struct {
	PacketsSent     uint64
	PacketsReceived uint64
	BytesSent       uint64
	BytesReceived   uint64
	Errors          uint64
	LastActivity    time.Time
	Uptime          time.Duration
}
