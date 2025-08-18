// SCION Sidecar - Production Gateway using Official SCION Stack
// Provides gRPC service for Betanet Gateway integration
package main

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/aivillage/scion-sidecar/internal/anti_replay"
	"github.com/aivillage/scion-sidecar/internal/metrics"
	"github.com/aivillage/scion-sidecar/internal/paths"
	"github.com/aivillage/scion-sidecar/internal/scionio"
	"github.com/aivillage/scion-sidecar/pkg/gateway"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
	"net/http"
)

var (
	// Version information
	version = "dev"
	commit  = "unknown"
	date    = "unknown"
)

// SidecarServer implements the Betanet Gateway gRPC service
type SidecarServer struct {
	gateway.UnimplementedBetanetGatewayServer

	// Core SCION components
	scionIO     *scionio.ScionIOHandler
	pathManager *paths.PathManager
	antiReplay  *anti_replay.AntiReplayManager
	metrics     *metrics.MetricsCollector

	// Server state
	ctx    context.Context
	cancel context.CancelFunc
	config *Config
}

// Config holds sidecar configuration
type Config struct {
	// Server configuration
	GRPCAddr      string `mapstructure:"grpc_addr"`
	MetricsAddr   string `mapstructure:"metrics_addr"`
	LogLevel      string `mapstructure:"log_level"`

	// SCION configuration
	SCIONAddr     string `mapstructure:"scion_addr"`
	LocalIA       string `mapstructure:"local_ia"`
	DispatcherAddr string `mapstructure:"dispatcher_addr"`
	SCIONDAddr    string `mapstructure:"sciond_addr"`

	// Path management
	PathRefreshInterval time.Duration `mapstructure:"path_refresh_interval"`
	PathCacheSize       int           `mapstructure:"path_cache_size"`
	MaxPathAge          time.Duration `mapstructure:"max_path_age"`

	// Anti-replay configuration
	ReplayWindowSize   int    `mapstructure:"replay_window_size"`
	ReplayDBPath       string `mapstructure:"replay_db_path"`
	ReplayCleanupTTL   time.Duration `mapstructure:"replay_cleanup_ttl"`

	// Performance tuning
	MaxConcurrentOps   int `mapstructure:"max_concurrent_ops"`
	PacketBufferSize   int `mapstructure:"packet_buffer_size"`
	WorkerPoolSize     int `mapstructure:"worker_pool_size"`
}

// DefaultConfig returns default configuration
func DefaultConfig() *Config {
	return &Config{
		GRPCAddr:            ":8080",
		MetricsAddr:         ":8081",
		LogLevel:            "info",
		SCIONAddr:           "127.0.0.1:30255",
		LocalIA:             "",
		DispatcherAddr:      "127.0.0.1:30041",
		SCIONDAddr:          "127.0.0.1:30255",
		PathRefreshInterval: 5 * time.Second,
		PathCacheSize:       1000,
		MaxPathAge:          30 * time.Minute,
		ReplayWindowSize:    1024,
		ReplayDBPath:        "/tmp/scion_replay.db",
		ReplayCleanupTTL:    1 * time.Hour,
		MaxConcurrentOps:    100,
		PacketBufferSize:    64 * 1024, // 64KB
		WorkerPoolSize:      10,
	}
}

var rootCmd = &cobra.Command{
	Use:   "scion-sidecar",
	Short: "SCION Gateway Sidecar for Betanet Integration",
	Long: `Production SCION sidecar using official SCION libraries.

Provides gRPC service for Betanet Gateway integration with:
- Full SCION packet processing using official stack
- Path discovery and management with caching
- Anti-replay protection with sliding window
- Production telemetry and metrics
- Multi-path failover capabilities`,
	Version: fmt.Sprintf("%s (%s) built at %s", version, commit, date),
	Run:     runSidecar,
}

func init() {
	// Configuration flags
	rootCmd.PersistentFlags().String("config", "", "config file path")
	rootCmd.PersistentFlags().String("grpc-addr", ":8080", "gRPC server address")
	rootCmd.PersistentFlags().String("metrics-addr", ":8081", "metrics server address")
	rootCmd.PersistentFlags().String("log-level", "info", "log level (debug, info, warn, error)")
	rootCmd.PersistentFlags().String("local-ia", "", "local ISD-AS (e.g., 1-ff00:0:110)")
	rootCmd.PersistentFlags().String("scion-addr", "127.0.0.1:30255", "SCION daemon address")

	// Bind flags to viper
	viper.BindPFlag("grpc_addr", rootCmd.PersistentFlags().Lookup("grpc-addr"))
	viper.BindPFlag("metrics_addr", rootCmd.PersistentFlags().Lookup("metrics-addr"))
	viper.BindPFlag("log_level", rootCmd.PersistentFlags().Lookup("log-level"))
	viper.BindPFlag("local_ia", rootCmd.PersistentFlags().Lookup("local-ia"))
	viper.BindPFlag("scion_addr", rootCmd.PersistentFlags().Lookup("scion-addr"))
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		log.WithError(err).Fatal("Failed to execute command")
		os.Exit(1)
	}
}

func runSidecar(cmd *cobra.Command, args []string) {
	// Load configuration
	config, err := loadConfig()
	if err != nil {
		log.WithError(err).Fatal("Failed to load configuration")
	}

	// Setup logging
	setupLogging(config.LogLevel)

	log.WithFields(log.Fields{
		"version": version,
		"commit":  commit,
		"date":    date,
		"config":  config,
	}).Info("Starting SCION sidecar")

	// Create context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Initialize server
	server, err := NewSidecarServer(ctx, config)
	if err != nil {
		log.WithError(err).Fatal("Failed to create sidecar server")
	}

	// Start server
	if err := server.Start(); err != nil {
		log.WithError(err).Fatal("Failed to start sidecar server")
	}

	// Wait for shutdown signal
	waitForShutdown(cancel)

	// Graceful shutdown
	if err := server.Stop(); err != nil {
		log.WithError(err).Error("Error during graceful shutdown")
	}

	log.Info("SCION sidecar stopped")
}

// NewSidecarServer creates a new sidecar server instance
func NewSidecarServer(ctx context.Context, config *Config) (*SidecarServer, error) {
	log.Info("Initializing SCION sidecar components")

	// Initialize metrics collector
	metricsCollector := metrics.NewMetricsCollector()

	// Initialize SCION I/O handler
	scionIO, err := scionio.NewScionIOHandler(ctx, &scionio.Config{
		LocalIA:        config.LocalIA,
		DispatcherAddr: config.DispatcherAddr,
		SCIONDAddr:     config.SCIONDAddr,
		PacketBufferSize: config.PacketBufferSize,
		WorkerPoolSize: config.WorkerPoolSize,
	}, metricsCollector)
	if err != nil {
		return nil, fmt.Errorf("failed to create SCION I/O handler: %w", err)
	}

	// Initialize path manager
	pathManager, err := paths.NewPathManager(ctx, &paths.Config{
		LocalIA:          config.LocalIA,
		SCIONDAddr:       config.SCIONDAddr,
		RefreshInterval:  config.PathRefreshInterval,
		CacheSize:        config.PathCacheSize,
		MaxPathAge:       config.MaxPathAge,
	}, metricsCollector)
	if err != nil {
		return nil, fmt.Errorf("failed to create path manager: %w", err)
	}

	// Initialize anti-replay manager
	antiReplay, err := anti_replay.NewAntiReplayManager(&anti_replay.Config{
		WindowSize:    config.ReplayWindowSize,
		DBPath:       config.ReplayDBPath,
		CleanupTTL:   config.ReplayCleanupTTL,
	}, metricsCollector)
	if err != nil {
		return nil, fmt.Errorf("failed to create anti-replay manager: %w", err)
	}

	server := &SidecarServer{
		scionIO:     scionIO,
		pathManager: pathManager,
		antiReplay:  antiReplay,
		metrics:     metricsCollector,
		ctx:         ctx,
		cancel:      context.CancelFunc(func() {}),
		config:      config,
	}

	log.Info("SCION sidecar components initialized successfully")
	return server, nil
}

// Start starts the sidecar server
func (s *SidecarServer) Start() error {
	log.Info("Starting SCION sidecar server")

	// Start metrics server
	go s.startMetricsServer()

	// Start gRPC server
	if err := s.startGRPCServer(); err != nil {
		return fmt.Errorf("failed to start gRPC server: %w", err)
	}

	return nil
}

// Stop gracefully stops the sidecar server
func (s *SidecarServer) Stop() error {
	log.Info("Stopping SCION sidecar server")

	// Cancel context to stop all operations
	s.cancel()

	// Stop components
	if err := s.antiReplay.Stop(); err != nil {
		log.WithError(err).Error("Failed to stop anti-replay manager")
	}

	if err := s.pathManager.Stop(); err != nil {
		log.WithError(err).Error("Failed to stop path manager")
	}

	if err := s.scionIO.Stop(); err != nil {
		log.WithError(err).Error("Failed to stop SCION I/O handler")
	}

	return nil
}

func (s *SidecarServer) startGRPCServer() error {
	listener, err := net.Listen("tcp", s.config.GRPCAddr)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", s.config.GRPCAddr, err)
	}

	// Create gRPC server with options
	grpcServer := grpc.NewServer(
		grpc.MaxRecvMsgSize(s.config.PacketBufferSize),
		grpc.MaxSendMsgSize(s.config.PacketBufferSize),
	)

	// Create gateway service
	gatewayService := gateway.NewService(s.scionIO, s.pathManager, s.antiReplay, s.metrics)

	// Register services
	gateway.RegisterBetanetGatewayServer(grpcServer, gatewayService)

	// Register health service
	healthServer := health.NewServer()
	healthServer.SetServingStatus("", grpc_health_v1.HealthCheckResponse_SERVING)
	grpc_health_v1.RegisterHealthServer(grpcServer, healthServer)

	// Register reflection for debugging
	reflection.Register(grpcServer)

	log.WithField("address", s.config.GRPCAddr).Info("Starting gRPC server")

	// Start server in goroutine
	go func() {
		if err := grpcServer.Serve(listener); err != nil {
			log.WithError(err).Error("gRPC server error")
		}
	}()

	return nil
}

func (s *SidecarServer) startMetricsServer() {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	server := &http.Server{
		Addr:    s.config.MetricsAddr,
		Handler: mux,
	}

	log.WithField("address", s.config.MetricsAddr).Info("Starting metrics server")

	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.WithError(err).Error("Metrics server error")
	}
}

func loadConfig() (*Config, error) {
	config := DefaultConfig()

	// Check for config file
	if configFile := viper.GetString("config"); configFile != "" {
		viper.SetConfigFile(configFile)
		if err := viper.ReadInConfig(); err != nil {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}
	} else {
		// Search for config in standard locations
		viper.SetConfigName("scion-sidecar")
		viper.SetConfigType("yaml")
		viper.AddConfigPath(".")
		viper.AddConfigPath("/etc/scion-sidecar/")
		viper.AddConfigPath("$HOME/.scion-sidecar/")

		if err := viper.ReadInConfig(); err != nil {
			if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
				return nil, fmt.Errorf("failed to read config: %w", err)
			}
		}
	}

	// Environment variable support
	viper.SetEnvPrefix("SCION_SIDECAR")
	viper.AutomaticEnv()

	// Unmarshal config
	if err := viper.Unmarshal(config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	// Validate required fields
	if config.LocalIA == "" {
		return nil, fmt.Errorf("local_ia is required")
	}

	return config, nil
}

func setupLogging(level string) {
	log.SetFormatter(&log.TextFormatter{
		FullTimestamp: true,
		TimestampFormat: time.RFC3339,
	})

	logLevel, err := log.ParseLevel(level)
	if err != nil {
		log.WithError(err).Warn("Invalid log level, using info")
		logLevel = log.InfoLevel
	}

	log.SetLevel(logLevel)
	log.WithField("level", logLevel).Info("Log level set")
}

func waitForShutdown(cancel context.CancelFunc) {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	<-sigChan
	log.Info("Shutdown signal received")
	cancel()
}
