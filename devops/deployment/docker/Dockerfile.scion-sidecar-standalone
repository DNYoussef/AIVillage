# SCION Sidecar Dockerfile
# Multi-stage build for minimal production image

# Build stage
FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    git \
    make \
    protobuf \
    protobuf-dev \
    gcc \
    musl-dev

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download && go mod verify

# Install protobuf tools
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@latest && \
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Copy source code
COPY . .

# Generate protobuf files
RUN make proto

# Build the binary
ARG VERSION=dev
ARG COMMIT=unknown
ARG DATE=unknown
RUN make build VERSION=${VERSION} COMMIT=${COMMIT} DATE=${DATE}

# Production stage
FROM alpine:3.18

# Install runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    tzdata \
    curl

# Create non-root user
RUN adduser -D -s /bin/sh scion-sidecar

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/bin/scion-sidecar /usr/local/bin/scion-sidecar

# Copy configuration
COPY --from=builder /app/configs/ /etc/scion-sidecar/

# Create data directory
RUN mkdir -p /var/lib/scion-sidecar && \
    chown -R scion-sidecar:scion-sidecar /var/lib/scion-sidecar

# Create log directory
RUN mkdir -p /var/log/scion-sidecar && \
    chown -R scion-sidecar:scion-sidecar /var/log/scion-sidecar

# Switch to non-root user
USER scion-sidecar

# Expose ports
EXPOSE 8080/tcp
EXPOSE 8081/tcp

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/scion-sidecar"]

# Default command
CMD ["--grpc-addr=:8080", "--metrics-addr=:8081", "--log-level=info"]

# Labels
LABEL maintainer="AIVillage Team" \
      version="${VERSION}" \
      description="SCION Sidecar for Betanet Gateway Integration" \
      org.opencontainers.image.title="scion-sidecar" \
      org.opencontainers.image.description="Production SCION gateway sidecar" \
      org.opencontainers.image.vendor="AIVillage" \
      org.opencontainers.image.licenses="MIT"
