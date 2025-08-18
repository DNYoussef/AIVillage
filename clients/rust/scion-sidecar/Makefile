# SCION Sidecar Makefile

# Go parameters
GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get
GOMOD=$(GOCMD) mod
GOFMT=gofmt
GOLINT=golangci-lint

# Build parameters
BINARY_NAME=scion-sidecar
BINARY_UNIX=$(BINARY_NAME)_unix
PROTO_DIR=../proto
PKG_DIR=pkg/gateway

# Version information
VERSION?=dev
COMMIT?=$(shell git rev-parse --short HEAD)
DATE?=$(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
LDFLAGS=-ldflags "-X main.version=$(VERSION) -X main.commit=$(COMMIT) -X main.date=$(DATE)"

.PHONY: all build clean test coverage bench lint format deps proto help

## Build
all: deps proto build ## Build the sidecar binary

build: ## Build the binary
	$(GOBUILD) $(LDFLAGS) -o bin/$(BINARY_NAME) -v ./cmd/scion_sidecar

build-linux: ## Build for Linux
	CGO_ENABLED=1 GOOS=linux GOARCH=amd64 $(GOBUILD) $(LDFLAGS) -o bin/$(BINARY_UNIX) -v ./cmd/scion_sidecar

build-docker: ## Build Docker image
	docker build -t scion-sidecar:$(VERSION) .

## Dependencies
deps: ## Download dependencies
	$(GOMOD) download
	$(GOMOD) tidy
	$(GOMOD) verify

deps-update: ## Update dependencies
	$(GOGET) -u ./...
	$(GOMOD) tidy

## Protobuf
proto: ## Generate protobuf files
	@echo "Generating protobuf files..."
	@mkdir -p $(PKG_DIR)
	protoc --proto_path=$(PROTO_DIR) \
		--go_out=$(PKG_DIR) \
		--go_opt=paths=source_relative \
		--go-grpc_out=$(PKG_DIR) \
		--go-grpc_opt=paths=source_relative \
		$(PROTO_DIR)/betanet_gateway.proto

proto-install: ## Install protobuf tools
	$(GOGET) google.golang.org/protobuf/cmd/protoc-gen-go@latest
	$(GOGET) google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

## Testing
test: ## Run tests
	$(GOTEST) -v -race -coverprofile=coverage.out ./...

test-short: ## Run short tests
	$(GOTEST) -v -short ./...

test-integration: ## Run integration tests
	$(GOTEST) -v -tags=integration ./...

coverage: test ## Generate coverage report
	$(GOCMD) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

bench: ## Run benchmarks
	$(GOTEST) -v -bench=. -benchmem ./...

bench-cpu: ## Run CPU profiling benchmarks
	$(GOTEST) -v -bench=. -benchmem -cpuprofile=cpu.prof ./...

bench-mem: ## Run memory profiling benchmarks
	$(GOTEST) -v -bench=. -benchmem -memprofile=mem.prof ./...

## Code Quality
lint: ## Run linters
	$(GOLINT) run ./...

lint-install: ## Install linting tools
	curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(shell go env GOPATH)/bin v1.54.2

format: ## Format code
	$(GOFMT) -s -w .

format-check: ## Check code formatting
	@test -z $(shell $(GOFMT) -l . | tee /dev/stderr) || (echo "Code is not formatted with gofmt" && exit 1)

vet: ## Run go vet
	$(GOCMD) vet ./...

## Security
security: ## Run security checks
	gosec ./...

security-install: ## Install security tools
	$(GOGET) github.com/securecodewarrior/gosec/v2/cmd/gosec@latest

## Development
run: ## Run the sidecar
	$(GOBUILD) $(LDFLAGS) -o bin/$(BINARY_NAME) ./cmd/scion_sidecar && ./bin/$(BINARY_NAME)

run-dev: ## Run with development config
	$(GOBUILD) $(LDFLAGS) -o bin/$(BINARY_NAME) ./cmd/scion_sidecar && \
	./bin/$(BINARY_NAME) --log-level=debug --local-ia=1-ff00:0:110

debug: ## Run with dlv debugger
	dlv debug ./cmd/scion_sidecar -- --log-level=debug

## Docker
docker-build: ## Build Docker image
	docker build -t scion-sidecar:$(VERSION) .
	docker tag scion-sidecar:$(VERSION) scion-sidecar:latest

docker-run: ## Run Docker container
	docker run -p 8080:8080 -p 8081:8081 scion-sidecar:$(VERSION)

docker-push: ## Push Docker image
	docker push scion-sidecar:$(VERSION)

## Cleanup
clean: ## Clean build artifacts
	$(GOCLEAN)
	rm -rf bin/
	rm -rf coverage.out coverage.html
	rm -rf cpu.prof mem.prof
	rm -rf $(PKG_DIR)/*.pb.go

clean-deps: ## Clean dependencies
	$(GOMOD) clean -cache

## Release
release: clean deps proto test lint build ## Full release build

release-all: clean deps proto test lint build build-linux docker-build ## Release all targets

## Utilities
check-tools: ## Check required tools
	@echo "Checking required tools..."
	@which go >/dev/null 2>&1 || (echo "Go is not installed" && exit 1)
	@which protoc >/dev/null 2>&1 || (echo "Protocol Buffers compiler is not installed" && exit 1)
	@which docker >/dev/null 2>&1 || (echo "Docker is not installed" && exit 1)
	@echo "All required tools are available"

install-tools: proto-install lint-install security-install ## Install all development tools

## Help
help: ## Show this help message
	@echo 'Usage:'
	@echo '  make <target>'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} { \
		if (/^[a-zA-Z_-]+:.*?##.*$$/) printf "  %-20s %s\n", $$1, $$2 \
	}' $(MAKEFILE_LIST)

# Default target
.DEFAULT_GOAL := help
