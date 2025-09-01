"""
Common constants for P2P infrastructure.

Centralizes magic literals to improve code maintainability and reduce connascence.
"""

# Network constants
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0

# Port constants
PORT_MIN = 1
PORT_MAX = 65535
DEFAULT_START_PORT = 8000
DEFAULT_MAX_PORT_ATTEMPTS = 100

# Network test constants
DEFAULT_DNS_SERVER = "8.8.8.8"
DEFAULT_TEST_PORT = 80
LOOPBACK_IP = "127.0.0.1"

# Address validation constants
MAX_HOSTNAME_LENGTH = 253
MAX_DNS_LABEL_LENGTH = 63

# Byte conversion constants
BYTES_PER_KB = 1024
BYTE_UNITS = ["B", "KB", "MB", "GB", "TB", "PB"]

# Time conversion constants
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400
MILLISECONDS_PER_SECOND = 1000

# Identifier constants
MIN_PEER_ID_LENGTH = 8
MAX_PEER_ID_LENGTH = 32
UUID_HEX_LENGTH = 8

# Hash algorithms
SUPPORTED_HASH_ALGORITHMS = {"sha256", "sha1", "md5", "blake2b"}

# Localhost variants
LOCALHOST_ADDRESSES = {"localhost", "127.0.0.1", "::1"}

# Multiaddress protocols
VALID_MULTIADDR_PROTOCOLS = {"ip4", "ip6", "tcp", "udp", "ws", "wss", "p2p"}

# Common peer ID prefixes to normalize
PEER_ID_PREFIXES = {"peer_", "node_", "id_"}

# Default session prefix
DEFAULT_SESSION_PREFIX = "session"
