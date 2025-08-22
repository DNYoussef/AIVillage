"""
Cross-Platform Mobile Integration

Unified mobile platform support for:
- Android BitChat service integration
- iOS MultipeerConnectivity networking
- Cross-platform device profiling
- Native mobile optimization bridges
"""

# Platform-specific implementations will be imported dynamically
# based on runtime platform detection

__all__ = [
    "get_platform_manager",
    "AndroidBitChatBridge",
    "IOSBitChatBridge",
    "CrossPlatformProfiler",
]


def get_platform_manager():
    """Get appropriate platform manager based on runtime environment"""
    import platform

    system = platform.system().lower()

    if system == "android" or "android" in platform.platform().lower():
        from .android_bridge import AndroidBitChatBridge

        return AndroidBitChatBridge()
    elif system == "darwin" and "ios" in platform.platform().lower():
        from .ios_bridge import IOSBitChatBridge

        return IOSBitChatBridge()
    else:
        # Desktop/laptop fallback
        from .desktop_bridge import DesktopBridge

        return DesktopBridge()


# Compatibility exports (will be implemented if mobile platform detected)
class AndroidBitChatBridge:
    """Bridge to Android BitChat service - placeholder for non-Android platforms"""

    def __init__(self):
        self.available = False


class IOSBitChatBridge:
    """Bridge to iOS MultipeerConnectivity - placeholder for non-iOS platforms"""

    def __init__(self):
        self.available = False


class CrossPlatformProfiler:
    """Cross-platform device profiling - available on all platforms"""

    def __init__(self):
        self.available = True
