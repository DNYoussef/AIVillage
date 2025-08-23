#!/usr/bin/env python3
"""Secure File Upload Validation System.

Provides comprehensive file upload security including virus scanning,
content validation, size limits, and CODEX-compliant security controls.
"""

import hashlib
import logging
import mimetypes
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import magic

logger = logging.getLogger(__name__)


class FileUploadError(Exception):
    """File upload validation error."""


class MaliciousFileError(Exception):
    """Malicious file detected."""


class SecureFileUploadValidator:
    """Secure file upload validation and processing."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize file upload validator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # File size limits (in bytes)
        self.max_file_size = self.config.get("max_file_size", 10 * 1024 * 1024)  # 10MB
        self.max_total_size = self.config.get("max_total_size", 100 * 1024 * 1024)  # 100MB

        # Allowed file types and extensions
        self.allowed_extensions = set(
            self.config.get(
                "allowed_extensions",
                [
                    "txt",
                    "pdf",
                    "doc",
                    "docx",
                    "xls",
                    "xlsx",
                    "ppt",
                    "pptx",
                    "png",
                    "jpg",
                    "jpeg",
                    "gif",
                    "svg",
                    "bmp",
                    "webp",
                    "mp3",
                    "wav",
                    "ogg",
                    "mp4",
                    "webm",
                    "avi",
                    "csv",
                    "json",
                    "xml",
                    "yaml",
                    "md",
                    "py",
                    "js",
                    "html",
                    "css",
                    "sql",
                ],
            )
        )

        self.allowed_mime_types = set(
            self.config.get(
                "allowed_mime_types",
                [
                    "text/plain",
                    "text/csv",
                    "text/html",
                    "text/css",
                    "application/pdf",
                    "application/json",
                    "application/xml",
                    "application/msword",
                    "application/vnd.ms-excel",
                    "application/vnd.ms-powerpoint",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    "image/png",
                    "image/jpeg",
                    "image/gif",
                    "image/svg+xml",
                    "image/bmp",
                    "image/webp",
                    "audio/mpeg",
                    "audio/wav",
                    "audio/ogg",
                    "video/mp4",
                    "video/webm",
                    "video/x-msvideo",
                    "application/x-python-code",
                    "application/javascript",
                ],
            )
        )

        # Blocked extensions and patterns
        self.blocked_extensions = set(
            self.config.get(
                "blocked_extensions",
                [
                    "exe",
                    "bat",
                    "cmd",
                    "com",
                    "pif",
                    "scr",
                    "vbs",
                    "js",
                    "jar",
                    "msi",
                    "dll",
                    "sys",
                    "drv",
                    "bin",
                    "deb",
                    "rpm",
                    "sh",
                    "bash",
                    "csh",
                    "ksh",
                    "zsh",
                    "ps1",
                    "psm1",
                    "psd1",
                    "ps1xml",
                    "psc1",
                    "psc2",
                    "php",
                    "asp",
                    "aspx",
                    "jsp",
                    "cgi",
                ],
            )
        )

        # Dangerous file patterns
        self.dangerous_patterns = [
            rb"<script[^>]*>",
            rb"javascript:",
            rb"vbscript:",
            rb"onload\s*=",
            rb"onerror\s*=",
            rb"onclick\s*=",
            rb"eval\s*\(",
            rb"document\.cookie",
            rb"document\.write",
            rb"<iframe[^>]*>",
            rb"<object[^>]*>",
            rb"<embed[^>]*>",
            rb"<applet[^>]*>",
        ]

        # Magic number signatures for common file types
        self.magic_signatures = {
            b"\x89PNG\r\n\x1a\n": "image/png",
            b"\xff\xd8\xff": "image/jpeg",
            b"GIF87a": "image/gif",
            b"GIF89a": "image/gif",
            b"%PDF-": "application/pdf",
            b"PK\x03\x04": "application/zip",  # Also used by office files
            b"PK\x05\x06": "application/zip",
            b"PK\x07\x08": "application/zip",
        }

        # Initialize virus scanning if available
        self.virus_scanner = self._init_virus_scanner()

        logger.info("Secure file upload validator initialized")

    def _init_virus_scanner(self) -> Any | None:
        """Initialize virus scanner if available."""
        try:
            # Try to import ClamAV Python bindings
            import pyclamd

            cd = pyclamd.ClamdUnixSocket()
            if cd.ping():
                logger.info("ClamAV virus scanner initialized")
                return cd
        except ImportError:
            logger.debug("ClamAV not available, virus scanning disabled")
        except Exception as e:
            logger.warning(f"Could not initialize virus scanner: {e}")

        return None

    def validate_file(
        self, file_path: str, filename: str | None = None, content: bytes | None = None
    ) -> dict[str, Any]:
        """Validate uploaded file for security issues.

        Args:
            file_path: Path to uploaded file
            filename: Original filename
            content: File content (optional, will read if not provided)

        Returns:
            Validation result dictionary

        Raises:
            FileUploadError: If file is invalid
            MaliciousFileError: If file is potentially malicious
        """
        file_path = Path(file_path)
        filename = filename or file_path.name

        # Read file content if not provided
        if content is None:
            with open(file_path, "rb") as f:
                content = f.read()

        validation_result = {
            "filename": filename,
            "file_path": str(file_path),
            "file_size": len(content),
            "is_safe": True,
            "warnings": [],
            "errors": [],
            "metadata": {},
        }

        try:
            # 1. File size validation
            self._validate_file_size(content, validation_result)

            # 2. Filename validation
            self._validate_filename(filename, validation_result)

            # 3. File extension validation
            self._validate_file_extension(filename, validation_result)

            # 4. MIME type validation
            self._validate_mime_type(content, filename, validation_result)

            # 5. Magic number validation
            self._validate_magic_numbers(content, validation_result)

            # 6. Content analysis
            self._analyze_file_content(content, filename, validation_result)

            # 7. Virus scanning
            if self.virus_scanner:
                self._scan_for_viruses(file_path, validation_result)

            # 8. Archive validation (if applicable)
            if self._is_archive_file(filename):
                self._validate_archive(content, validation_result)

            # 9. Calculate file hash
            validation_result["metadata"]["sha256"] = hashlib.sha256(content).hexdigest()
            validation_result["metadata"]["md5"] = hashlib.md5(
                content, usedforsecurity=False
            ).hexdigest()  # Used for file integrity, not security

            # Determine overall safety
            if validation_result["errors"]:
                validation_result["is_safe"] = False
                error_msg = "; ".join(validation_result["errors"])
                msg = f"File validation failed: {error_msg}"
                raise MaliciousFileError(msg)

            if validation_result["warnings"]:
                logger.warning(
                    f"File validation warnings for {filename}: " f"{'; '.join(validation_result['warnings'])}"
                )

            return validation_result

        except (FileUploadError, MaliciousFileError):
            raise
        except Exception as e:
            logger.exception(f"File validation error for {filename}: {e}")
            msg = f"File validation failed: {e}"
            raise FileUploadError(msg)

    def _validate_file_size(self, content: bytes, result: dict[str, Any]) -> None:
        """Validate file size limits."""
        file_size = len(content)

        if file_size > self.max_file_size:
            result["errors"].append(f"File size {file_size} exceeds maximum allowed {self.max_file_size}")

        if file_size == 0:
            result["errors"].append("File is empty")

        result["metadata"]["file_size"] = file_size

    def _validate_filename(self, filename: str, result: dict[str, Any]) -> None:
        """Validate filename for security issues."""
        # Check for path traversal attempts
        if ".." in filename or "/" in filename or "\\" in filename:
            result["errors"].append("Filename contains path traversal characters")

        # Check for null bytes
        if "\x00" in filename:
            result["errors"].append("Filename contains null bytes")

        # Check for control characters
        if any(ord(c) < 32 for c in filename if c not in "\t\n\r"):
            result["errors"].append("Filename contains control characters")

        # Check filename length
        if len(filename) > 255:
            result["errors"].append("Filename too long")

        # Check for reserved names (Windows)
        reserved_names = [
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
        ]

        basename = filename.split(".")[0].upper()
        if basename in reserved_names:
            result["errors"].append(f"Filename uses reserved name: {basename}")

    def _validate_file_extension(self, filename: str, result: dict[str, Any]) -> None:
        """Validate file extension."""
        # Get file extension
        extension = filename.lower().split(".")[-1] if "." in filename else ""

        # Check if extension is blocked
        if extension in self.blocked_extensions:
            result["errors"].append(f"File extension '{extension}' is not allowed")

        # Check if extension is in allowed list (if specified)
        if self.allowed_extensions and extension not in self.allowed_extensions:
            result["warnings"].append(f"File extension '{extension}' is not in allowed list")

        result["metadata"]["extension"] = extension

    def _validate_mime_type(self, content: bytes, filename: str, result: dict[str, Any]) -> None:
        """Validate MIME type."""
        # Get MIME type from python-magic
        try:
            detected_mime = magic.from_buffer(content, mime=True)
        except Exception:
            detected_mime = "application/octet-stream"

        # Get MIME type from filename
        guessed_mime, _ = mimetypes.guess_type(filename)

        result["metadata"]["detected_mime_type"] = detected_mime
        result["metadata"]["guessed_mime_type"] = guessed_mime

        # Check if detected MIME type is allowed
        if self.allowed_mime_types and detected_mime not in self.allowed_mime_types:
            result["warnings"].append(f"MIME type '{detected_mime}' is not in allowed list")

        # Check for MIME type mismatch
        if guessed_mime and detected_mime != guessed_mime:
            result["warnings"].append(f"MIME type mismatch: detected '{detected_mime}', expected '{guessed_mime}'")

    def _validate_magic_numbers(self, content: bytes, result: dict[str, Any]) -> None:
        """Validate file magic numbers (signatures)."""
        detected_signature = None

        for signature, mime_type in self.magic_signatures.items():
            if content.startswith(signature):
                detected_signature = mime_type
                break

        result["metadata"]["detected_signature"] = detected_signature

        # Compare with detected MIME type
        detected_mime = result["metadata"]["detected_mime_type"]
        if detected_signature and detected_signature != detected_mime:
            result["warnings"].append(
                f"File signature mismatch: signature '{detected_signature}', " f"MIME type '{detected_mime}'"
            )

    def _analyze_file_content(self, content: bytes, filename: str, result: dict[str, Any]) -> None:
        """Analyze file content for dangerous patterns."""
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                result["errors"].append(f"File contains dangerous pattern: {pattern.decode('utf-8', errors='ignore')}")

        # Check for embedded executables in images
        if result["metadata"]["detected_mime_type"].startswith("image/"):
            # Look for PE header (Windows executables)
            if b"MZ" in content and b"PE\x00\x00" in content:
                result["errors"].append("Image file contains embedded executable")

            # Look for ELF header (Linux executables)
            if content.startswith(b"\x7fELF"):
                result["errors"].append("Image file contains ELF executable")

        # Check for macros in office documents
        if any(ext in filename.lower() for ext in [".docm", ".xlsm", ".pptm"]):
            result["warnings"].append("Office document may contain macros")

        # Check for suspicious URLs
        url_pattern = rb'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, content)
        if urls:
            result["metadata"]["embedded_urls"] = [url.decode("utf-8", errors="ignore") for url in urls[:10]]
            if len(urls) > 5:
                result["warnings"].append(f"File contains {len(urls)} URLs")

    def _scan_for_viruses(self, file_path: Path, result: dict[str, Any]) -> None:
        """Scan file for viruses using ClamAV."""
        try:
            scan_result = self.virus_scanner.scan_file(str(file_path))

            if scan_result:
                # File is infected
                virus_name = scan_result[str(file_path)][1] if str(file_path) in scan_result else "Unknown"
                result["errors"].append(f"Virus detected: {virus_name}")

            result["metadata"]["virus_scanned"] = True

        except Exception as e:
            logger.warning(f"Virus scan failed: {e}")
            result["warnings"].append("Virus scan could not be completed")
            result["metadata"]["virus_scanned"] = False

    def _is_archive_file(self, filename: str) -> bool:
        """Check if file is an archive."""
        archive_extensions = [".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"]
        return any(filename.lower().endswith(ext) for ext in archive_extensions)

    def _validate_archive(self, content: bytes, result: dict[str, Any]) -> None:
        """Validate archive files."""
        try:
            # Create temporary file for archive validation
            with tempfile.NamedTemporaryFile() as temp_file:
                temp_file.write(content)
                temp_file.flush()

                # Try to open as ZIP archive
                try:
                    with zipfile.ZipFile(temp_file.name, "r") as zip_file:
                        self._validate_zip_archive(zip_file, result)
                except zipfile.BadZipFile:
                    pass  # Not a ZIP file, that's okay

        except Exception as e:
            result["warnings"].append(f"Archive validation failed: {e}")

    def _validate_zip_archive(self, zip_file: zipfile.ZipFile, result: dict[str, Any]) -> None:
        """Validate ZIP archive contents."""
        file_count = 0
        total_uncompressed_size = 0

        for info in zip_file.filelist:
            file_count += 1
            total_uncompressed_size += info.file_size

            # Check for zip bombs
            if info.compress_size > 0:
                compression_ratio = info.file_size / info.compress_size
                if compression_ratio > 100:  # Suspicious compression ratio
                    result["warnings"].append(f"Suspicious compression ratio in archive: {compression_ratio:.1f}")

            # Check for path traversal in archive
            if ".." in info.filename or info.filename.startswith("/"):
                result["errors"].append(f"Archive contains path traversal: {info.filename}")

            # Check individual file names
            try:
                self._validate_filename(info.filename, {"errors": [], "warnings": []})
            except Exception:
                result["warnings"].append(f"Archive contains file with invalid name: {info.filename}")

        # Check for zip bomb indicators
        if file_count > 1000:
            result["warnings"].append(f"Archive contains many files: {file_count}")

        if total_uncompressed_size > 100 * 1024 * 1024:  # 100MB
            result["warnings"].append(f"Archive uncompressed size is large: {total_uncompressed_size}")

        result["metadata"]["archive_file_count"] = file_count
        result["metadata"]["archive_uncompressed_size"] = total_uncompressed_size

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove path components
        filename = os.path.basename(filename)

        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)

        # Limit length
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = name[: 200 - len(ext)] + ext

        # Ensure it doesn't start with dot or dash
        if filename.startswith((".", "-")):
            filename = "file_" + filename

        return filename

    def get_safe_upload_path(self, base_dir: str, filename: str, create_subdirs: bool = True) -> Path:
        """Get safe path for file upload.

        Args:
            base_dir: Base upload directory
            filename: Original filename
            create_subdirs: Whether to create date-based subdirectories

        Returns:
            Safe file path
        """
        base_path = Path(base_dir)

        # Sanitize filename
        safe_filename = self.sanitize_filename(filename)

        # Create date-based subdirectories if requested
        if create_subdirs:
            from datetime import datetime

            now = datetime.now()
            subdir = base_path / now.strftime("%Y") / now.strftime("%m") / now.strftime("%d")
            subdir.mkdir(parents=True, exist_ok=True)
            base_path = subdir
        else:
            base_path.mkdir(parents=True, exist_ok=True)

        # Handle filename conflicts
        target_path = base_path / safe_filename
        counter = 1
        original_name, extension = os.path.splitext(safe_filename)

        while target_path.exists():
            new_filename = f"{original_name}_{counter}{extension}"
            target_path = base_path / new_filename
            counter += 1

        return target_path


# Example usage
if __name__ == "__main__":
    import tempfile

    # Create test files
    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".txt", delete=False) as test_file:
        test_content = b"This is a test file content."
        test_file.write(test_content)
        test_file_path = test_file.name

    # Initialize validator
    validator = SecureFileUploadValidator()

    try:
        # Validate test file
        result = validator.validate_file(test_file_path, "test_file.txt")
        print("Validation Result:")
        for key, value in result.items():
            print(f"  {key}: {value}")

        # Test filename sanitization
        dangerous_filename = "../../../etc/passwd"
        safe_filename = validator.sanitize_filename(dangerous_filename)
        print("\nFilename sanitization:")
        print(f"  Original: {dangerous_filename}")
        print(f"  Sanitized: {safe_filename}")

    finally:
        # Cleanup
        os.unlink(test_file_path)
