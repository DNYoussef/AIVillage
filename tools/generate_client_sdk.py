#!/usr/bin/env python3
"""
Client SDK Generator for AIVillage API

This script generates client SDKs in multiple languages from the OpenAPI specification.
Uses openapi-generator-cli to create type-safe client libraries.
"""

import shutil
import subprocess
import sys
from pathlib import Path


class SDKGenerator:
    """Generate client SDKs from OpenAPI specification."""

    def __init__(self, openapi_spec_path: str, output_dir: str = "clients"):
        self.spec_path = Path(openapi_spec_path)
        self.output_dir = Path(output_dir)
        self.project_root = Path(__file__).parent.parent

        if not self.spec_path.exists():
            raise FileNotFoundError(f"OpenAPI spec not found: {self.spec_path}")

    def _check_openapi_generator(self) -> bool:
        """Check if openapi-generator-cli is available."""
        try:
            result = subprocess.run(["openapi-generator-cli", "version"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def _install_openapi_generator(self) -> bool:
        """Install openapi-generator-cli using npm."""
        print("Installing openapi-generator-cli...")
        try:
            # Try npm install
            subprocess.run(["npm", "install", "-g", "@openapitools/openapi-generator-cli"], check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Failed to install via npm. Please install manually:")
            print("npm install -g @openapitools/openapi-generator-cli")
            return False

    def _generate_sdk(
        self,
        language: str,
        output_path: Path,
        additional_properties: dict[str, str] | None = None,
        template_dir: str | None = None,
    ) -> bool:
        """Generate SDK for specific language."""
        print(f"Generating {language} SDK...")

        # Prepare output directory
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            "openapi-generator-cli",
            "generate",
            "-i",
            str(self.spec_path),
            "-g",
            language,
            "-o",
            str(output_path),
        ]

        # Add additional properties
        if additional_properties:
            for key, value in additional_properties.items():
                cmd.extend(["--additional-properties", f"{key}={value}"])

        # Add template directory if specified
        if template_dir:
            cmd.extend(["-t", template_dir])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {language} SDK generated successfully")
                return True
            else:
                print(f"❌ {language} SDK generation failed:")
                print(result.stderr)
                return False
        except Exception as e:
            print(f"❌ Error generating {language} SDK: {e}")
            return False

    def generate_typescript_sdk(self) -> bool:
        """Generate TypeScript/JavaScript SDK."""
        properties = {
            "npmName": "aivillage-client",
            "npmVersion": "1.0.0",
            "npmDescription": "AIVillage API Client",
            "npmAuthor": "AIVillage Team",
            "supportsES6": "true",
            "withInterfaces": "true",
            "enumPropertyNaming": "PascalCase",
        }

        return self._generate_sdk("typescript-axios", self.output_dir / "typescript", properties)

    def generate_python_sdk(self) -> bool:
        """Generate Python SDK."""
        properties = {
            "packageName": "aivillage_client",
            "projectName": "aivillage-client",
            "packageVersion": "1.0.0",
            "packageUrl": "https://github.com/DNYoussef/AIVillage",
            "packageCompany": "AIVillage",
            "packageAuthor": "AIVillage Team",
            "generateSourceCodeOnly": "false",
        }

        return self._generate_sdk("python", self.output_dir / "python", properties)

    def generate_java_sdk(self) -> bool:
        """Generate Java SDK."""
        properties = {
            "groupId": "io.aivillage",
            "artifactId": "aivillage-client",
            "artifactVersion": "1.0.0",
            "artifactDescription": "AIVillage API Client for Java",
            "developerName": "AIVillage Team",
            "developerOrganization": "AIVillage",
            "developerOrganizationUrl": "https://github.com/DNYoussef/AIVillage",
            "licenseName": "MIT",
            "licenseUrl": "https://opensource.org/licenses/MIT",
        }

        return self._generate_sdk("java", self.output_dir / "java", properties)

    def generate_swift_sdk(self) -> bool:
        """Generate Swift SDK for iOS."""
        properties = {
            "projectName": "AIVillageClient",
            "classPrefix": "AV",
            "podVersion": "1.0.0",
            "podDescription": "AIVillage API Client for iOS",
            "podAuthor": "AIVillage Team",
            "podHomepage": "https://github.com/DNYoussef/AIVillage",
            "podLicense": "MIT",
        }

        return self._generate_sdk("swift5", self.output_dir / "swift", properties)

    def generate_kotlin_sdk(self) -> bool:
        """Generate Kotlin SDK for Android."""
        properties = {
            "groupId": "io.aivillage",
            "artifactId": "aivillage-client-kotlin",
            "artifactVersion": "1.0.0",
            "packageName": "io.aivillage.client",
        }

        return self._generate_sdk("kotlin", self.output_dir / "kotlin", properties)

    def generate_go_sdk(self) -> bool:
        """Generate Go SDK."""
        properties = {
            "packageName": "aivillage",
            "packageVersion": "1.0.0",
            "moduleName": "github.com/DNYoussef/AIVillage/clients/go",
        }

        return self._generate_sdk("go", self.output_dir / "go", properties)

    def generate_rust_sdk(self) -> bool:
        """Generate Rust SDK."""
        properties = {
            "packageName": "aivillage-client",
            "packageVersion": "1.0.0",
            "packageAuthors": "AIVillage Team",
            "packageDescription": "AIVillage API Client for Rust",
        }

        return self._generate_sdk("rust", self.output_dir / "rust", properties)

    def generate_all_sdks(self) -> dict[str, bool]:
        """Generate all supported SDKs."""
        generators = {
            "typescript": self.generate_typescript_sdk,
            "python": self.generate_python_sdk,
            "java": self.generate_java_sdk,
            "swift": self.generate_swift_sdk,
            "kotlin": self.generate_kotlin_sdk,
            "go": self.generate_go_sdk,
            "rust": self.generate_rust_sdk,
        }

        results = {}
        for name, generator in generators.items():
            try:
                results[name] = generator()
            except Exception as e:
                print(f"❌ Failed to generate {name} SDK: {e}")
                results[name] = False

        return results

    def create_readme(self, results: dict[str, bool]):
        """Create README file for generated SDKs."""
        readme_content = """# AIVillage API Client SDKs

This directory contains auto-generated client SDKs for the AIVillage API in multiple programming languages.

## Generated SDKs

"""

        for language, success in results.items():
            status = "✅" if success else "❌"
            readme_content += f"- {status} **{language.title()}**: `{language}/`\n"

        readme_content += """
## Usage

Each SDK includes:
- Type-safe API client classes
- Request/response models matching the OpenAPI schema
- Built-in error handling and retry logic
- Authentication support (Bearer token and API key)
- Rate limiting awareness

### Quick Start Examples

#### TypeScript/JavaScript
```typescript
import { AIVillageApi, Configuration } from 'aivillage-client';

const config = new Configuration({
    basePath: 'https://api.aivillage.io/v1',
    accessToken: 'your-api-key'
});

const client = new AIVillageApi(config);
const response = await client.chat({
    message: 'Hello, how can I optimize my mobile app?',
    mode: 'balanced'
});
```

#### Python
```python
import aivillage_client
from aivillage_client.rest import ApiException

configuration = aivillage_client.Configuration(
    host="https://api.aivillage.io/v1",
    access_token="your-api-key"
)

with aivillage_client.ApiClient(configuration) as api_client:
    api_instance = aivillage_client.ChatApi(api_client)

    try:
        response = api_instance.chat({
            'message': 'Hello, how can I optimize my mobile app?',
            'mode': 'balanced'
        })
        print(response)
    except ApiException as e:
        print(f"Exception: {e}")
```

#### Java
```java
import io.aivillage.client.ApiClient;
import io.aivillage.client.ApiException;
import io.aivillage.client.api.ChatApi;
import io.aivillage.client.model.ChatRequest;

ApiClient defaultClient = new ApiClient();
defaultClient.setBasePath("https://api.aivillage.io/v1");
defaultClient.setBearerToken("your-api-key");

ChatApi apiInstance = new ChatApi(defaultClient);
ChatRequest request = new ChatRequest()
    .message("Hello, how can I optimize my mobile app?")
    .mode("balanced");

try {
    ChatResponse result = apiInstance.chat(request);
    System.out.println(result);
} catch (ApiException e) {
    System.err.println("Exception: " + e.getResponseBody());
}
```

## Authentication

All SDKs support two authentication methods:

1. **Bearer Token Authentication**:
   ```
   Authorization: Bearer <your-api-key>
   ```

2. **API Key Header**:
   ```
   x-api-key: <your-api-key>
   ```

## Rate Limiting

All clients automatically handle rate limiting:
- Respect `X-RateLimit-*` headers
- Implement exponential backoff on 429 responses
- Configurable retry policies

## Error Handling

SDKs provide structured error handling:
- HTTP status code errors
- Rate limiting errors (429)
- Validation errors (400)
- Network/timeout errors

## Configuration

Each SDK supports configuration for:
- Base URL/endpoint
- Authentication credentials
- Timeout settings
- Retry policies
- Logging levels

## Generated From

These SDKs are automatically generated from the OpenAPI 3.0 specification:
- **Spec File**: `docs/api/openapi.yaml`
- **Generator**: OpenAPI Generator CLI
- **Version**: Check individual SDK documentation

## Regeneration

To regenerate SDKs after API changes:
```bash
python tools/generate_client_sdk.py
```

## Support

For issues with the SDKs or API:
- [GitHub Issues](https://github.com/DNYoussef/AIVillage/issues)
- [API Documentation](https://docs.aivillage.io)
- [OpenAPI Specification](./docs/api/openapi.yaml)
"""

        readme_path = self.output_dir / "README.md"
        readme_path.parent.mkdir(parents=True, exist_ok=True)
        readme_path.write_text(readme_content)
        print(f"📝 Created README at {readme_path}")

    def validate_openapi_spec(self) -> bool:
        """Validate OpenAPI specification."""
        try:
            import yaml

            with open(self.spec_path) as f:
                spec = yaml.safe_load(f)

            # Basic validation
            required_fields = ["openapi", "info", "paths"]
            for field in required_fields:
                if field not in spec:
                    print(f"❌ Missing required field: {field}")
                    return False

            print("✅ OpenAPI specification is valid")
            return True

        except Exception as e:
            print(f"❌ OpenAPI validation failed: {e}")
            return False


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate AIVillage API client SDKs")
    parser.add_argument("--spec", default="docs/api/openapi.yaml", help="Path to OpenAPI specification file")
    parser.add_argument("--output", default="clients", help="Output directory for generated SDKs")
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=["typescript", "python", "java", "swift", "kotlin", "go", "rust", "all"],
        default=["all"],
        help="Languages to generate (default: all)",
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate OpenAPI spec without generating SDKs"
    )

    args = parser.parse_args()

    # Initialize generator
    try:
        generator = SDKGenerator(args.spec, args.output)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Validate specification
    if not generator.validate_openapi_spec():
        sys.exit(1)

    if args.validate_only:
        print("✅ OpenAPI specification validation complete")
        sys.exit(0)

    # Check and install openapi-generator if needed
    if not generator._check_openapi_generator():
        print("OpenAPI Generator CLI not found.")
        if not generator._install_openapi_generator():
            sys.exit(1)

    # Generate SDKs
    if "all" in args.languages:
        print("Generating all supported SDKs...")
        results = generator.generate_all_sdks()
    else:
        results = {}
        for lang in args.languages:
            method_name = f"generate_{lang}_sdk"
            if hasattr(generator, method_name):
                results[lang] = getattr(generator, method_name)()
            else:
                print(f"❌ Unsupported language: {lang}")
                results[lang] = False

    # Create README
    generator.create_readme(results)

    # Summary
    successful = sum(1 for success in results.values() if success)
    total = len(results)

    print(f"\n📊 Summary: {successful}/{total} SDKs generated successfully")

    if successful < total:
        print("\n❌ Failed SDKs:")
        for lang, success in results.items():
            if not success:
                print(f"  - {lang}")

    print(f"\n🎉 Client SDKs available in: {generator.output_dir}")


if __name__ == "__main__":
    main()
