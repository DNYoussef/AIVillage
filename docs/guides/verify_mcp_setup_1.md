# MCP Setup Verification Guide

## Overview
This guide helps verify that the MCP (Model Context Protocol) servers are properly configured and functional in the AIVillage workspace.

## Configuration Files
- **Canonical Config**: `infra/mcp/servers.jsonc` - Contains full configuration with comments
- **Claude Config**: `.roo/mcp.json` - Processed configuration for Claude Code
- **Import Scripts**:
  - `scripts/import_mcp_to_claude.sh` (Unix/Linux/macOS)
  - `scripts/import_mcp_to_claude.ps1` (Windows PowerShell)

## Quick Verification Steps

### 1. Check Configuration Files
```bash
# Verify canonical config exists
ls -la infra/mcp/servers.jsonc

# Verify Claude config exists
ls -la .roo/mcp.json

# Check if configs are in sync
diff <(jq -S . .roo/mcp.json) <(node -e "const fs=require('fs'); const jsonc=fs.readFileSync('infra/mcp/servers.jsonc','utf8'); console.log(JSON.stringify(JSON.parse(jsonc.replace(/\/\/.*$/gm,'').replace(/\/\*[\s\S]*?\*\//g,'').replace(/\${input:[^}]+}/g,'')),null,2))")
```

### 2. Test Import Scripts
```bash
# Test with dry-run (safe)
./scripts/import_mcp_to_claude.sh --dry-run
# or on Windows
.\scripts\import_mcp_to_claude.ps1 -DryRun

# Apply changes
./scripts/import_mcp_to_claude.sh
# or on Windows
.\scripts\import_mcp_to_claude.ps1
```

### 3. Verify Server Installation
```bash
# Check if required packages are available
npx -y @modelcontextprotocol/server-github --help
npx -y @modelcontextprotocol/server-huggingface --help
```

### 4. Test Individual Servers
```bash
# Test GitHub server (requires token)
export GITHUB_PERSONAL_ACCESS_TOKEN="your_token_here"
npx -y @modelcontextprotocol/server-github

# Test Hugging Face server (requires token)
export HF_TOKEN="your_token_here"
npx -y @modelcontextprotocol/server-huggingface

# Test markdown server
export MARKDOWN_ROOT_PATH="./docs"
npx -y @modelcontextprotocol/server-markdown
```

## Server Status Table

| Server | Status | Required Token | Description |
|--------|--------|----------------|-------------|
| GitHub | ✅ Configured | GITHUB_PERSONAL_ACCESS_TOKEN | Repository access and management |
| Hugging Face | ✅ Configured | HF_TOKEN | Model search and information |
| Markdown | ✅ Configured | None | Local markdown documentation |
| Memory | ✅ Configured | None | Persistent knowledge graph |
| Deep Wiki | ✅ Configured | None | Knowledge base management |
| Sequential Thinking | ✅ Configured | None | Multi-step reasoning |
| Context7 | ✅ Configured | None | Code context search |
| Firecrawl | ✅ Configured | FIRECRAWL_API_KEY | Web scraping (restricted domains) |
| Apify | ✅ Configured | APIFY_TOKEN | Web automation (restricted actors) |

## Environment Variables Required

### Required Tokens (for full functionality)
```bash
# GitHub
export GITHUB_PERSONAL_ACCESS_TOKEN="ghp_..."

# Hugging Face
export HF_TOKEN="hf_..."

# Firecrawl (optional)
export FIRECRAWL_API_KEY="fc-..."

# Apify (optional)
export APIFY_TOKEN="apify_api_..."
```

### Directory Configuration
```bash
# Markdown server
export MARKDOWN_ROOT_PATH="./docs"

# Memory server
export MEMORY_FILE_PATH="./.mcp/memory.db"

# Deep Wiki server
export WIKI_ROOT_PATH="./knowledgebase"

# Context7 server
export WORKSPACE_ROOT="."
```

## Troubleshooting

### Common Issues

1. **"No MCP servers configured" in Claude Code**
   - Ensure `.roo/mcp.json` exists and is valid JSON
   - Restart Claude Code after configuration changes
   - Check file permissions

2. **Token placeholders not replaced**
   - The import scripts intentionally clear `${input:...}` placeholders
   - Manually add tokens to `.roo/mcp.json` or set environment variables

3. **Server startup failures**
   - Check Node.js and npm/npx are installed
   - Verify network connectivity for external services
   - Check server-specific error messages

4. **Permission denied errors**
   - Ensure scripts have execute permissions: `chmod +x scripts/import_mcp_to_claude.sh`
   - Check file ownership and permissions

### Debug Commands
```bash
# Check Node.js version
node --version
npm --version

# Test individual server startup
npx -y @modelcontextprotocol/server-github --version

# Validate JSON configuration
node -e "console.log(JSON.stringify(JSON.parse(require('fs').readFileSync('.roo/mcp.json')), null, 2))"

# Check environment variables
env | grep -E "(GITHUB|HF|FIRECRAWL|APIFY)"
```

## Security Notes

- **Never commit tokens** to version control
- Use environment variables or secure vaults for sensitive data
- The `.roo/mcp.json` file should be in `.gitignore`
- Regularly rotate API tokens
- Use least-privilege tokens when possible

## Next Steps

1. Set up required environment variables
2. Test each server individually
3. Configure Claude Code to use the MCP servers
4. Create documentation for your specific use cases
5. Set up monitoring for server health

## Support

For issues with specific MCP servers, refer to:
- [GitHub MCP Server](https://github.com/modelcontextprotocol/servers/tree/main/src/github)
- [Hugging Face MCP Server](https://github.com/modelcontextprotocol/servers/tree/main/src/huggingface)
- [General MCP Documentation](https://modelcontextprotocol.io/)
