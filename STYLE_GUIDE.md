# AIVillage Documentation Style Guide

This guide ensures consistency and maintainability across all AIVillage documentation.

## Documentation Principles

### 1. Honesty First
- **Never document aspirational features as complete**
- Clearly distinguish between working, partial, and planned features
- Use status indicators: ✅ Working, 🟡 Partial, ❌ Not Implemented
- Update documentation when implementation status changes

### 2. Accuracy and Verification
- All documentation claims must be verifiable in code
- Include file references for implementation details
- Link to relevant test files for feature verification
- Regular documentation audits against actual implementation

### 3. User-Centric Organization
- Start with what users need to know most
- Provide clear navigation paths
- Include practical examples and usage patterns
- Separate development documentation from user documentation

## File Naming Conventions

### Standard Files
- `README.md` - Overview and quick start (every directory)
- `CONTRIBUTING.md` - Development and contribution guidelines
- `CHANGELOG.md` - Version history and changes
- `LICENSE` - Project licensing information

### Versioned Documentation
Some documentation files use versioning suffixes:
- `filename_1.md` - Version 1 of the document
- Use versioning when maintaining multiple iterations
- Archive old versions rather than delete them

### Directory Structure
```
docs/
├── README.md              # Documentation index
├── architecture/          # System design and architecture
├── api/                   # API documentation and specifications
├── guides/               # User and developer guides
├── components/           # Individual component documentation
├── development/          # Development processes and standards
└── reference/            # Reference materials and matrices
```

## Content Standards

### Status Indicators
Use consistent status indicators throughout documentation:

- ✅ **Complete**: Fully implemented, tested, and production-ready
- 🟡 **Partial**: Implemented but incomplete or missing features
- ❌ **Missing**: Not implemented, stub only, or planned feature
- 🔧 **Maintenance**: Working but needs updates or improvements
- 📋 **Documentation**: Needs documentation updates

### Implementation Evidence
When documenting features, always include:

```markdown
## Feature Name
**Status**: ✅ Complete
**Implementation**: `src/path/to/implementation.py:123`
**Tests**: `tests/path/to/test_feature.py`
**Performance**: Benchmarked at X ms/operation
```

### Cross-References
- Use relative paths for internal links
- Verify all links work before committing
- Include context for external links
- Update links when files are moved or renamed

### Code Examples
```markdown
# Good: Specific, working example
python main.py --mode rag --action query --question "What is AI?"

# Bad: Generic, non-specific example
python main.py [options]
```

## Writing Style

### Tone
- Professional but approachable
- Direct and concise
- Avoid marketing language or hype
- Be specific rather than vague

### Structure
- Use clear headings (H1, H2, H3)
- Include table of contents for long documents
- Use bullet points and numbered lists appropriately
- Break up long paragraphs

### Technical Accuracy
- Include version requirements
- Specify exact commands and paths
- Provide error handling information
- Update when dependencies change

## Review Process

### Before Committing
1. **Link Check**: Verify all internal links work
2. **Accuracy Check**: Verify all technical claims against code
3. **Completeness Check**: Ensure all sections are complete
4. **Formatting Check**: Follow markdown standards

### Regular Maintenance
- Monthly documentation audits
- Update status indicators when implementation changes
- Archive outdated documentation rather than delete
- Maintain accuracy of feature matrices and status documents

## Special Documentation Types

### API Documentation
- Include request/response examples
- Document error conditions
- Specify authentication requirements
- Provide curl examples where applicable

### Architecture Documentation
- Include diagrams when helpful
- Show actual vs. theoretical implementations
- Document design decisions and rationale
- Update when architecture changes

### Status Reports
- Use evidence-based assessments
- Include specific metrics and benchmarks
- Reference actual test results
- Update regularly as implementation progresses

## Quality Assurance

### Automated Checks
- Link integrity verification
- Markdown formatting validation
- Spell checking for published documentation
- Regular audits against implementation

### Manual Reviews
- Peer review for major documentation changes
- User testing of setup instructions
- Regular accuracy validation
- Consistency checks across documents

## Tools and Extensions

### Recommended Tools
- Markdown linters for formatting consistency
- Link checkers for broken reference detection
- Spell checkers for professional appearance
- Diagram tools for architecture documentation

### File Organization
- Keep related files together
- Use clear, descriptive filenames
- Maintain consistent directory structure
- Archive rather than delete historical documentation

---

This style guide ensures AIVillage documentation remains accurate, helpful, and maintainable as the project evolves.

**Last Updated**: August 1, 2025