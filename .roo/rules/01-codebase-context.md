# Codebase Context & Planning

**Always use ScytheContextEngine** for:
- Large tasks and planning work
- Gathering codebase context based on queries
- Unclear or ambiguous user requests
- Preference over multiple file reads (if >4 read_file operations estimated, use context engine instead)

Query the engine with specific, targeted questions about system components, patterns, or architectural decisions.
