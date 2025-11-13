Commit all changes using the following ruleset:

# Batched Commit Strategy

When committing multiple related changes, group files by functionality and commit them together with descriptive commit messages.

## Commit Message Format

Use current tense with comma-separated changes:

```
change, change, change, change, etc
```

Each change should be:

- Short but descriptive
- In current tense (add, update, remove, etc.)
- Focus on what was changed, not generic descriptions like "update file.py" or "update implementation in file.py"

## Batching Strategy

Group files by logical functionality:

1. **New Features/Operations**: All files related to a single new feature or operation

    - Implementation files
    - Configuration files
    - Documentation
    - Tests (if applicable)

2. **Secondary Operations**: Related secondary pipeline operations and their configs

3. **Core Module Updates**: Changes to existing modules that work together

    - Core logic updates
    - Training/model files
    - Test data updates

4. **Documentation**: Documentation updates separate from code changes

## Example Workflow

```bash
# Get size of changes in each file
git diff --stat

# Get detailed diff for every changed file
git diff

# Stage related files
git add file1.py file2.py config.json

# Commit with descriptive message
git commit -m "add new operation implementation, add operation configuration, update pipeline integration"

# Repeat for other batches
git add doc1.md doc2.md
git commit -m "update operation documentation, update implementation guide"
```

## Best Practices

Verify commits don't break builds before pushing

- Use descriptive action verbs (add, update, remove, refactor, etc.)
- Avoid generic messages like "fix bugs" or "update code"
- Group by feature/functionality, not by file type
- Test commits don't break builds before pushing
