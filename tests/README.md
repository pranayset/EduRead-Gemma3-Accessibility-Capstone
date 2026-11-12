# Tests

Add tests as reusable source modules are implemented. Prefer deterministic CPU tests that do not download a model or require a GPU.

Suggested coverage:
- Manifest schema and required-field validation.
- Image path and image-format validation.
- Task-type and prompt validation.
- Target text and label formatting.
- Inference input validation and output schema.
- Mathematical helper functions, where applicable.

Run the test suite with:

```bash
python -m pytest
```

The current repository may contain no executable tests yet. Do not interpret an empty test suite as evidence of model quality or successful end-to-end inference. Document GPU/model-dependent smoke tests separately.
