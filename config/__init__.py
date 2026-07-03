"""Pipeline configuration package: YAML loading and validation.

See ``config/default.yaml`` for the documented schema.
"""

from .validate import (  # noqa: F401
    KNOWN_STATISTICS,
    ConfigError,
    PipelineConfig,
    load_config,
    validate,
)
