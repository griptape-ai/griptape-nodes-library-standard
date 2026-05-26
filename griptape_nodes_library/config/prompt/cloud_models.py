"""Shared catalog of Griptape Cloud-backed chat models.

This module is the single source of truth for every node that offers a
Griptape Cloud model dropdown (e.g. the `Agent` node, the `GriptapeCloudPrompt`
node). It mirrors the active `model_type=chat` rows in Griptape Cloud's
ServiceModelConfig table.

The canonical definitions live in `griptape_nodes.drivers.cloud_models` so
they can also be served to the chat sidebar by the engine's `agent_manager`
without the chat sidebar depending on the standard library being loaded.
This module re-exports them so existing consumers keep working unchanged.
When Cloud's catalog changes, edit the engine module and every consumer
picks up the change.
"""

from griptape_nodes.drivers.cloud_models import (
    DEPRECATED_MODELS,
    IMAGE_DEPRECATED_MODELS,
    IMAGE_MODEL_CHOICES,
    IMAGE_MODEL_CHOICES_ARGS,
    MODEL_CHOICES,
    MODEL_CHOICES_ARGS,
    O_SERIES_MODELS,
    VISION_MODEL_CHOICES,
)

__all__ = [
    "DEPRECATED_MODELS",
    "IMAGE_DEPRECATED_MODELS",
    "IMAGE_MODEL_CHOICES",
    "IMAGE_MODEL_CHOICES_ARGS",
    "MODEL_CHOICES",
    "MODEL_CHOICES_ARGS",
    "O_SERIES_MODELS",
    "VISION_MODEL_CHOICES",
]
