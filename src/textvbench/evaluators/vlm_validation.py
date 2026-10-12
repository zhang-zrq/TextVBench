"""Shared validation errors for recoverable VLM responses."""

from __future__ import annotations

from typing import Any


class InvalidCandidateSelectionError(ValueError):
    """A structurally valid response containing unusable candidate IDs."""

    def __init__(
        self,
        message: str,
        *,
        recovered_response: dict[str, Any],
        removed_ids: list[str],
    ) -> None:
        super().__init__(message)
        self.recovered_response = recovered_response
        self.removed_ids = removed_ids
