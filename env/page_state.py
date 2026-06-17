from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


class PageStore:
    def __init__(self, metadata_path: str | Path = "pages/generated_pages/metadata.json") -> None:
        path = Path(metadata_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Page metadata not found: {path}. Run pages/page_generator.py first.")
        self.path = path
        self.pages: dict[str, dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))

    def get(self, page_id: str) -> dict[str, Any]:
        try:
            return self.pages[page_id]
        except KeyError as exc:
            raise KeyError(f"Unknown page_id: {page_id}") from exc

    def find_element(self, page_id: str, element_id: str) -> dict[str, Any] | None:
        page = self.get(page_id)
        for element in page.get("elements", []):
            if element["element_id"] == element_id:
                return element
        return None
