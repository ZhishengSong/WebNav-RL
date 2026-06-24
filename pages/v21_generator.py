from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pages.page_generator import write_json
from pages.v2_generator import (
    COURSES_V2,
    PRODUCTS_V2,
    LayoutContext,
    create_layout,
    page_html,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "pages" / "generated_pages_v21"


def generate_v21_pages(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seed: int = 71,
    train_instances: int = 20,
    eval_instances: int = 5,
) -> tuple[dict[str, Any], list[LayoutContext], dict[str, Any]]:
    if train_instances < 4:
        raise ValueError("V2.1 requires at least four train instances to cover four-item candidate lists")
    if eval_instances < 1:
        raise ValueError("V2.1 requires at least one eval instance")
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {}
    contexts: list[LayoutContext] = []
    all_element_ids: set[str] = set()

    specs = []
    for index in range(train_instances):
        specs.append(
            (
                f"v21_train_{index:02d}",
                "train",
                "list" if index % 2 == 0 else "compact",
                seed + 1000 + index,
                index,
            )
        )
    for index in range(eval_instances):
        specs.append(
            (
                f"v21_eval_{index:02d}",
                "eval",
                "grid",
                seed + 5000 + index,
                train_instances + index,
            )
        )

    for layout_id, split, style, layout_seed, rotation_index in specs:
        pages, context = create_layout(
            layout_id,
            split,
            style,
            layout_seed,
            rotation_index=rotation_index,
        )
        page_overlap = set(metadata).intersection(pages)
        id_overlap = all_element_ids.intersection(context.element_ids)
        if page_overlap:
            raise ValueError(f"Duplicate V2.1 page ids: {sorted(page_overlap)[:5]}")
        if id_overlap:
            raise ValueError(f"Duplicate V2.1 element ids: {sorted(id_overlap)[:5]}")
        metadata.update(pages)
        contexts.append(context)
        all_element_ids.update(context.element_ids)

    styles = {context.layout_id: context.style for context in contexts}
    for page_id, page in metadata.items():
        page["html_path"] = f"pages/generated_pages_v21/{page_id}.html"
        html = page_html(
            page_id.replace("_", " ").title(),
            page["visible_text"],
            page.get("elements", []),
            styles[page["layout_id"]],
        )
        (output_path / f"{page_id}.html").write_text(html, encoding="utf-8")
    write_json(output_path / "metadata.json", metadata)

    train_ids = set().union(*(context.element_ids for context in contexts if context.split == "train"))
    eval_ids = set().union(*(context.element_ids for context in contexts if context.split == "eval"))
    manifest = {
        "version": "v2.1",
        "seed": seed,
        "position_strategy": "cyclic_rotation",
        "train_instances": train_instances,
        "eval_instances": eval_instances,
        "num_pages": len(metadata),
        "num_products": len(PRODUCTS_V2),
        "num_courses": len(COURSES_V2),
        "num_element_ids": len(all_element_ids),
        "train_eval_element_id_overlap": len(train_ids.intersection(eval_ids)),
        "layouts": {
            context.layout_id: {
                "split": context.split,
                "style": context.style,
                "shop_home": context.shop_home,
                "course_home": context.course_home,
                "num_pages": len(context.page_ids),
                "num_element_ids": len(context.element_ids),
            }
            for context in contexts
        },
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata, contexts, manifest
