from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from pages.page_generator import write_jsonl
from pages.v2_generator import COURSES_V2, PRODUCTS_V2, LayoutContext


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_DIR = ROOT / "tasks" / "v2"


def shopping_specs(context: LayoutContext) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for product in PRODUCTS_V2:
        specs.extend(
            [
                {
                    "goal": f"Find the item named {product.name}.",
                    "answer": product.name,
                    "clicks": context.shop_direct[product.name],
                    "difficulty": "easy",
                    "template": "v2_shopping_name",
                },
                {
                    "goal": f"Find the {product.color} {product.category}.",
                    "answer": product.name,
                    "clicks": context.shop_category[(product.category, product.name)],
                    "difficulty": "medium",
                    "template": "v2_shopping_color_category",
                },
                {
                    "goal": f"Which product costs {product.price} dollars?",
                    "answer": product.name,
                    "clicks": context.shop_direct[product.name],
                    "difficulty": "easy",
                    "template": "v2_shopping_price_lookup",
                },
            ]
        )

    for category in sorted({product.category for product in PRODUCTS_V2}):
        matches = [product for product in PRODUCTS_V2 if product.category == category]
        highest = max(matches, key=lambda product: (product.rating, product.name))
        lowest = min(matches, key=lambda product: (product.price, product.name))
        specs.extend(
            [
                {
                    "goal": f"Filter to {category} and find the highest rated candidate.",
                    "answer": highest.name,
                    "clicks": context.shop_category[(category, highest.name)],
                    "difficulty": "hard",
                    "template": "v2_shopping_category_highest_rating",
                },
                {
                    "goal": f"Filter to {category} and find the lowest priced candidate.",
                    "answer": lowest.name,
                    "clicks": context.shop_category[(category, lowest.name)],
                    "difficulty": "hard",
                    "template": "v2_shopping_category_lowest_price",
                },
            ]
        )
        under_100 = [product for product in matches if product.price < 100]
        if len(under_100) >= 2:
            best_budget = max(under_100, key=lambda product: (product.rating, product.name))
            specs.append(
                {
                    "goal": f"Among {category} products under 100 dollars, find the highest rated one.",
                    "answer": best_budget.name,
                    "clicks": context.shop_category[(category, best_budget.name)],
                    "difficulty": "hard",
                    "template": "v2_shopping_category_budget_highest_rating",
                }
            )

    under_100 = [product for product in PRODUCTS_V2 if product.price < 100]
    highest_under_100 = max(under_100, key=lambda product: (product.rating, product.name))
    lowest_under_100 = min(under_100, key=lambda product: (product.price, product.name))
    specs.extend(
        [
            {
                "goal": "Filter to products under 100 dollars and find the highest rated candidate.",
                "answer": highest_under_100.name,
                "clicks": context.shop_under_100[highest_under_100.name],
                "difficulty": "hard",
                "template": "v2_shopping_under_100_highest_rating",
            },
            {
                "goal": "Filter to products under 100 dollars and find the lowest priced candidate.",
                "answer": lowest_under_100.name,
                "clicks": context.shop_under_100[lowest_under_100.name],
                "difficulty": "hard",
                "template": "v2_shopping_under_100_lowest_price",
            },
        ]
    )
    return specs


def course_specs(context: LayoutContext) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for course in COURSES_V2:
        specs.extend(
            [
                {
                    "goal": f"Find the course with code {course.code}.",
                    "answer": course.code,
                    "clicks": context.course_direct[course.code],
                    "difficulty": "easy",
                    "template": "v2_course_code",
                },
                {
                    "goal": f"Find the course titled {course.title}.",
                    "answer": course.code,
                    "clicks": context.course_direct[course.code],
                    "difficulty": "easy",
                    "template": "v2_course_title",
                },
                {
                    "goal": f"Filter to {course.department} and find the course at {course.time}.",
                    "answer": course.code,
                    "clicks": context.course_department[(course.department, course.code)],
                    "difficulty": "medium",
                    "template": "v2_course_department_time",
                },
            ]
        )

    departments = sorted({course.department for course in COURSES_V2})
    for department in departments:
        matches = [course for course in COURSES_V2 if course.department == department]
        highest = max(matches, key=lambda course: (course.rating, course.code))
        specs.append(
            {
                "goal": f"Filter to {department} and find the highest rated course.",
                "answer": highest.code,
                "clicks": context.course_department[(department, highest.code)],
                "difficulty": "hard",
                "template": "v2_course_department_highest_rating",
            }
        )
        for credits in sorted({course.credits for course in matches}):
            credit_matches = [course for course in matches if course.credits == credits]
            highest_credit = max(credit_matches, key=lambda course: (course.rating, course.code))
            specs.append(
                {
                    "goal": (
                        f"Within {department}, find the highest rated course that has {credits} credits."
                    ),
                    "answer": highest_credit.code,
                    "clicks": context.course_department[(department, highest_credit.code)],
                    "difficulty": "hard",
                    "template": "v2_course_department_credits_highest_rating",
                }
            )

    for credits in sorted({course.credits for course in COURSES_V2}):
        matches = [course for course in COURSES_V2 if course.credits == credits]
        highest = max(matches, key=lambda course: (course.rating, course.code))
        specs.append(
            {
                "goal": f"Filter to {credits} credit courses and find the highest rated candidate.",
                "answer": highest.code,
                "clicks": context.course_credits[(credits, highest.code)],
                "difficulty": "hard",
                "template": "v2_course_credits_highest_rating",
            }
        )
        for department in sorted({course.department for course in matches}):
            department_matches = [course for course in matches if course.department == department]
            target = max(department_matches, key=lambda course: (course.rating, course.code))
            specs.append(
                {
                    "goal": (
                        f"Filter to {credits} credit courses and find the highest rated {department} candidate."
                    ),
                    "answer": target.code,
                    "clicks": context.course_credits[(credits, target.code)],
                    "difficulty": "hard",
                    "template": "v2_course_credits_department",
                }
            )
    return specs


def paraphrase(goal: str, start_page: str, rng: random.Random) -> str:
    openers = [
        "",
        "Please ",
        "Using the visible page information, ",
        "Navigate carefully and ",
    ]
    endings = [
        " Return the exact answer.",
        " Submit the answer after opening the matching detail.",
        " Use the clickable element IDs shown in observations.",
        "",
    ]
    opener = rng.choice(openers)
    if opener:
        goal = goal[:1].lower() + goal[1:]
    return f"Start from page {start_page}. {opener}{goal}{rng.choice(endings)}".strip()


def balanced_tasks(
    count: int,
    contexts: list[LayoutContext],
    split: str,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pools: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for context in contexts:
        by_template: dict[str, list[dict[str, Any]]] = {}
        for spec in shopping_specs(context) + course_specs(context):
            by_template.setdefault(spec["template"], []).append(spec)
        pools[context.layout_id] = by_template
    template_order = sorted(next(iter(pools.values())))
    rng.shuffle(template_order)
    orders = {
        (layout_id, template): list(range(len(specs)))
        for layout_id, templates in pools.items()
        for template, specs in templates.items()
    }
    for order in orders.values():
        rng.shuffle(order)

    tasks = []
    positions: Counter[tuple[str, str]] = Counter()
    for index in range(count):
        context = contexts[index % len(contexts)]
        template = template_order[index % len(template_order)]
        pool = pools[context.layout_id][template]
        key = (context.layout_id, template)
        order = orders[key]
        position = positions[key]
        if position > 0 and position % len(order) == 0:
            rng.shuffle(order)
        spec = pool[order[position % len(order)]]
        positions[key] += 1
        is_shopping = spec["template"].startswith("v2_shopping")
        start_page = context.shop_home if is_shopping else context.course_home
        tasks.append(
            {
                "task_id": f"v2_{split}_{index + 1:05d}",
                "start_page": start_page,
                "instruction": paraphrase(spec["goal"], start_page, rng),
                "target_answer": spec["answer"],
                "difficulty": spec["difficulty"],
                "page_type": "shopping" if is_shopping else "course",
                "template": spec["template"],
                "expert_clicks": list(spec["clicks"]),
                "max_steps": 8,
                "split": split,
                "layout_id": context.layout_id,
                "structure_split": "seen_layout" if split == "train" else "held_out_layout",
                "dataset_version": "v2",
            }
        )
    return tasks


def generate_v2_tasks(
    contexts: list[LayoutContext],
    train_count: int = 3000,
    eval_count: int = 500,
    seed: int = 37,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_contexts = [context for context in contexts if context.split == "train"]
    eval_contexts = [context for context in contexts if context.split == "eval"]
    if not train_contexts or not eval_contexts:
        raise ValueError("V2 generation requires both train and eval layout contexts")
    train_tasks = balanced_tasks(train_count, train_contexts, "train", seed + 1)
    eval_tasks = balanced_tasks(eval_count, eval_contexts, "eval", seed + 2)

    train_ids = {click for task in train_tasks for click in task["expert_clicks"]}
    eval_ids = {click for task in eval_tasks for click in task["expert_clicks"]}
    manifest = {
        "version": "v2",
        "seed": seed,
        "train_tasks": len(train_tasks),
        "eval_tasks": len(eval_tasks),
        "train_layouts": sorted({task["layout_id"] for task in train_tasks}),
        "eval_layouts": sorted({task["layout_id"] for task in eval_tasks}),
        "expert_element_id_overlap": len(train_ids.intersection(eval_ids)),
        "train_template_counts": dict(Counter(task["template"] for task in train_tasks)),
        "eval_template_counts": dict(Counter(task["template"] for task in eval_tasks)),
        "difficulty_counts": dict(Counter(task["difficulty"] for task in train_tasks + eval_tasks)),
    }
    return train_tasks, eval_tasks, manifest


def write_v2_tasks(
    train_tasks: list[dict[str, Any]],
    eval_tasks: list[dict[str, Any]],
    manifest: dict[str, Any],
    output_dir: str | Path = DEFAULT_TASK_DIR,
) -> None:
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path / "train_tasks.jsonl", train_tasks)
    write_jsonl(output_path / "eval_tasks.jsonl", eval_tasks)
    write_jsonl(output_path / "all_tasks.jsonl", train_tasks + eval_tasks)
    (output_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
