from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from pages.page_generator import COURSES, PRODUCTS, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "tasks"


def product_detail_click(product_index: int) -> list[str]:
    return [f"shop_item_{product_index + 1:03d}"]


def product_under_100_click(product_index: int) -> list[str]:
    under_100 = [idx for idx, product in enumerate(PRODUCTS) if product.price < 100]
    return ["filter_price_under_100", f"shop_under_100_item_{under_100.index(product_index) + 1:03d}"]


def product_rating_click(product_index: int) -> list[str]:
    ranked = sorted(range(len(PRODUCTS)), key=lambda idx: PRODUCTS[idx].rating, reverse=True)
    return ["sort_rating_desc", f"shop_rating_item_{ranked.index(product_index) + 1:03d}"]


def course_detail_click(course_index: int) -> list[str]:
    return [f"course_item_{course_index + 1:03d}"]


def course_cs_click(course_index: int) -> list[str]:
    cs_courses = [idx for idx, course in enumerate(COURSES) if course.department == "Computer Science"]
    return ["filter_department_computer_science", f"course_cs_item_{cs_courses.index(course_index) + 1:03d}"]


def course_credits_4_click(course_index: int) -> list[str]:
    four_credit = [idx for idx, course in enumerate(COURSES) if course.credits == 4]
    return ["filter_credits_4", f"course_credits_4_item_{four_credit.index(course_index) + 1:03d}"]


def shopping_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    for idx, product in enumerate(PRODUCTS):
        specs.extend(
            [
                {
                    "instruction": f"Find the item named {product.name}.",
                    "answer": product.name,
                    "clicks": product_detail_click(idx),
                    "difficulty": "easy",
                    "template": "shopping_name",
                },
                {
                    "instruction": f"Find the {product.color} {product.category}.",
                    "answer": product.name,
                    "clicks": product_detail_click(idx),
                    "difficulty": "easy",
                    "template": "shopping_color_category",
                },
                {
                    "instruction": f"Which product costs {product.price} dollars?",
                    "answer": product.name,
                    "clicks": product_detail_click(idx),
                    "difficulty": "easy",
                    "template": "shopping_price_lookup",
                },
            ]
        )

    for category in sorted({product.category for product in PRODUCTS}):
        matches = [idx for idx, product in enumerate(PRODUCTS) if product.category == category]
        highest = max(matches, key=lambda idx: PRODUCTS[idx].rating)
        lowest = min(matches, key=lambda idx: PRODUCTS[idx].price)
        specs.extend(
            [
                {
                    "instruction": f"Find the highest rated {category}.",
                    "answer": PRODUCTS[highest].name,
                    "clicks": product_rating_click(highest),
                    "difficulty": "medium",
                    "template": "shopping_category_highest_rating",
                },
                {
                    "instruction": f"Find the lowest priced {category}.",
                    "answer": PRODUCTS[lowest].name,
                    "clicks": product_detail_click(lowest),
                    "difficulty": "medium",
                    "template": "shopping_category_lowest_price",
                },
            ]
        )

    under_100 = [idx for idx, product in enumerate(PRODUCTS) if product.price < 100]
    highest_under_100 = max(under_100, key=lambda idx: PRODUCTS[idx].rating)
    lowest_under_100 = min(under_100, key=lambda idx: PRODUCTS[idx].price)
    specs.extend(
        [
            {
                "instruction": "Find the highest rated item under 100 dollars.",
                "answer": PRODUCTS[highest_under_100].name,
                "clicks": product_under_100_click(highest_under_100),
                "difficulty": "medium",
                "template": "shopping_under_100_highest_rating",
            },
            {
                "instruction": "Find the lowest priced item under 100 dollars.",
                "answer": PRODUCTS[lowest_under_100].name,
                "clicks": product_under_100_click(lowest_under_100),
                "difficulty": "medium",
                "template": "shopping_under_100_lowest_price",
            },
        ]
    )
    return specs


def course_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    for idx, course in enumerate(COURSES):
        specs.extend(
            [
                {
                    "instruction": f"Find the course with code {course.code}.",
                    "answer": course.code,
                    "clicks": course_detail_click(idx),
                    "difficulty": "easy",
                    "template": "course_code",
                },
                {
                    "instruction": f"Find the course titled {course.title}.",
                    "answer": course.code,
                    "clicks": course_detail_click(idx),
                    "difficulty": "easy",
                    "template": "course_title",
                },
                {
                    "instruction": f"Find the {course.department} course at {course.time}.",
                    "answer": course.code,
                    "clicks": course_cs_click(idx) if course.department == "Computer Science" else course_detail_click(idx),
                    "difficulty": "medium" if course.department == "Computer Science" else "easy",
                    "template": "course_department_time",
                },
            ]
        )

    for department in sorted({course.department for course in COURSES}):
        matches = [idx for idx, course in enumerate(COURSES) if course.department == department]
        highest = max(matches, key=lambda idx: COURSES[idx].rating)
        specs.append(
            {
                "instruction": f"Find the highest rated {department} course.",
                "answer": COURSES[highest].code,
                "clicks": course_cs_click(highest) if department == "Computer Science" else course_detail_click(highest),
                "difficulty": "medium",
                "template": "course_department_highest_rating",
            }
        )

    four_credit = [idx for idx, course in enumerate(COURSES) if course.credits == 4]
    highest_four_credit = max(four_credit, key=lambda idx: COURSES[idx].rating)
    specs.append(
        {
            "instruction": "Find the highest rated 4 credit course.",
            "answer": COURSES[highest_four_credit].code,
            "clicks": course_credits_4_click(highest_four_credit),
            "difficulty": "medium",
            "template": "course_4_credit_highest_rating",
        }
    )
    for idx in four_credit:
        course = COURSES[idx]
        specs.append(
            {
                "instruction": f"Find the 4 credit {course.department} course.",
                "answer": course.code,
                "clicks": course_credits_4_click(idx),
                "difficulty": "medium",
                "template": "course_4_credit_department",
            }
        )
    return specs


def paraphrase_instruction(instruction: str, rng: random.Random) -> str:
    prefixes = ["", "Please ", "Can you ", "In the local page, "]
    suffixes = ["", " Return the exact answer.", " Submit the answer once found.", " Use the page information."]
    prefix = rng.choice(prefixes)
    suffix = rng.choice(suffixes)
    if prefix in {"Can you ", "Please "}:
        instruction = instruction[:1].lower() + instruction[1:]
    if prefix == "Can you ":
        if instruction.endswith("."):
            instruction = instruction[:-1] + "?"
    return f"{prefix}{instruction}{suffix}".strip()


def build_generated_tasks(num_tasks: int = 1000, train_ratio: float = 0.8, seed: int = 7) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    base_specs = shopping_specs() + course_specs()
    tasks: list[dict[str, Any]] = []
    for idx in range(num_tasks):
        spec = rng.choice(base_specs)
        is_shopping = spec["template"].startswith("shopping")
        split = "train" if idx < int(num_tasks * train_ratio) else "eval"
        prefix = "shop" if is_shopping else "course"
        tasks.append(
            {
                "task_id": f"{prefix}_{idx + 1:05d}",
                "start_page": "shop_home" if is_shopping else "course_home",
                "instruction": paraphrase_instruction(spec["instruction"], rng),
                "target_answer": spec["answer"],
                "difficulty": spec["difficulty"],
                "page_type": "shopping" if is_shopping else "course",
                "template": spec["template"],
                "expert_clicks": list(spec["clicks"]),
                "max_steps": 8,
                "split": split,
            }
        )
    return tasks


def write_task_splits(tasks: list[dict[str, Any]]) -> None:
    train_tasks = [task for task in tasks if task["split"] == "train"]
    eval_tasks = [task for task in tasks if task["split"] == "eval"]
    write_jsonl(TASK_DIR / "all_tasks.jsonl", tasks)
    write_jsonl(TASK_DIR / "train_tasks.jsonl", train_tasks)
    write_jsonl(TASK_DIR / "eval_tasks.jsonl", eval_tasks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tasks", type=int, default=1000)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    tasks = build_generated_tasks(args.num_tasks, args.train_ratio, args.seed)
    write_task_splits(tasks)
    print(f"Generated {len(tasks)} tasks: {sum(task['split'] == 'train' for task in tasks)} train, {sum(task['split'] == 'eval' for task in tasks)} eval.")


if __name__ == "__main__":
    main()
