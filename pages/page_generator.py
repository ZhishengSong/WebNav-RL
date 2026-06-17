from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = ROOT / "pages" / "generated_pages"
TASK_DIR = ROOT / "tasks"


@dataclass(frozen=True)
class Product:
    name: str
    category: str
    price: int
    rating: float
    color: str


@dataclass(frozen=True)
class Course:
    code: str
    title: str
    department: str
    credits: int
    rating: float
    time: str


PRODUCTS = [
    Product("SoundCore A20", "bluetooth earbuds", 89, 4.8, "black"),
    Product("BassFlow Mini", "bluetooth earbuds", 59, 4.2, "blue"),
    Product("QuietLite Pro", "bluetooth earbuds", 129, 4.7, "white"),
    Product("PixelPad S", "tablet", 299, 4.6, "silver"),
    Product("NoteTab Air", "tablet", 229, 4.4, "gray"),
    Product("RunBeat Clip", "fitness tracker", 69, 4.5, "green"),
    Product("PulseBand Neo", "fitness tracker", 99, 4.7, "black"),
    Product("DeskLamp Halo", "desk lamp", 45, 4.3, "white"),
    Product("FocusLamp Max", "desk lamp", 78, 4.9, "black"),
    Product("TravelMug One", "travel mug", 24, 4.6, "red"),
]

COURSES = [
    Course("CS101", "Intro to Programming", "Computer Science", 3, 4.6, "Mon 09:00"),
    Course("CS220", "Data Structures", "Computer Science", 4, 4.8, "Tue 10:00"),
    Course("CS330", "Web Systems", "Computer Science", 3, 4.5, "Wed 14:00"),
    Course("MATH120", "Linear Algebra", "Mathematics", 4, 4.7, "Thu 09:00"),
    Course("MATH210", "Probability", "Mathematics", 3, 4.4, "Fri 11:00"),
    Course("BIO105", "Cell Biology", "Biology", 3, 4.3, "Mon 13:00"),
    Course("BIO260", "Genetics", "Biology", 4, 4.9, "Wed 10:00"),
    Course("ECON101", "Microeconomics", "Economics", 3, 4.2, "Tue 15:00"),
    Course("ECON240", "Game Theory", "Economics", 3, 4.8, "Thu 14:00"),
    Course("HIST180", "Modern World History", "History", 3, 4.1, "Fri 09:00"),
]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + "\n", encoding="utf-8")


def product_card(product: Product, element_id: str) -> str:
    return (
        f'<article class="card" data-element-id="{element_id}">'
        f"<h2>{product.name}</h2>"
        f"<p>Category: {product.category}</p>"
        f"<p>Price: ${product.price}</p>"
        f"<p>Rating: {product.rating}</p>"
        f"<p>Color: {product.color}</p>"
        f"<button>Open detail</button>"
        "</article>"
    )


def course_card(course: Course, element_id: str) -> str:
    return (
        f'<article class="card" data-element-id="{element_id}">'
        f"<h2>{course.code}: {course.title}</h2>"
        f"<p>Department: {course.department}</p>"
        f"<p>Credits: {course.credits}</p>"
        f"<p>Rating: {course.rating}</p>"
        f"<p>Time: {course.time}</p>"
        f"<button>Open detail</button>"
        "</article>"
    )


def page_html(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 960px; margin: 32px auto; line-height: 1.5; }}
    header {{ border-bottom: 1px solid #ddd; margin-bottom: 20px; }}
    .toolbar {{ display: flex; gap: 12px; margin: 16px 0; flex-wrap: wrap; }}
    .card {{ border: 1px solid #ddd; border-radius: 6px; padding: 12px; margin: 12px 0; }}
    button {{ padding: 6px 10px; }}
  </style>
</head>
<body>
  <header><h1>{title}</h1></header>
  {body}
</body>
</html>
"""


def make_shopping_pages() -> dict[str, dict[str, Any]]:
    pages: dict[str, dict[str, Any]] = {}
    home_id = "shop_home"
    filter_under_100_id = "filter_price_under_100"
    sort_rating_id = "sort_rating_desc"
    elements = [
        {"element_id": filter_under_100_id, "text": "Filter price under 100", "target_page": "shop_under_100"},
        {"element_id": sort_rating_id, "text": "Sort by rating descending", "target_page": "shop_rating_desc"},
    ]
    cards = []
    for idx, product in enumerate(PRODUCTS, start=1):
        detail_page = f"shop_item_{idx:03d}"
        element_id = f"shop_item_{idx:03d}"
        elements.append({"element_id": element_id, "text": product.name, "target_page": detail_page})
        cards.append(product_card(product, element_id))
        pages[detail_page] = {
            "page_id": detail_page,
            "page_type": "shopping",
            "visible_text": (
                f"Product detail. Name: {product.name}. Category: {product.category}. "
                f"Price: ${product.price}. Rating: {product.rating}. Color: {product.color}."
            ),
            "elements": [],
            "answer": product.name,
        }

    body = (
        '<div class="toolbar">'
        f'<button data-element-id="{filter_under_100_id}">Price &lt; 100</button>'
        f'<button data-element-id="{sort_rating_id}">Sort rating high to low</button>'
        "</div>"
        + "\n".join(cards)
    )
    pages[home_id] = {
        "page_id": home_id,
        "page_type": "shopping",
        "visible_text": "Shopping home. " + " ".join(
            f"{p.name}: {p.category}, ${p.price}, rating {p.rating}, {p.color}." for p in PRODUCTS
        ),
        "elements": elements,
    }

    filtered = [p for p in PRODUCTS if p.price < 100]
    pages["shop_under_100"] = {
        "page_id": "shop_under_100",
        "page_type": "shopping",
        "visible_text": "Products with price under 100. " + " ".join(
            f"{p.name}: {p.category}, ${p.price}, rating {p.rating}, {p.color}." for p in filtered
        ),
        "elements": [
            {
                "element_id": f"shop_under_100_item_{idx:03d}",
                "text": product.name,
                "target_page": f"shop_item_{PRODUCTS.index(product) + 1:03d}",
            }
            for idx, product in enumerate(filtered, start=1)
        ],
    }
    sorted_products = sorted(PRODUCTS, key=lambda p: p.rating, reverse=True)
    pages["shop_rating_desc"] = {
        "page_id": "shop_rating_desc",
        "page_type": "shopping",
        "visible_text": "Products sorted by rating high to low. " + " ".join(
            f"{p.name}: {p.category}, ${p.price}, rating {p.rating}, {p.color}." for p in sorted_products
        ),
        "elements": [
            {
                "element_id": f"shop_rating_item_{idx:03d}",
                "text": product.name,
                "target_page": f"shop_item_{PRODUCTS.index(product) + 1:03d}",
            }
            for idx, product in enumerate(sorted_products, start=1)
        ],
    }

    for page_id, meta in pages.items():
        if page_id == home_id:
            html = page_html("Local Shop", body)
        else:
            html = page_html(page_id.replace("_", " ").title(), f"<p>{meta['visible_text']}</p>")
        meta["html_path"] = f"pages/generated_pages/{page_id}.html"
        (GENERATED_DIR / f"{page_id}.html").write_text(html, encoding="utf-8")
    return pages


def make_course_pages() -> dict[str, dict[str, Any]]:
    pages: dict[str, dict[str, Any]] = {}
    home_id = "course_home"
    cs_filter_id = "filter_department_computer_science"
    four_credit_filter_id = "filter_credits_4"
    elements = [
        {"element_id": cs_filter_id, "text": "Filter Computer Science", "target_page": "course_cs"},
        {"element_id": four_credit_filter_id, "text": "Filter 4 credits", "target_page": "course_credits_4"},
    ]
    cards = []
    for idx, course in enumerate(COURSES, start=1):
        detail_page = f"course_detail_{idx:03d}"
        element_id = f"course_item_{idx:03d}"
        elements.append({"element_id": element_id, "text": course.code, "target_page": detail_page})
        cards.append(course_card(course, element_id))
        pages[detail_page] = {
            "page_id": detail_page,
            "page_type": "course",
            "visible_text": (
                f"Course detail. Code: {course.code}. Title: {course.title}. Department: {course.department}. "
                f"Credits: {course.credits}. Rating: {course.rating}. Time: {course.time}."
            ),
            "elements": [],
            "answer": course.code,
        }

    body = (
        '<div class="toolbar">'
        f'<button data-element-id="{cs_filter_id}">Computer Science</button>'
        f'<button data-element-id="{four_credit_filter_id}">4 credits</button>'
        "</div>"
        + "\n".join(cards)
    )
    pages[home_id] = {
        "page_id": home_id,
        "page_type": "course",
        "visible_text": "Course home. " + " ".join(
            f"{c.code} {c.title}: {c.department}, {c.credits} credits, rating {c.rating}, {c.time}." for c in COURSES
        ),
        "elements": elements,
    }
    cs_courses = [c for c in COURSES if c.department == "Computer Science"]
    pages["course_cs"] = {
        "page_id": "course_cs",
        "page_type": "course",
        "visible_text": "Computer Science courses. " + " ".join(
            f"{c.code} {c.title}: {c.credits} credits, rating {c.rating}, {c.time}." for c in cs_courses
        ),
        "elements": [
            {
                "element_id": f"course_cs_item_{idx:03d}",
                "text": course.code,
                "target_page": f"course_detail_{COURSES.index(course) + 1:03d}",
            }
            for idx, course in enumerate(cs_courses, start=1)
        ],
    }
    four_credit = [c for c in COURSES if c.credits == 4]
    pages["course_credits_4"] = {
        "page_id": "course_credits_4",
        "page_type": "course",
        "visible_text": "4 credit courses. " + " ".join(
            f"{c.code} {c.title}: {c.department}, rating {c.rating}, {c.time}." for c in four_credit
        ),
        "elements": [
            {
                "element_id": f"course_credits_4_item_{idx:03d}",
                "text": course.code,
                "target_page": f"course_detail_{COURSES.index(course) + 1:03d}",
            }
            for idx, course in enumerate(four_credit, start=1)
        ],
    }

    for page_id, meta in pages.items():
        if page_id == home_id:
            html = page_html("Course Catalog", body)
        else:
            html = page_html(page_id.replace("_", " ").title(), f"<p>{meta['visible_text']}</p>")
        meta["html_path"] = f"pages/generated_pages/{page_id}.html"
        (GENERATED_DIR / f"{page_id}.html").write_text(html, encoding="utf-8")
    return pages


def build_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    task_id = 1

    shopping_specs = [
        ("Find the bluetooth earbuds under 100 dollars with the highest rating.", "SoundCore A20", ["filter_price_under_100", "shop_under_100_item_001"]),
        ("Find the highest rated desk lamp.", "FocusLamp Max", ["sort_rating_desc", "shop_rating_item_001"]),
        ("Find the lowest priced travel mug.", "TravelMug One", ["shop_item_010"]),
        ("Find the black fitness tracker with the highest rating.", "PulseBand Neo", ["shop_item_007"]),
        ("Find the tablet with the lowest price.", "NoteTab Air", ["shop_item_005"]),
        ("Find the product with rating 4.9.", "FocusLamp Max", ["sort_rating_desc", "shop_rating_item_001"]),
        ("Find the blue bluetooth earbuds.", "BassFlow Mini", ["shop_item_002"]),
        ("Find the white bluetooth earbuds.", "QuietLite Pro", ["shop_item_003"]),
        ("Find the item named RunBeat Clip.", "RunBeat Clip", ["shop_item_006"]),
        ("Find the highest rated item under 100 dollars.", "FocusLamp Max", ["filter_price_under_100", "shop_under_100_item_006"]),
    ]
    for instruction, answer, clicks in shopping_specs:
        tasks.append({
            "task_id": f"shop_{task_id:03d}",
            "start_page": "shop_home",
            "instruction": instruction,
            "target_answer": answer,
            "difficulty": "easy" if len(clicks) == 1 else "medium",
            "page_type": "shopping",
            "expert_clicks": clicks,
            "max_steps": 8,
        })
        task_id += 1

    course_specs = [
        ("Find the highest rated Computer Science course.", "CS220", ["filter_department_computer_science", "course_cs_item_002"]),
        ("Find the 4 credit Biology course.", "BIO260", ["filter_credits_4", "course_credits_4_item_003"]),
        ("Find the course with code ECON240.", "ECON240", ["course_item_009"]),
        ("Find the highest rated 4 credit course.", "BIO260", ["filter_credits_4", "course_credits_4_item_003"]),
        ("Find the Computer Science course at Wed 14:00.", "CS330", ["filter_department_computer_science", "course_cs_item_003"]),
        ("Find the Mathematics course with rating 4.7.", "MATH120", ["course_item_004"]),
        ("Find the course titled Game Theory.", "ECON240", ["course_item_009"]),
        ("Find the course with the lowest rating.", "HIST180", ["course_item_010"]),
        ("Find the 4 credit Computer Science course.", "CS220", ["filter_department_computer_science", "course_cs_item_002"]),
        ("Find the Biology course with the highest rating.", "BIO260", ["course_item_007"]),
    ]
    for instruction, answer, clicks in course_specs:
        tasks.append({
            "task_id": f"course_{task_id:03d}",
            "start_page": "course_home",
            "instruction": instruction,
            "target_answer": answer,
            "difficulty": "easy" if len(clicks) == 1 else "medium",
            "page_type": "course",
            "expert_clicks": clicks,
            "max_steps": 8,
        })
        task_id += 1
    return tasks


def generate(num_tasks: int = 1000, train_ratio: float = 0.8, seed: int = 7) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {}
    metadata.update(make_shopping_pages())
    metadata.update(make_course_pages())
    write_json(GENERATED_DIR / "metadata.json", metadata)
    from tasks.task_generator import build_generated_tasks, write_task_splits

    tasks = build_generated_tasks(num_tasks=num_tasks, train_ratio=train_ratio, seed=seed)
    write_task_splits(tasks)
    train_count = sum(task["split"] == "train" for task in tasks)
    eval_count = len(tasks) - train_count
    print(f"Generated {len(metadata)} pages and {len(tasks)} tasks ({train_count} train, {eval_count} eval).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(GENERATED_DIR), help="Kept for CLI compatibility.")
    parser.add_argument("--num-tasks", type=int, default=1000)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    generate(num_tasks=args.num_tasks, train_ratio=args.train_ratio, seed=args.seed)


if __name__ == "__main__":
    main()
