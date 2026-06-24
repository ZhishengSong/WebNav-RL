from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Iterable, TypeVar

from pages.page_generator import Course, Product, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "pages" / "generated_pages_v2"


PRODUCTS_V2 = [
    Product("SoundCore A20", "bluetooth earbuds", 89, 4.8, "black"),
    Product("BassFlow Mini", "bluetooth earbuds", 59, 4.2, "blue"),
    Product("QuietLite Pro", "bluetooth earbuds", 129, 4.7, "white"),
    Product("EchoBud Flex", "bluetooth earbuds", 74, 4.4, "green"),
    Product("PixelPad S", "tablet", 299, 4.6, "silver"),
    Product("NoteTab Air", "tablet", 229, 4.4, "gray"),
    Product("SlateGo 11", "tablet", 349, 4.8, "black"),
    Product("MiniCanvas", "tablet", 199, 4.1, "blue"),
    Product("RunBeat Clip", "fitness tracker", 69, 4.5, "green"),
    Product("PulseBand Neo", "fitness tracker", 99, 4.7, "black"),
    Product("MoveSense Lite", "fitness tracker", 49, 4.0, "white"),
    Product("TrailMetric X", "fitness tracker", 139, 4.9, "orange"),
    Product("DeskLamp Halo", "desk lamp", 45, 4.3, "white"),
    Product("FocusLamp Max", "desk lamp", 78, 4.9, "black"),
    Product("GlowArc Mini", "desk lamp", 36, 4.1, "yellow"),
    Product("StudioBeam", "desk lamp", 112, 4.6, "silver"),
    Product("TravelMug One", "travel mug", 24, 4.6, "red"),
    Product("ThermoSip Pro", "travel mug", 39, 4.8, "black"),
    Product("Commuter Cup", "travel mug", 19, 4.0, "blue"),
    Product("TrailTumbler", "travel mug", 54, 4.4, "green"),
    Product("KeyType 75", "mechanical keyboard", 119, 4.7, "gray"),
    Product("QuietKeys Mini", "mechanical keyboard", 84, 4.3, "white"),
    Product("SwitchBoard Pro", "mechanical keyboard", 149, 4.9, "black"),
    Product("CodeKeys Lite", "mechanical keyboard", 67, 4.1, "blue"),
]


COURSES_V2 = [
    Course("CS101", "Intro to Programming", "Computer Science", 3, 4.6, "Mon 09:00"),
    Course("CS220", "Data Structures", "Computer Science", 4, 4.8, "Tue 10:00"),
    Course("CS330", "Web Systems", "Computer Science", 3, 4.5, "Wed 14:00"),
    Course("CS410", "Distributed Systems", "Computer Science", 4, 5.0, "Thu 16:00"),
    Course("MATH120", "Linear Algebra", "Mathematics", 4, 4.7, "Thu 09:00"),
    Course("MATH210", "Probability", "Mathematics", 3, 4.4, "Fri 11:00"),
    Course("MATH250", "Discrete Mathematics", "Mathematics", 3, 4.6, "Tue 13:00"),
    Course("MATH340", "Numerical Methods", "Mathematics", 4, 4.8, "Wed 15:00"),
    Course("BIO105", "Cell Biology", "Biology", 3, 4.3, "Mon 13:00"),
    Course("BIO260", "Genetics", "Biology", 4, 4.9, "Wed 10:00"),
    Course("BIO310", "Ecology", "Biology", 3, 4.5, "Thu 11:00"),
    Course("BIO355", "Molecular Evolution", "Biology", 4, 4.7, "Fri 14:00"),
    Course("ECON101", "Microeconomics", "Economics", 3, 4.2, "Tue 15:00"),
    Course("ECON240", "Game Theory", "Economics", 3, 4.8, "Thu 14:00"),
    Course("ECON315", "Labor Economics", "Economics", 4, 4.5, "Mon 16:00"),
    Course("ECON360", "Behavioral Economics", "Economics", 4, 4.7, "Wed 09:00"),
    Course("HIST180", "Modern World History", "History", 3, 4.1, "Fri 09:00"),
    Course("HIST230", "History of Science", "History", 4, 4.6, "Tue 11:00"),
    Course("HIST305", "Global Empires", "History", 3, 4.4, "Wed 13:00"),
    Course("HIST350", "Digital History", "History", 4, 4.8, "Thu 15:00"),
    Course("ART110", "Visual Design", "Art", 3, 4.5, "Mon 10:00"),
    Course("ART205", "Photography", "Art", 4, 4.7, "Tue 14:00"),
    Course("ART280", "Interactive Media", "Art", 3, 4.9, "Wed 16:00"),
    Course("ART330", "Design Research", "Art", 4, 4.6, "Fri 10:00"),
]


T = TypeVar("T")


class ElementIdFactory:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.ids: set[str] = set()

    def new(self) -> str:
        alphabet = string.ascii_lowercase + string.digits
        while True:
            value = "el_" + "".join(self.rng.choice(alphabet) for _ in range(10))
            if value not in self.ids:
                self.ids.add(value)
                return value


@dataclass
class LayoutContext:
    layout_id: str
    split: str
    style: str
    shop_home: str
    course_home: str
    shop_direct: dict[str, list[str]] = field(default_factory=dict)
    shop_category: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    shop_under_100: dict[str, list[str]] = field(default_factory=dict)
    shop_rating: dict[str, list[str]] = field(default_factory=dict)
    course_direct: dict[str, list[str]] = field(default_factory=dict)
    course_department: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    course_credits: dict[tuple[int, str], list[str]] = field(default_factory=dict)
    element_ids: set[str] = field(default_factory=set)
    page_ids: set[str] = field(default_factory=set)


def ordered(items: Iterable[T], rng: random.Random, style: str) -> list[T]:
    values = list(items)
    if style == "list":
        return values
    if style == "compact":
        return list(reversed(values))
    rng.shuffle(values)
    return values


def product_text(product: Product, element_id: str, style: str) -> str:
    if style == "list":
        return (
            f"[element_id={element_id}] {product.name}; category {product.category}; "
            f"price ${product.price}; rating {product.rating}; color {product.color}."
        )
    if style == "compact":
        return (
            f"ID {element_id} | {product.name} | {product.color} {product.category} | "
            f"${product.price} | {product.rating} stars."
        )
    return (
        f"CARD({element_id}) name={product.name}, color={product.color}, type={product.category}, "
        f"rating={product.rating}, cost=${product.price}."
    )


def course_text(course: Course, element_id: str, style: str) -> str:
    if style == "list":
        return (
            f"[element_id={element_id}] {course.code} {course.title}; department {course.department}; "
            f"{course.credits} credits; rating {course.rating}; time {course.time}."
        )
    if style == "compact":
        return (
            f"ID {element_id} | {course.code}: {course.title} | {course.department} | "
            f"{course.credits} cr | {course.rating} stars | {course.time}."
        )
    return (
        f"ROW({element_id}) code={course.code}, title={course.title}, dept={course.department}, "
        f"credits={course.credits}, score={course.rating}, schedule={course.time}."
    )


def page_html(title: str, visible_text: str, elements: list[dict[str, Any]], style: str) -> str:
    controls = "".join(
        f'<button data-element-id="{escape(item["element_id"])}">{escape(item["text"])}</button>'
        for item in elements
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 24px auto; line-height: 1.45; }}
    main {{ display: {'grid' if style == 'grid' else 'block'}; gap: 10px; }}
    button {{ margin: 4px; padding: 7px 10px; }}
    .observation {{ white-space: pre-wrap; border-top: 1px solid #bbb; padding-top: 14px; }}
  </style>
</head>
<body>
  <header><h1>{escape(title)}</h1></header>
  <main>{controls}</main>
  <p class="observation">{escape(visible_text)}</p>
</body>
</html>
"""


def control_text(element_id: str, label: str, style: str) -> str:
    if style == "grid":
        return f"ACTION({element_id})={label}"
    return f"[element_id={element_id}] {label}"


def create_layout(layout_id: str, split: str, style: str, seed: int) -> tuple[dict[str, Any], LayoutContext]:
    rng = random.Random(seed)
    ids = ElementIdFactory(rng)
    namespace = f"v2_{layout_id}"
    shop_home = f"{namespace}_shop_home"
    course_home = f"{namespace}_course_home"
    context = LayoutContext(layout_id, split, style, shop_home, course_home)
    pages: dict[str, dict[str, Any]] = {}

    product_details: dict[str, str] = {}
    for index, product in enumerate(PRODUCTS_V2):
        page_id = f"{namespace}_product_{index:03d}"
        product_details[product.name] = page_id
        pages[page_id] = {
            "page_id": page_id,
            "page_type": "shopping",
            "layout_id": layout_id,
            "split": split,
            "visible_text": (
                f"Product detail. Name: {product.name}. Category: {product.category}. Price: ${product.price}. "
                f"Rating: {product.rating}. Color: {product.color}."
            ),
            "elements": [],
            "answer": product.name,
        }

    category_pages: dict[str, tuple[str, str]] = {}
    for category in sorted({product.category for product in PRODUCTS_V2}):
        page_id = f"{namespace}_shop_category_{len(category_pages):02d}"
        filter_id = ids.new()
        category_pages[category] = (page_id, filter_id)
        elements = []
        texts = []
        matches = ordered((p for p in PRODUCTS_V2 if p.category == category), rng, style)
        for product in matches:
            element_id = ids.new()
            elements.append({"element_id": element_id, "text": product.name, "target_page": product_details[product.name]})
            texts.append(product_text(product, element_id, style))
            context.shop_category[(category, product.name)] = [filter_id, element_id]
        pages[page_id] = {
            "page_id": page_id,
            "page_type": "shopping",
            "layout_id": layout_id,
            "split": split,
            "visible_text": f"Filtered category: {category}. " + " ".join(texts),
            "elements": elements,
        }

    under_page = f"{namespace}_shop_under_100"
    under_filter_id = ids.new()
    under_elements = []
    under_texts = []
    for product in ordered((p for p in PRODUCTS_V2 if p.price < 100), rng, style):
        element_id = ids.new()
        under_elements.append({"element_id": element_id, "text": product.name, "target_page": product_details[product.name]})
        under_texts.append(product_text(product, element_id, style))
        context.shop_under_100[product.name] = [under_filter_id, element_id]
    pages[under_page] = {
        "page_id": under_page,
        "page_type": "shopping",
        "layout_id": layout_id,
        "split": split,
        "visible_text": "Products under $100. " + " ".join(under_texts),
        "elements": under_elements,
    }

    rating_page = f"{namespace}_shop_rating"
    rating_filter_id = ids.new()
    rating_elements = []
    rating_texts = []
    ranked_products = sorted(PRODUCTS_V2, key=lambda product: (-product.rating, product.name))
    for product in ranked_products:
        element_id = ids.new()
        rating_elements.append({"element_id": element_id, "text": product.name, "target_page": product_details[product.name]})
        rating_texts.append(product_text(product, element_id, style))
        context.shop_rating[product.name] = [rating_filter_id, element_id]
    pages[rating_page] = {
        "page_id": rating_page,
        "page_type": "shopping",
        "layout_id": layout_id,
        "split": split,
        "visible_text": "Products sorted by rating descending. " + " ".join(rating_texts),
        "elements": rating_elements,
    }

    shop_elements = []
    shop_controls = []
    for category, (page_id, filter_id) in category_pages.items():
        shop_elements.append({"element_id": filter_id, "text": f"Filter category {category}", "target_page": page_id})
        shop_controls.append(control_text(filter_id, f"Filter category {category}", style))
    shop_elements.append({"element_id": under_filter_id, "text": "Filter price under 100", "target_page": under_page})
    shop_controls.append(control_text(under_filter_id, "Filter price under 100", style))
    shop_elements.append({"element_id": rating_filter_id, "text": "Sort all products by rating", "target_page": rating_page})
    shop_controls.append(control_text(rating_filter_id, "Sort all products by rating", style))
    for label in ["Open shopping help", "View delivery information"]:
        distractor_page = f"{namespace}_shop_info_{len(pages):03d}"
        distractor_id = ids.new()
        pages[distractor_page] = {
            "page_id": distractor_page,
            "page_type": "shopping",
            "layout_id": layout_id,
            "split": split,
            "visible_text": f"{label}. No product results are shown on this page.",
            "elements": [],
        }
        shop_elements.append({"element_id": distractor_id, "text": label, "target_page": distractor_page})
        shop_controls.append(control_text(distractor_id, label, style))

    shop_texts = []
    for product in ordered(PRODUCTS_V2, rng, style):
        element_id = ids.new()
        shop_elements.append({"element_id": element_id, "text": product.name, "target_page": product_details[product.name]})
        shop_texts.append(product_text(product, element_id, style))
        context.shop_direct[product.name] = [element_id]
    pages[shop_home] = {
        "page_id": shop_home,
        "page_type": "shopping",
        "layout_id": layout_id,
        "split": split,
        "visible_text": "Shopping catalog. Controls: " + " ".join(shop_controls) + " Products: " + " ".join(shop_texts),
        "elements": shop_elements,
    }

    course_details: dict[str, str] = {}
    for index, course in enumerate(COURSES_V2):
        page_id = f"{namespace}_course_{index:03d}"
        course_details[course.code] = page_id
        pages[page_id] = {
            "page_id": page_id,
            "page_type": "course",
            "layout_id": layout_id,
            "split": split,
            "visible_text": (
                f"Course detail. Code: {course.code}. Title: {course.title}. Department: {course.department}. "
                f"Credits: {course.credits}. Rating: {course.rating}. Time: {course.time}."
            ),
            "elements": [],
            "answer": course.code,
        }

    department_pages: dict[str, tuple[str, str]] = {}
    for department in sorted({course.department for course in COURSES_V2}):
        page_id = f"{namespace}_course_department_{len(department_pages):02d}"
        filter_id = ids.new()
        department_pages[department] = (page_id, filter_id)
        elements = []
        texts = []
        matches = ordered((c for c in COURSES_V2 if c.department == department), rng, style)
        for course in matches:
            element_id = ids.new()
            elements.append({"element_id": element_id, "text": course.code, "target_page": course_details[course.code]})
            texts.append(course_text(course, element_id, style))
            context.course_department[(department, course.code)] = [filter_id, element_id]
        pages[page_id] = {
            "page_id": page_id,
            "page_type": "course",
            "layout_id": layout_id,
            "split": split,
            "visible_text": f"Filtered department: {department}. " + " ".join(texts),
            "elements": elements,
        }

    credit_pages: dict[int, tuple[str, str]] = {}
    for credits in sorted({course.credits for course in COURSES_V2}):
        page_id = f"{namespace}_course_credits_{credits}"
        filter_id = ids.new()
        credit_pages[credits] = (page_id, filter_id)
        elements = []
        texts = []
        matches = ordered((c for c in COURSES_V2 if c.credits == credits), rng, style)
        for course in matches:
            element_id = ids.new()
            elements.append({"element_id": element_id, "text": course.code, "target_page": course_details[course.code]})
            texts.append(course_text(course, element_id, style))
            context.course_credits[(credits, course.code)] = [filter_id, element_id]
        pages[page_id] = {
            "page_id": page_id,
            "page_type": "course",
            "layout_id": layout_id,
            "split": split,
            "visible_text": f"Filtered credits: {credits}. " + " ".join(texts),
            "elements": elements,
        }

    course_elements = []
    course_controls = []
    for department, (page_id, filter_id) in department_pages.items():
        course_elements.append({"element_id": filter_id, "text": f"Filter department {department}", "target_page": page_id})
        course_controls.append(control_text(filter_id, f"Filter department {department}", style))
    for credits, (page_id, filter_id) in credit_pages.items():
        course_elements.append({"element_id": filter_id, "text": f"Filter {credits} credits", "target_page": page_id})
        course_controls.append(control_text(filter_id, f"Filter {credits} credits", style))
    for label in ["Open enrollment help", "View campus map"]:
        distractor_page = f"{namespace}_course_info_{len(pages):03d}"
        distractor_id = ids.new()
        pages[distractor_page] = {
            "page_id": distractor_page,
            "page_type": "course",
            "layout_id": layout_id,
            "split": split,
            "visible_text": f"{label}. No course results are shown on this page.",
            "elements": [],
        }
        course_elements.append({"element_id": distractor_id, "text": label, "target_page": distractor_page})
        course_controls.append(control_text(distractor_id, label, style))

    course_texts = []
    for course in ordered(COURSES_V2, rng, style):
        element_id = ids.new()
        course_elements.append({"element_id": element_id, "text": course.code, "target_page": course_details[course.code]})
        course_texts.append(course_text(course, element_id, style))
        context.course_direct[course.code] = [element_id]
    pages[course_home] = {
        "page_id": course_home,
        "page_type": "course",
        "layout_id": layout_id,
        "split": split,
        "visible_text": "Course catalog. Controls: " + " ".join(course_controls) + " Courses: " + " ".join(course_texts),
        "elements": course_elements,
    }

    context.element_ids = set(ids.ids)
    context.page_ids = set(pages)
    return pages, context


def generate_v2_pages(output_dir: str | Path = DEFAULT_OUTPUT_DIR, seed: int = 31) -> tuple[dict[str, Any], list[LayoutContext], dict[str, Any]]:
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    layout_specs = [
        ("train_a", "train", "list", seed + 101),
        ("train_b", "train", "compact", seed + 202),
        ("eval_c", "eval", "grid", seed + 303),
    ]
    metadata: dict[str, Any] = {}
    contexts: list[LayoutContext] = []
    for layout_id, split, style, layout_seed in layout_specs:
        pages, context = create_layout(layout_id, split, style, layout_seed)
        overlap = set(metadata).intersection(pages)
        if overlap:
            raise ValueError(f"Duplicate V2 page ids: {sorted(overlap)[:5]}")
        metadata.update(pages)
        contexts.append(context)

    styles = {context.layout_id: context.style for context in contexts}
    for page_id, page in metadata.items():
        page["html_path"] = f"pages/generated_pages_v2/{page_id}.html"
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
        "version": "v2",
        "seed": seed,
        "num_pages": len(metadata),
        "num_products": len(PRODUCTS_V2),
        "num_courses": len(COURSES_V2),
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
    (output_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata, contexts, manifest
