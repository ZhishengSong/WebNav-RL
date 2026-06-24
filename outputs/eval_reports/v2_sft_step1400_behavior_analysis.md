# V2 SFT Behavior Analysis

## Funnel

- Overall success: **31.2%**
- Filtered tasks: **368**
- Correct filter rate: **95.9%**
- Candidate accuracy after a correct filter: **15.0%**
- Wrong-candidate failures: **300**
- Wrong-candidate failures choosing position 1: **111 (37.0%)**

## Candidate Positions

- Model: `{'1': 135, '12': 1, '2': 108, '3': 5, '4': 70, '9': 34}`
- Expert: `{'1': 58, '10': 2, '11': 3, '12': 3, '15': 33, '2': 62, '3': 55, '4': 56, '5': 3, '6': 19, '7': 19, '8': 34, '9': 6}`
- Wrong-only: `{'1': 111, '12': 1, '2': 100, '3': 2, '4': 55, '9': 31}`

## Templates

| Template | Success | First-position rate | Main errors |
| --- | ---: | ---: | --- |
| `v2_shopping_name` | 32/33 (97.0%) | - | wrong_click_path=1 |
| `v2_course_title` | 31/33 (93.9%) | - | wrong_click_path=2 |
| `v2_course_code` | 23/33 (69.7%) | - | wrong_click_path=10 |
| `v2_shopping_price_lookup` | 17/33 (51.5%) | - | wrong_click_path=15, invalid_tool_call=1 |
| `v2_course_department_credits_highest_rating` | 11/33 (33.3%) | 19.4% | wrong_click_path=2, wrong_candidate_after_filter=20 |
| `v2_shopping_category_lowest_price` | 11/34 (32.4%) | 67.6% | wrong_candidate_after_filter=23 |
| `v2_shopping_color_category` | 7/33 (21.2%) | 67.7% | invalid_tool_call=1, wrong_candidate_after_filter=24, format_error=1 |
| `v2_shopping_category_budget_highest_rating` | 7/34 (20.6%) | 58.8% | wrong_candidate_after_filter=27 |
| `v2_course_department_time` | 6/33 (18.2%) | 21.4% | wrong_candidate_after_filter=22, invalid_tool_call=5 |
| `v2_course_credits_department` | 6/34 (17.6%) | 44.1% | wrong_candidate_after_filter=28 |
| `v2_course_department_highest_rating` | 5/33 (15.2%) | 18.5% | wrong_candidate_after_filter=22, invalid_tool_call=6 |
| `v2_course_credits_highest_rating` | 0/33 (0.0%) | 51.5% | wrong_candidate_after_filter=33 |
| `v2_shopping_category_highest_rating` | 0/34 (0.0%) | 64.7% | wrong_candidate_after_filter=34 |
| `v2_shopping_under_100_highest_rating` | 0/34 (0.0%) | 0.0% | wrong_candidate_after_filter=34 |
| `v2_shopping_under_100_lowest_price` | 0/33 (0.0%) | 0.0% | wrong_candidate_after_filter=33 |
