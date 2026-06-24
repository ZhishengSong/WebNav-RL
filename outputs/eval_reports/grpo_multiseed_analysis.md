# GRPO-KL Multi-Seed Analysis

Baseline: `SFT step200` on 200 tasks.

## Overall Results

| Run | Successes | Success rate | Delta vs baseline | Invalid tool-call rate | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT step200 | 121 | 60.50% | - | 1.57% | - |
| seed7 | 128 | 64.00% | +3.50 pp | 1.57% | 0.1435 |
| seed17 | 126 | 63.00% | +2.50 pp | 1.88% | 0.4049 |
| seed29 | 119 | 59.50% | -1.00 pp | 1.74% | 0.8318 |

## Aggregate

- Mean success rate: **62.17%**
- Sample standard deviation: **2.36 pp**
- Mean delta vs baseline: **+1.67 pp**
- Best candidate: `seed7`
- Robust improvements (majority of seeds): **11**
- Robust regressions (majority of seeds): **6**

## Paired Transitions

| Run | Wrong to correct | Correct to wrong | Net | Both correct | Both wrong |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed7 | 12 | 5 | +7 | 116 | 67 |
| seed17 | 14 | 9 | +5 | 112 | 65 |
| seed29 | 10 | 12 | -2 | 109 | 69 |

## Template-Level Mean Delta

| Template | Tasks | Baseline | GRPO mean | Delta |
| --- | ---: | ---: | ---: | ---: |
| `course_title` | 20 | 60.00% | 81.67% | +21.67 pp |
| `course_4_credit_department` | 6 | 33.33% | 44.44% | +11.11 pp |
| `shopping_name` | 27 | 40.74% | 50.62% | +9.88 pp |
| `shopping_category_lowest_price` | 11 | 63.64% | 69.70% | +6.06 pp |
| `shopping_color_category` | 26 | 42.31% | 44.87% | +2.56 pp |
| `course_department_time` | 26 | 88.46% | 88.46% | +0.00 pp |
| `course_4_credit_highest_rating` | 5 | 100.00% | 100.00% | +0.00 pp |
| `shopping_category_highest_rating` | 13 | 7.69% | 7.69% | +0.00 pp |
| `shopping_price_lookup` | 27 | 44.44% | 44.44% | +0.00 pp |
| `shopping_under_100_lowest_price` | 2 | 0.00% | 0.00% | +0.00 pp |
| `course_code` | 26 | 100.00% | 93.59% | -6.41 pp |
| `course_department_highest_rating` | 11 | 100.00% | 63.64% | -36.36 pp |
