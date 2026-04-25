# Task 4: Optimization Model — Manufacturing Profit Maximization

## Overview
Solves a real-world **manufacturing profit maximization** problem using **Linear Programming** with Python's **PuLP** library.

**Goal**: Determine how many units of each product to produce per week to **maximize total profit**, subject to machine capacity, raw material, and demand constraints.

---

## Files

| File | Description |
|------|-------------|
| `task4_optimization.py` | Main script — problem setup, LP model, solution, sensitivity analysis |
| `task4_output/optimization_results.png` | 4-panel visualization (generated on run) |
| `README.md` | Project documentation |

---

## Business Problem

A company makes **4 products (A, B, C, D)** using 3 machines and limited raw material.

| Product | Profit/Unit | Machine 1 hrs | Machine 2 hrs | Machine 3 hrs | Material (kg) |
|---------|-------------|---------------|---------------|---------------|---------------|
| A | ₹25 | 2.0 | 1.5 | 0.5 | 3 |
| B | ₹30 | 1.0 | 2.0 | 1.0 | 4 |
| C | ₹15 | 3.0 | 0.5 | 0.5 | 2 |
| D | ₹40 | 1.5 | 2.5 | 2.0 | 5 |

**Constraints:**
- Machine 1: 240 hrs/week
- Machine 2: 180 hrs/week
- Machine 3: 150 hrs/week
- Raw material: 400 kg/week
- Min/Max production bounds per product

---

## What the Script Does

1. **Problem Definition** — sets up all products, profits, constraints clearly
2. **LP Model** — builds and solves using PuLP (CBC solver)
3. **Results** — optimal units per product + total profit
4. **Resource Utilization** — shows how much of each resource is used
5. **Sensitivity Analysis** — tests how profit changes if Product D's margin varies
6. **Visualizations** — 4 charts saved as PNG
7. **Business Insights** — actionable recommendations

---

## How to Run

```bash
# Install dependencies
pip install pulp pandas numpy matplotlib

# Run
python task4_optimization.py
```

---

## Sample Output

```
Product         Units Produced    Profit Contribution
Product_A            10.0                    ₹250
Product_B            10.0                    ₹300
Product_C             5.0                     ₹75
Product_D            40.0                  ₹1,600

TOTAL PROFIT                                ₹2,225
```

---

## Dependencies

```
pulp
pandas
numpy
matplotlib
```

---

## Internship Details
- **Organization**: CODTECH IT Solutions
- **Task**: Task 4 — Optimization Model
- **Domain**: Data Science
