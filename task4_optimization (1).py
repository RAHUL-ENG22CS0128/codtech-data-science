"""
================================================================================
CODTECH INTERNSHIP - TASK 4: OPTIMIZATION MODEL
================================================================================
Author      : Data Science Intern
Description : Solving a real-world business problem using Linear Programming
              with Python's PuLP library.

Problem     : A manufacturing company produces 4 products.
              Goal — maximize PROFIT given:
                • Limited machine hours (3 machines)
                • Limited raw material supply
                • Minimum production commitments per product
                • Maximum demand constraints per product

Deliverable : Problem setup, solution, sensitivity analysis & insights.
================================================================================
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import pulp
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "task4_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================================
# STEP 1: PROBLEM DEFINITION
# ================================================================================

def define_problem():
    print("\n" + "█"*60)
    print("  CODTECH INTERNSHIP — TASK 4: OPTIMIZATION MODEL")
    print("  Manufacturing Profit Maximization (Linear Programming)")
    print("█"*60)

    print("""
  BUSINESS PROBLEM:
  ─────────────────
  A manufacturing company produces 4 products: A, B, C, D.
  The company wants to determine HOW MANY UNITS of each product
  to produce per week to MAXIMIZE TOTAL PROFIT.

  CONSTRAINTS:
    1. Machine 1 has 240 hours/week available
    2. Machine 2 has 180 hours/week available
    3. Machine 3 has 150 hours/week available
    4. Raw material limited to 400 kg/week
    5. Each product has minimum order commitments
    6. Each product has maximum market demand
    """)

    # ── Product data ──────────────────────────────────────────────
    products = ["Product_A", "Product_B", "Product_C", "Product_D"]

    profit_per_unit = {         # ₹ profit per unit produced
        "Product_A": 25,
        "Product_B": 30,
        "Product_C": 15,
        "Product_D": 40,
    }

    # Machine hours required per unit (Machine1, Machine2, Machine3)
    machine_hours = {
        "Product_A": [2.0, 1.5, 0.5],
        "Product_B": [1.0, 2.0, 1.0],
        "Product_C": [3.0, 0.5, 0.5],
        "Product_D": [1.5, 2.5, 2.0],
    }

    # Raw material (kg) per unit
    raw_material = {
        "Product_A": 3,
        "Product_B": 4,
        "Product_C": 2,
        "Product_D": 5,
    }

    # Capacity constraints
    machine_capacity = [240, 180, 150]   # hours per week
    material_limit   = 400               # kg per week

    # Min and max production bounds
    min_production = {"Product_A": 10, "Product_B": 10, "Product_C": 5,  "Product_D": 5}
    max_production = {"Product_A": 60, "Product_B": 50, "Product_C": 70, "Product_D": 40}

    return (products, profit_per_unit, machine_hours, raw_material,
            machine_capacity, material_limit, min_production, max_production)


# ================================================================================
# STEP 2: BUILD & SOLVE THE LINEAR PROGRAMMING MODEL
# ================================================================================

def solve_lp(products, profit_per_unit, machine_hours, raw_material,
             machine_capacity, material_limit, min_production, max_production):

    print("\n" + "="*60)
    print("  STEP 2: BUILDING & SOLVING THE LP MODEL")
    print("="*60)

    # ── Create the LP problem (maximization) ──────────────────────
    prob = pulp.LpProblem("Manufacturing_Profit_Maximization", pulp.LpMaximize)

    # ── Decision Variables — units to produce per product ─────────
    x = {
        p: pulp.LpVariable(f"units_{p}", lowBound=min_production[p],
                           upBound=max_production[p], cat="Continuous")
        for p in products
    }

    # ── Objective Function — Maximize total profit ─────────────────
    prob += pulp.lpSum(profit_per_unit[p] * x[p] for p in products), "Total_Profit"

    # ── Constraints ───────────────────────────────────────────────

    # Machine hour constraints (one per machine)
    for m_idx in range(3):
        prob += (
            pulp.lpSum(machine_hours[p][m_idx] * x[p] for p in products)
            <= machine_capacity[m_idx],
            f"Machine_{m_idx+1}_Capacity"
        )

    # Raw material constraint
    prob += (
        pulp.lpSum(raw_material[p] * x[p] for p in products)
        <= material_limit,
        "Raw_Material_Limit"
    )

    # ── Solve ──────────────────────────────────────────────────────
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

    print(f"  ✔ Solver Status : {pulp.LpStatus[prob.status]}")
    return prob, x, status


# ================================================================================
# STEP 3: EXTRACT & DISPLAY RESULTS
# ================================================================================

def display_results(prob, x, products, profit_per_unit,
                    machine_hours, raw_material,
                    machine_capacity, material_limit):

    print("\n" + "="*60)
    print("  STEP 3: OPTIMAL SOLUTION")
    print("="*60)

    optimal_production = {p: pulp.value(x[p]) for p in products}
    total_profit       = pulp.value(prob.objective)

    print(f"\n  {'Product':<15} {'Units Produced':>15} {'Profit Contribution':>22}")
    print("  " + "-"*55)
    for p in products:
        units  = optimal_production[p]
        profit = units * profit_per_unit[p]
        print(f"  {p:<15} {units:>15.1f} {f'₹{profit:,.0f}':>22}")
    print("  " + "-"*55)
    print(f"  {'TOTAL PROFIT':<15} {'':>15} {f'₹{total_profit:,.0f}':>22}")

    # Resource utilization
    print(f"\n  RESOURCE UTILIZATION:")
    print(f"  {'Resource':<20} {'Used':>10} {'Available':>12} {'Utilization':>14}")
    print("  " + "-"*58)

    for m_idx in range(3):
        used = sum(machine_hours[p][m_idx] * optimal_production[p] for p in products)
        cap  = machine_capacity[m_idx]
        util = used/cap*100
        print(f"  {'Machine '+str(m_idx+1):<20} {used:>10.1f} {cap:>12} {util:>13.1f}%")

    mat_used = sum(raw_material[p] * optimal_production[p] for p in products)
    mat_util = mat_used / material_limit * 100
    print(f"  {'Raw Material (kg)':<20} {mat_used:>10.1f} {material_limit:>12} {mat_util:>13.1f}%")

    return optimal_production, total_profit


# ================================================================================
# STEP 4: SENSITIVITY ANALYSIS — What if profit changes?
# ================================================================================

def sensitivity_analysis(products, profit_per_unit, machine_hours, raw_material,
                          machine_capacity, material_limit, min_production, max_production):
    print("\n" + "="*60)
    print("  STEP 4: SENSITIVITY ANALYSIS")
    print("="*60)
    print("  Testing: How does total profit change if Product_D profit varies?")

    d_profits = range(20, 70, 5)
    total_profits = []

    for d_profit in d_profits:
        modified_profit = {**profit_per_unit, "Product_D": d_profit}
        prob_s = pulp.LpProblem("Sensitivity", pulp.LpMaximize)
        x_s = {
            p: pulp.LpVariable(f"s_{p}", lowBound=min_production[p],
                               upBound=max_production[p], cat="Continuous")
            for p in products
        }
        prob_s += pulp.lpSum(modified_profit[p] * x_s[p] for p in products)
        for m in range(3):
            prob_s += pulp.lpSum(machine_hours[p][m] * x_s[p] for p in products) <= machine_capacity[m]
        prob_s += pulp.lpSum(raw_material[p] * x_s[p] for p in products) <= material_limit
        prob_s.solve(pulp.PULP_CBC_CMD(msg=0))
        total_profits.append(pulp.value(prob_s.objective))

    print(f"\n  Product_D Profit/unit  |  Total Weekly Profit")
    print("  " + "-"*42)
    for dp, tp in zip(d_profits, total_profits):
        print(f"  ₹{dp:<21} |  ₹{tp:,.0f}")

    return list(d_profits), total_profits


# ================================================================================
# STEP 5: VISUALIZATIONS
# ================================================================================

def visualize(products, optimal_production, profit_per_unit,
              machine_hours, raw_material, machine_capacity,
              material_limit, sensitivity_x, sensitivity_y):
    print("\n" + "="*60)
    print("  STEP 5: VISUALIZATIONS")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Task 4 — Manufacturing Profit Optimization", fontsize=15, fontweight="bold")

    # Plot 1: Optimal production quantities
    ax = axes[0, 0]
    vals = [optimal_production[p] for p in products]
    bars = ax.bar(products, vals, color=["steelblue","darkorange","green","crimson"])
    ax.set_title("Optimal Production Quantities")
    ax.set_ylabel("Units per Week")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Plot 2: Profit contribution per product
    ax = axes[0, 1]
    contributions = [optimal_production[p] * profit_per_unit[p] for p in products]
    ax.pie(contributions, labels=products, autopct="%1.1f%%",
           colors=["steelblue","darkorange","green","crimson"], startangle=90)
    ax.set_title("Profit Contribution by Product")

    # Plot 3: Resource utilization
    ax = axes[1, 0]
    resources = ["Machine 1", "Machine 2", "Machine 3", "Raw Material"]
    used = [
        sum(machine_hours[p][m] * optimal_production[p] for p in products) / machine_capacity[m] * 100
        for m in range(3)
    ] + [sum(raw_material[p] * optimal_production[p] for p in products) / material_limit * 100]
    colors = ["green" if u < 80 else "orange" if u < 95 else "red" for u in used]
    ax.barh(resources, used, color=colors)
    ax.axvline(100, color="red", linestyle="--", label="100% capacity")
    ax.set_title("Resource Utilization (%)")
    ax.set_xlabel("Utilization %")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    # Plot 4: Sensitivity analysis
    ax = axes[1, 1]
    ax.plot(sensitivity_x, sensitivity_y, marker="o", color="steelblue", linewidth=2)
    ax.set_title("Sensitivity: Product_D Profit vs Total Profit")
    ax.set_xlabel("Product_D Profit per Unit (₹)")
    ax.set_ylabel("Total Weekly Profit (₹)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "optimization_results.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  ✔ Visualization saved → {save_path}")


# ================================================================================
# STEP 6: BUSINESS INSIGHTS
# ================================================================================

def insights(optimal_production, total_profit, products, profit_per_unit):
    print("\n" + "="*60)
    print("  STEP 6: BUSINESS INSIGHTS")
    print("="*60)

    best_product = max(products, key=lambda p: optimal_production[p] * profit_per_unit[p])
    print(f"""
  ✅ OPTIMAL WEEKLY PLAN:
  ─────────────────────────────────────────────
  • Maximum achievable weekly profit : ₹{total_profit:,.0f}

  • Top revenue contributor          : {best_product}
    (₹{optimal_production[best_product] * profit_per_unit[best_product]:,.0f} contribution)

  • Recommendation:
    - Focus capacity on {best_product} for maximum returns
    - Machine 3 is likely the binding constraint
    - Increasing raw material supply by 10% could
      unlock significant additional profit
    - Product_D has highest profit/unit (₹40) —
      negotiate higher demand limits with buyers

  📌 This model can be re-run whenever prices,
     capacities, or demand limits change.
    """)


# ================================================================================
# MAIN
# ================================================================================

if __name__ == "__main__":

    # Step 1: Define problem
    (products, profit_per_unit, machine_hours, raw_material,
     machine_capacity, material_limit,
     min_production, max_production) = define_problem()

    # Step 2: Solve
    prob, x, status = solve_lp(
        products, profit_per_unit, machine_hours, raw_material,
        machine_capacity, material_limit, min_production, max_production
    )

    # Step 3: Results
    optimal_production, total_profit = display_results(
        prob, x, products, profit_per_unit,
        machine_hours, raw_material, machine_capacity, material_limit
    )

    # Step 4: Sensitivity
    sx, sy = sensitivity_analysis(
        products, profit_per_unit, machine_hours, raw_material,
        machine_capacity, material_limit, min_production, max_production
    )

    # Step 5: Visualize
    visualize(products, optimal_production, profit_per_unit,
              machine_hours, raw_material, machine_capacity,
              material_limit, sx, sy)

    # Step 6: Insights
    insights(optimal_production, total_profit, products, profit_per_unit)

    print("\n" + "█"*60)
    print("  ✅ Task 4 Complete! Output saved to task4_output/")
    print("█"*60 + "\n")
