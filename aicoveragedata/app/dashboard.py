import base64
import json
import os
import sys
from io import BytesIO
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = APP_DIR.parent
PROJECT_ROOT = PACKAGE_DIR.parent
SITE_DIR = PACKAGE_DIR / "site"

sys.path.append(str(PROJECT_ROOT))
from aicoveragedata.data import load_dataset
from aicoveragedata.app.agent_widget import (
    agent_widget_markup,
    agent_widget_script,
    agent_widget_styles,
)
from aicoveragedata.regression.build_report import write_page as write_regression_page


OUTPUT_FILE = SITE_DIR / "index.html"
PROFILE_OUTPUT_FILE = SITE_DIR / "industry_country_profiles.html"
DOWNLOAD_DIR = SITE_DIR / "downloads" / "dashboard"

DASHBOARD_THEME_CSS = """
        :root {
            color-scheme: dark;
            --page-bg: #05070c;
            --surface: rgba(12, 18, 30, 0.94);
            --surface-strong: rgba(18, 28, 45, 0.96);
            --line: rgba(96, 220, 255, 0.24);
            --line-strong: rgba(96, 220, 255, 0.38);
            --text: #e8f7ff;
            --muted: #91a9bd;
            --accent: #36d8ff;
            --accent-hot: #ff65c8;
            --accent-soft: rgba(54, 216, 255, 0.16);
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
            background:
                repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.035) 0 1px, transparent 1px 96px),
                repeating-linear-gradient(0deg, rgba(255, 255, 255, 0.025) 0 1px, transparent 1px 96px),
                linear-gradient(135deg, #05070c 0%, #101927 52%, #06080f 100%);
            color: var(--text);
            line-height: 1.45;
        }
        header {
            background: linear-gradient(180deg, rgba(15, 24, 38, 0.98), rgba(6, 9, 16, 0.96));
            border-bottom: 1px solid var(--line);
            box-shadow: 0 18px 48px rgba(54, 216, 255, 0.08);
        }
        h1 {
            letter-spacing: 0;
            text-shadow: 0 0 18px rgba(54, 216, 255, 0.38);
        }
        header p, .stat span, .compare-controls label, .compare-axis-label,
        .compare-legend {
            color: var(--muted);
        }
        a {
            color: var(--accent);
        }
        .top-links a, .download, .image-download, .dashboard-tab,
        .compare-controls select {
            background: var(--surface);
            border-color: var(--line-strong);
            color: var(--text);
            box-shadow: inset 0 0 14px rgba(54, 216, 255, 0.07);
            transition: transform 180ms ease, background 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
        }
        .top-links a:hover, .download:hover, .image-download:hover,
        .dashboard-tab:hover {
            background: var(--surface-strong);
            border-color: var(--accent);
            color: var(--text);
            box-shadow: 0 0 18px rgba(54, 216, 255, 0.14), inset 0 0 18px rgba(54, 216, 255, 0.08);
            transform: translateY(-1px);
        }
        .dashboard-tab.active {
            background: linear-gradient(135deg, rgba(54, 216, 255, 0.24), rgba(255, 101, 200, 0.18));
            border-color: var(--accent);
            color: #ffffff;
            box-shadow: 0 0 24px rgba(54, 216, 255, 0.22), inset 0 0 20px rgba(255, 101, 200, 0.10);
        }
        .tab-panel.active {
            animation: dashboardPanelIn 260ms ease both;
        }
        .stat, .card, .compare-card {
            background: linear-gradient(180deg, var(--surface-strong), var(--surface));
            border-color: var(--line);
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.30), 0 0 24px rgba(54, 216, 255, 0.07);
        }
        .stat strong, .card h2, .compare-card h3, .downloads h2, .compare-section h2 {
            color: var(--text);
        }
        .compare-chart {
            border-top-color: var(--line);
        }
        .card img {
            background: #080d16;
            border: 1px solid var(--line);
            border-radius: 6px;
            box-shadow: 0 0 18px rgba(54, 216, 255, 0.06);
        }
        .legend-chip {
            box-shadow: 0 0 10px currentColor;
        }
        .top-links a:focus-visible, .download:focus-visible,
        .image-download:focus-visible, .dashboard-tab:focus-visible,
        .compare-controls select:focus-visible {
            outline: 3px solid rgba(54, 216, 255, 0.25);
            outline-offset: 2px;
        }
        @keyframes dashboardPanelIn {
            from {
                opacity: 0;
                transform: translateY(10px) scale(0.99);
                filter: blur(2px);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
                filter: blur(0);
            }
        }
        @media (prefers-reduced-motion: reduce) {
            .top-links a, .download, .image-download, .dashboard-tab,
            .tab-panel.active {
                animation: none;
                transition: none;
            }
        }
"""

PROFILE_THEME_CSS = """
        :root {
            color-scheme: dark;
            --page-bg: #05070c;
            --surface: rgba(12, 18, 30, 0.94);
            --surface-strong: rgba(18, 28, 45, 0.96);
            --line: rgba(96, 220, 255, 0.24);
            --line-strong: rgba(96, 220, 255, 0.38);
            --text: #e8f7ff;
            --muted: #91a9bd;
            --accent: #36d8ff;
            --accent-hot: #ff65c8;
            --accent-soft: rgba(54, 216, 255, 0.16);
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
            background:
                repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.035) 0 1px, transparent 1px 96px),
                repeating-linear-gradient(0deg, rgba(255, 255, 255, 0.025) 0 1px, transparent 1px 96px),
                linear-gradient(135deg, #05070c 0%, #101927 52%, #06080f 100%);
            color: var(--text);
            line-height: 1.45;
        }
        header {
            background: linear-gradient(180deg, rgba(15, 24, 38, 0.98), rgba(6, 9, 16, 0.96));
            border-bottom: 1px solid var(--line);
            box-shadow: 0 18px 48px rgba(54, 216, 255, 0.08);
        }
        h1 {
            letter-spacing: 0;
            text-shadow: 0 0 18px rgba(54, 216, 255, 0.38);
        }
        a {
            color: var(--accent);
        }
        label, .score-card span, .component-item span, .component-item small,
        .score-row-item span, .rank-item span, th, .trend-axis-label {
            color: var(--muted);
            fill: var(--muted);
        }
        button, select {
            background: var(--surface);
            border-color: var(--line-strong);
            color: var(--text);
            box-shadow: inset 0 0 14px rgba(54, 216, 255, 0.07);
            transition: transform 180ms ease, background 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
        }
        button:hover {
            background: var(--surface-strong);
            border-color: var(--accent);
            box-shadow: 0 0 18px rgba(54, 216, 255, 0.14), inset 0 0 18px rgba(54, 216, 255, 0.08);
            transform: translateY(-1px);
        }
        button.active {
            background: linear-gradient(135deg, rgba(54, 216, 255, 0.24), rgba(255, 101, 200, 0.18));
            border-color: var(--accent);
            color: #ffffff;
            box-shadow: 0 0 24px rgba(54, 216, 255, 0.22), inset 0 0 20px rgba(255, 101, 200, 0.10);
        }
        .score-card, .panel, .trend-panel, .component-panel, .breakdown-panel,
        .fixed-score-panel, .component-item {
            background: linear-gradient(180deg, var(--surface-strong), var(--surface));
            border-color: var(--line);
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.30), 0 0 24px rgba(54, 216, 255, 0.07);
        }
        .trend-chart, th, td, .score-row-item, .rank-item {
            border-color: var(--line);
        }
        .trend-value-label {
            fill: var(--text);
            filter: drop-shadow(0 0 6px rgba(54, 216, 255, 0.30));
        }
        table {
            background: rgba(8, 13, 22, 0.72);
        }
        .page.page-pulse section:not(.toolbar) {
            animation: profilePanelIn 260ms ease both;
        }
        button:focus-visible, select:focus-visible, a:focus-visible {
            outline: 3px solid rgba(54, 216, 255, 0.25);
            outline-offset: 2px;
        }
        @keyframes profilePanelIn {
            from {
                opacity: 0;
                transform: translateY(10px);
                filter: blur(2px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
                filter: blur(0);
            }
        }
        @media (prefers-reduced-motion: reduce) {
            button, .page.page-pulse section:not(.toolbar) {
                animation: none;
                transition: none;
            }
        }
"""


def chart_to_base64(title, draw_chart):
    plt.figure(figsize=(10, 6))
    draw_chart()
    plt.title(title)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=140)
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def chart_card(title, image_base64):
    filename = title.lower().replace(" ", "_").replace("|", "").replace(":", "") + ".png"
    return f"""
    <section class="card">
        <div class="card-header">
            <h2>{title}</h2>
            <a class="image-download" href="data:image/png;base64,{image_base64}" download="{filename}">
                Download Image
            </a>
        </div>
        <img src="data:image/png;base64,{image_base64}" alt="{title}">
    </section>
    """


def download_link(filename, label):
    return f'<a class="download" href="downloads/dashboard/{filename}" download>{label}</a>'


def adoption_compare_payload(table):
    years = [int(year) for year in table.index]
    entities = [str(entity) for entity in table.columns]
    values = {
        str(entity): [
            None if pd.isna(value) else round(float(value), 4)
            for value in table[entity]
        ]
        for entity in table.columns
    }
    return years, entities, values


def minmax_score(series):
    low = series.min()
    high = series.max()
    if pd.isna(low) or pd.isna(high) or high == low:
        return pd.Series(50, index=series.index)
    return ((series - low) / (high - low)) * 100


def format_profile_value(column, value):
    if column in ["rows", "companies", "first_year", "last_year"]:
        return f"{int(value):,}"
    if column in ["ai_investment_usd", "cost_savings", "revenue_impact", "total_benefit", "net_value"]:
        return f"${value / 1_000_000:,.2f}M"
    return f"{value:,.3f}"


def profile_payload(df, group_col):
    numeric_columns = [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in ["company_id", "year"]
    ]
    profile = df.groupby(group_col).agg(
        rows=("company_id", "size"),
        companies=("company_id", "nunique"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        **{column: (column, "mean") for column in numeric_columns},
    )

    score_columns = [
        "automation_rate",
        "productivity_gain",
        "employee_ai_training_hours",
        "ai_maturity_score",
        "deployment_count",
        "roi",
        "net_value",
    ]
    score_labels = {
        "automation_rate": "Automation Rate",
        "productivity_gain": "Productivity Gain",
        "employee_ai_training_hours": "Training Hours",
        "ai_maturity_score": "AI Maturity Score",
        "deployment_count": "Deployment Count",
        "roi": "ROI",
        "net_value": "Net Value",
    }
    score_frame = pd.DataFrame(
        {column: minmax_score(profile[column]) for column in score_columns}
    )
    profile["final_ai_score"] = score_frame.mean(axis=1).round(1)
    profile = profile.sort_values("final_ai_score", ascending=False)

    records = []
    for name, row in profile.iterrows():
        variables = []
        for column in profile.columns:
            value = row[column]
            variables.append(
                {
                    "name": column.replace("_", " ").title(),
                    "value": format_profile_value(column, value),
                }
            )
        components = [
            {
                "name": score_labels[column],
                "rawValue": format_profile_value(column, row[column]),
                "score": round(float(score_frame.loc[name, column]), 1),
            }
            for column in score_columns
        ]
        raw_values = {
            column: round(float(row[column]), 4)
            for column in profile.columns
            if column not in ["rows", "companies", "first_year", "last_year"]
        }
        records.append(
            {
                "name": str(name),
                "score": float(row["final_ai_score"]),
                "rank": len(records) + 1,
                "variables": variables,
                "components": components,
                "rawValues": raw_values,
            }
        )
    return records


def profile_export_rows(profiles, entity_type):
    rows = []
    for profile in profiles:
        row = {
            "entity_type": entity_type,
            "name": profile["name"],
            "rank": profile["rank"],
            "final_ai_score": profile["score"],
        }
        for key, value in profile["rawValues"].items():
            row[f"avg_{key}"] = value
        for component in profile["components"]:
            key = component["name"].lower().replace(" ", "_")
            row[f"{key}_component_score"] = component["score"]
        rows.append(row)
    return rows


def rate_breakdown_payload(df, group_col, breakdown_col):
    grouped = (
        df.groupby([group_col, breakdown_col])
        .agg(
            rows=("company_id", "size"),
            companies=("company_id", "nunique"),
            avg_ai_adoption_level=("ai_adoption_level", "mean"),
            avg_ai_investment_usd=("ai_investment_usd", "mean"),
            avg_productivity_gain=("productivity_gain", "mean"),
            avg_roi=("roi", "mean"),
        )
        .reset_index()
        .sort_values([group_col, "avg_ai_adoption_level"], ascending=[True, False])
    )

    payload = {}
    for group, values in grouped.groupby(group_col):
        payload[str(group)] = [
            {
                "name": str(row[breakdown_col]),
                "rows": int(row["rows"]),
                "companies": int(row["companies"]),
                "adoption": round(float(row["avg_ai_adoption_level"]), 4),
                "investment": f"${row['avg_ai_investment_usd'] / 1_000_000:,.2f}M",
                "productivity": round(float(row["avg_productivity_gain"]), 4),
                "roi": round(float(row["avg_roi"]), 4),
            }
            for _, row in values.iterrows()
        ]
    return payload, grouped


def write_profiles_page(
    industry_profiles,
    country_profiles,
    industry_years,
    industry_values,
    country_years,
    country_values,
    industry_country_rates,
    country_industry_rates,
):
    html = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Industry and Country AI Profiles</title>
    <style>
        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            color: #17202a;
        }}
        header {{
            padding: 24px 32px 16px;
            background: #ffffff;
            border-bottom: 1px solid #d8dee4;
        }}
        h1 {{
            margin: 0 0 10px;
            font-size: 30px;
        }}
        a {{
            color: #1769aa;
            text-decoration: none;
        }}
        .page {{
            padding: 20px 32px 32px;
        }}
        .toolbar, .score-row, .profile-grid {{
            display: grid;
            gap: 12px;
        }}
        .toolbar {{
            grid-template-columns: auto minmax(220px, 320px);
            align-items: end;
            margin-bottom: 16px;
        }}
        .tabs {{
            display: flex;
            gap: 8px;
        }}
        button, select {{
            border: 1px solid #b7c2cc;
            border-radius: 6px;
            background: #ffffff;
            color: #17202a;
        }}
        button {{
            padding: 9px 12px;
            cursor: pointer;
        }}
        button.active {{
            background: #17202a;
            color: #ffffff;
        }}
        label {{
            display: grid;
            gap: 5px;
            color: #5b6670;
            font-size: 13px;
        }}
        select {{
            padding: 9px 10px;
        }}
        .score-row {{
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            margin-bottom: 16px;
        }}
        .score-card, .panel {{
            background: #ffffff;
            border: 1px solid #d8dee4;
            border-radius: 8px;
            padding: 16px;
        }}
        .score-card span {{
            display: block;
            color: #5b6670;
            font-size: 13px;
            margin-bottom: 7px;
        }}
        .score-card strong {{
            font-size: 26px;
        }}
        .trend-panel {{
            background: #ffffff;
            border: 1px solid #d8dee4;
            border-radius: 8px;
            margin-bottom: 16px;
            padding: 16px;
        }}
        .trend-chart {{
            min-height: 360px;
            overflow-x: auto;
            border-top: 1px solid #d8dee4;
            padding-top: 12px;
        }}
        .trend-svg {{
            min-width: 960px;
            width: 100%;
            height: 360px;
        }}
        .trend-axis-label {{
            fill: #5b6670;
            font-size: 13px;
        }}
        .trend-value-label {{
            fill: #17202a;
            font-size: 13px;
            font-weight: 700;
        }}
        .component-panel {{
            background: #ffffff;
            border: 1px solid #d8dee4;
            border-radius: 8px;
            margin-bottom: 16px;
            padding: 16px;
        }}
        .breakdown-panel {{
            background: #ffffff;
            border: 1px solid #d8dee4;
            border-radius: 8px;
            margin-bottom: 16px;
            padding: 16px;
        }}
        .component-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 10px;
        }}
        .component-item {{
            border: 1px solid #e1e6eb;
            border-radius: 6px;
            padding: 10px;
        }}
        .component-item span {{
            display: block;
            color: #5b6670;
            font-size: 12px;
            margin-bottom: 5px;
        }}
        .component-item strong {{
            font-size: 20px;
        }}
        .component-item small {{
            display: block;
            color: #5b6670;
            margin-top: 5px;
        }}
        .profile-grid {{
            grid-template-columns: minmax(0, 1.4fr) minmax(280px, 0.8fr);
        }}
        .fixed-score-panel {{
            background: #ffffff;
            border: 1px solid #d8dee4;
            border-radius: 8px;
            margin-bottom: 16px;
            padding: 16px;
        }}
        .fixed-score-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
        }}
        .score-list {{
            display: grid;
            gap: 8px;
        }}
        .score-row-item {{
            display: grid;
            grid-template-columns: 42px minmax(0, 1fr) 64px;
            gap: 10px;
            align-items: center;
            padding: 9px 0;
            border-bottom: 1px solid #e1e6eb;
        }}
        .score-row-item span {{
            color: #5b6670;
            font-size: 13px;
        }}
        .score-row-item strong {{
            font-size: 14px;
        }}
        h2 {{
            margin: 0 0 12px;
            font-size: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            padding: 10px 8px;
            border-bottom: 1px solid #e1e6eb;
            text-align: left;
        }}
        th {{
            color: #5b6670;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .rank-list {{
            display: grid;
            gap: 8px;
        }}
        .rank-item {{
            display: flex;
            justify-content: space-between;
            gap: 12px;
            padding: 9px 0;
            border-bottom: 1px solid #e1e6eb;
        }}
        .rank-item strong {{
            font-size: 14px;
        }}
        .rank-item span {{
            color: #5b6670;
            font-size: 13px;
        }}
        {PROFILE_THEME_CSS}
        {agent_widget_styles()}
        @media (max-width: 760px) {{
            header, .page {{
                padding-left: 14px;
                padding-right: 14px;
            }}
            .toolbar, .profile-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>Industry and Country AI Profiles</h1>
        <a href="index.html">Back to dashboard</a>
    </header>
    <div class="page">
        <section class="fixed-score-panel">
            <h2>Fixed AI Quality Score Rankings</h2>
            <div class="fixed-score-grid">
                <div>
                    <h2>Industries</h2>
                    <div class="score-list" id="fixed-industry-scores"></div>
                </div>
                <div>
                    <h2>Countries</h2>
                    <div class="score-list" id="fixed-country-scores"></div>
                </div>
            </div>
        </section>
        <section class="toolbar">
            <div class="tabs">
                <button class="active" data-type="industry">Industries</button>
                <button data-type="country">Countries</button>
            </div>
            <label>
                Profile
                <select id="entity-select"></select>
            </label>
        </section>
        <section class="score-row">
            <div class="score-card"><span>AI Quality Score</span><strong id="score"></strong></div>
            <div class="score-card"><span>Rank</span><strong id="rank"></strong></div>
            <div class="score-card"><span>Companies</span><strong id="companies"></strong></div>
            <div class="score-card"><span>Years</span><strong id="years"></strong></div>
        </section>
        <section class="trend-panel">
            <h2 id="trend-title"></h2>
            <div class="trend-chart" id="trend-chart"></div>
        </section>
        <section class="component-panel">
            <h2>AI Quality Score Components</h2>
            <div class="component-grid" id="components"></div>
        </section>
        <section class="breakdown-panel">
            <h2 id="breakdown-title"></h2>
            <table>
                <thead>
                    <tr>
                        <th id="breakdown-name-header"></th>
                        <th>Adoption Rate</th>
                        <th>Companies</th>
                        <th>Avg Investment</th>
                        <th>Productivity</th>
                        <th>ROI</th>
                    </tr>
                </thead>
                <tbody id="breakdown-rows"></tbody>
            </table>
        </section>
        <section class="profile-grid">
            <div class="panel">
                <h2 id="table-title"></h2>
                <table>
                    <thead><tr><th>Variable</th><th>Value</th></tr></thead>
                    <tbody id="variables"></tbody>
                </table>
            </div>
            <div class="panel">
                <h2>Top AI Scores</h2>
                <div class="rank-list" id="rank-list"></div>
            </div>
        </section>
    </div>
    {agent_widget_markup()}
    <script>
        {agent_widget_script(os.getenv("AGENT_API_BASE", ""))}
        const profiles = {{
            industry: {json.dumps(industry_profiles)},
            country: {json.dumps(country_profiles)}
        }};
        const adoptionData = {{
            industry: {{
                years: {json.dumps(industry_years)},
                values: {json.dumps(industry_values)}
            }},
            country: {{
                years: {json.dumps(country_years)},
                values: {json.dumps(country_values)}
            }}
        }};
        const rateBreakdowns = {{
            industry: {json.dumps(industry_country_rates)},
            country: {json.dumps(country_industry_rates)}
        }};
        const tabs = document.querySelectorAll("[data-type]");
        const select = document.getElementById("entity-select");
        const trendChart = document.getElementById("trend-chart");
        const pageShell = document.querySelector(".page");
        let currentType = "industry";

        function renderFixedScores() {{
            document.getElementById("fixed-industry-scores").innerHTML = profiles.industry.map((profile) => `
                <div class="score-row-item">
                    <span>#${{profile.rank}}</span>
                    <strong>${{profile.name}}</strong>
                    <span>${{profile.score.toFixed(1)}}</span>
                </div>
            `).join("");
            document.getElementById("fixed-country-scores").innerHTML = profiles.country.map((profile) => `
                <div class="score-row-item">
                    <span>#${{profile.rank}}</span>
                    <strong>${{profile.name}}</strong>
                    <span>${{profile.score.toFixed(1)}}</span>
                </div>
            `).join("");
        }}

        function valueOf(profile, variableName) {{
            const item = profile.variables.find((variable) => variable.name === variableName);
            return item ? item.value : "";
        }}

        function loadOptions() {{
            select.innerHTML = "";
            profiles[currentType].forEach((profile) => {{
                select.add(new Option(profile.name, profile.name));
            }});
        }}

        function renderTrend(profile) {{
            const years = adoptionData[currentType].years;
            const series = adoptionData[currentType].values[profile.name];
            const values = series.filter((value) => value !== null);
            const maxValue = Math.max(...values, 1);
            const width = Math.max(960, years.length * 76);
            const height = 360;
            const left = 54;
            const right = 24;
            const top = 26;
            const bottom = 46;
            const plotWidth = width - left - right;
            const plotHeight = height - top - bottom;
            const points = series.map((value, index) => {{
                const x = left + (index / Math.max(years.length - 1, 1)) * plotWidth;
                const y = top + plotHeight - (((value ?? 0) / maxValue) * plotHeight);
                return {{ x, y, value: value ?? 0, year: years[index] }};
            }});
            const path = points.map((point, index) => `${{index === 0 ? "M" : "L"}} ${{point.x}} ${{point.y}}`).join(" ");
            const circles = points.map((point) => `
                <circle cx="${{point.x}}" cy="${{point.y}}" r="6.5" fill="#2f80ed">
                    <title>${{profile.name}} ${{point.year}}: ${{point.value.toFixed(3)}}</title>
                </circle>
            `).join("");
            const labels = points.map((point) => `
                <text x="${{point.x}}" y="${{point.y - 12}}" text-anchor="middle" class="trend-value-label">${{point.value.toFixed(2)}}</text>
            `).join("");
            const yearLabels = points.map((point) => `
                <text x="${{point.x}}" y="${{height - 13}}" text-anchor="middle" class="trend-axis-label">${{point.year}}</text>
            `).join("");

            document.getElementById("trend-title").textContent = `${{profile.name}} Adoption Trend`;
            trendChart.innerHTML = `
                <svg class="trend-svg" viewBox="0 0 ${{width}} ${{height}}" role="img">
                    <line x1="${{left}}" y1="${{top}}" x2="${{left}}" y2="${{height - bottom}}" stroke="#b7c2cc" />
                    <line x1="${{left}}" y1="${{height - bottom}}" x2="${{width - right}}" y2="${{height - bottom}}" stroke="#b7c2cc" />
                    <text x="8" y="${{top + 8}}" class="trend-axis-label">${{maxValue.toFixed(2)}}</text>
                    <text x="22" y="${{height - bottom}}" class="trend-axis-label">0</text>
                    <path d="${{path}}" fill="none" stroke="#2f80ed" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" />
                    ${{circles}}
                    ${{labels}}
                    ${{yearLabels}}
                </svg>
            `;
        }}

        function renderBreakdown(profile) {{
            const rows = rateBreakdowns[currentType][profile.name] || [];
            const breakdownName = currentType === "country" ? "Industry" : "Country";
            document.getElementById("breakdown-title").textContent =
                currentType === "country"
                    ? `${{profile.name}} Industry Rates By Amount`
                    : `${{profile.name}} Country Rates By Amount`;
            document.getElementById("breakdown-name-header").textContent = breakdownName;
            document.getElementById("breakdown-rows").innerHTML = rows.map((row) => `
                <tr>
                    <td>${{row.name}}</td>
                    <td>${{row.adoption.toFixed(3)}}</td>
                    <td>${{row.companies.toLocaleString()}}</td>
                    <td>${{row.investment}}</td>
                    <td>${{row.productivity.toFixed(3)}}</td>
                    <td>${{row.roi.toFixed(3)}}</td>
                </tr>
            `).join("");
        }}

        function renderProfile() {{
            const profile = profiles[currentType].find((item) => item.name === select.value);
            document.getElementById("score").textContent = profile.score.toFixed(1);
            document.getElementById("rank").textContent = `#${{profile.rank}}`;
            document.getElementById("companies").textContent = valueOf(profile, "Companies");
            document.getElementById("years").textContent = `${{valueOf(profile, "First Year")}}-${{valueOf(profile, "Last Year")}}`;
            document.getElementById("table-title").textContent = `${{profile.name}} Variables`;
            document.getElementById("variables").innerHTML = profile.variables.map((variable) => `
                <tr><td>${{variable.name}}</td><td>${{variable.value}}</td></tr>
            `).join("");
            document.getElementById("components").innerHTML = profile.components.map((component) => `
                <div class="component-item">
                    <span>${{component.name}}</span>
                    <strong>${{component.score.toFixed(1)}} / 100</strong>
                    <small>Raw: ${{component.rawValue}}</small>
                </div>
            `).join("");
            renderTrend(profile);
            renderBreakdown(profile);
            document.getElementById("rank-list").innerHTML = profiles[currentType].slice(0, 10).map((item) => `
                <div class="rank-item">
                    <strong>#${{item.rank}} ${{item.name}}</strong>
                    <span>${{item.score.toFixed(1)}}</span>
                </div>
            `).join("");
        }}

        tabs.forEach((tab) => {{
            tab.addEventListener("click", () => {{
                currentType = tab.dataset.type;
                tabs.forEach((item) => item.classList.toggle("active", item === tab));
                loadOptions();
                renderProfile();
                if (pageShell) {{
                    pageShell.classList.remove("page-pulse");
                    void pageShell.offsetWidth;
                    pageShell.classList.add("page-pulse");
                }}
            }});
        }});
        select.addEventListener("change", renderProfile);
        renderFixedScores();
        loadOptions();
        renderProfile();
        setupAgentWidget();
    </script>
</body>
</html>
"""
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_OUTPUT_FILE.write_text(html, encoding="utf-8")


def main():
    df = load_dataset()

    df["total_benefit"] = df["cost_savings"] + df["revenue_impact"]
    df["net_value"] = df["total_benefit"] - df["ai_investment_usd"]
    df["roi"] = df["total_benefit"] / df["ai_investment_usd"]

    summary = {
        "Rows": f"{len(df):,}",
        "Variables": f"{len(df.columns):,}",
        "Years": f"{int(df['year'].min())} - {int(df['year'].max())}",
        "Companies": f"{df['company_id'].nunique():,}",
        "Industries": f"{df['industry'].nunique():,}",
        "Countries": f"{df['country'].nunique():,}",
    }

    yearly_adoption = df.groupby("year")["ai_adoption_level"].mean()
    value_trend = df.groupby("year").agg(
        avg_investment=("ai_investment_usd", "mean"),
        avg_total_benefit=("total_benefit", "mean"),
    ) / 1_000_000
    industry_adoption = df.groupby(["year", "industry"])["ai_adoption_level"].mean().unstack()
    country_adoption = df.groupby(["year", "country"])["ai_adoption_level"].mean().unstack()
    industry_net_value = df.groupby("industry")["net_value"].mean().sort_values() / 1_000_000
    industry_years, industry_entities, industry_values = adoption_compare_payload(industry_adoption)
    country_years, country_entities, country_values = adoption_compare_payload(country_adoption)

    df["training_group"] = pd.qcut(
        df["employee_ai_training_hours"],
        q=4,
        labels=["Low", "Mid-Low", "Mid-High", "High"],
    )
    deployment_by_training = df.groupby("training_group", observed=True)["deployment_count"].mean()

    df["invest_group"] = pd.qcut(
        df["ai_investment_usd"],
        q=4,
        labels=["Low", "Mid-Low", "Mid-High", "High"],
    )
    roi_by_investment = df.groupby("invest_group", observed=True)["roi"].mean()

    yearly_corr = df.groupby("year")[["ai_adoption_level", "productivity_gain"]].mean()
    corr_value = yearly_corr["ai_adoption_level"].corr(yearly_corr["productivity_gain"])
    industry_profiles = profile_payload(df, "industry")
    country_profiles = profile_payload(df, "country")
    industry_country_rates, industry_country_rates_table = rate_breakdown_payload(
        df,
        "industry",
        "country",
    )
    country_industry_rates, country_industry_rates_table = rate_breakdown_payload(
        df,
        "country",
        "industry",
    )

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DOWNLOAD_DIR / "full_dataset.csv", index=False)
    pd.DataFrame(summary.items(), columns=["metric", "value"]).to_csv(
        DOWNLOAD_DIR / "summary.csv",
        index=False,
    )
    yearly_adoption.to_csv(DOWNLOAD_DIR / "yearly_adoption.csv", header=["ai_adoption_level"])
    industry_adoption.to_csv(DOWNLOAD_DIR / "industry_adoption.csv")
    country_adoption.to_csv(DOWNLOAD_DIR / "country_adoption.csv")
    value_trend.to_csv(DOWNLOAD_DIR / "investment_benefit_by_year.csv")
    industry_net_value.to_csv(DOWNLOAD_DIR / "industry_net_value.csv", header=["net_value_millions"])
    deployment_by_training.to_csv(
        DOWNLOAD_DIR / "deployment_by_training_quartile.csv",
        header=["avg_deployment_count"],
    )
    roi_by_investment.to_csv(
        DOWNLOAD_DIR / "roi_by_investment_quartile.csv",
        header=["avg_roi"],
    )
    yearly_corr.to_csv(DOWNLOAD_DIR / "adoption_productivity_by_year.csv")
    pd.DataFrame(profile_export_rows(industry_profiles, "industry")).to_csv(
        DOWNLOAD_DIR / "industry_ai_profiles.csv",
        index=False,
    )
    pd.DataFrame(profile_export_rows(country_profiles, "country")).to_csv(
        DOWNLOAD_DIR / "country_ai_profiles.csv",
        index=False,
    )
    industry_country_rates_table.to_csv(
        DOWNLOAD_DIR / "industry_country_rates_by_amount.csv",
        index=False,
    )
    country_industry_rates_table.to_csv(
        DOWNLOAD_DIR / "country_industry_rates_by_amount.csv",
        index=False,
    )
    write_profiles_page(
        industry_profiles,
        country_profiles,
        industry_years,
        industry_values,
        country_years,
        country_values,
        industry_country_rates,
        country_industry_rates,
    )

    chart_cards = {
        "yearly_adoption": chart_card(
            "Average AI Adoption by Year",
            chart_to_base64(
                "Average AI Adoption by Year",
                lambda: (
                    yearly_adoption.plot(kind="line", marker="o"),
                    plt.xlabel("Year"),
                    plt.ylabel("AI Adoption Level"),
                    plt.grid(True),
                ),
            ),
        ),
        "industry_adoption": chart_card(
            "Industry Adoption by Year",
            chart_to_base64(
                "Industry Adoption by Year",
                lambda: (
                    plt.imshow(industry_adoption.T, aspect="auto", cmap="YlGnBu"),
                    plt.colorbar(label="Average AI Adoption Level"),
                    plt.xlabel("Year"),
                    plt.ylabel("Industry"),
                    plt.xticks(range(len(industry_adoption.index)), industry_adoption.index.astype(int), rotation=45),
                    plt.yticks(range(len(industry_adoption.columns)), industry_adoption.columns),
                ),
            ),
        ),
        "country_adoption": chart_card(
            "Country Adoption by Year",
            chart_to_base64(
                "Country Adoption by Year",
                lambda: (
                    plt.imshow(country_adoption.T, aspect="auto", cmap="YlGnBu"),
                    plt.colorbar(label="Average AI Adoption Level"),
                    plt.xlabel("Year"),
                    plt.ylabel("Country"),
                    plt.xticks(range(len(country_adoption.index)), country_adoption.index.astype(int), rotation=45),
                    plt.yticks(range(len(country_adoption.columns)), country_adoption.columns),
                ),
            ),
        ),
        "investment_benefit_year": chart_card(
            "Investment vs Total Benefit by Year",
            chart_to_base64(
                "Investment vs Total Benefit by Year",
                lambda: (
                    value_trend.plot(kind="line", marker="o", ax=plt.gca()),
                    plt.xlabel("Year"),
                    plt.ylabel("USD Millions"),
                    plt.grid(True),
                ),
            ),
        ),
        "investment_benefit_scatter": chart_card(
            "AI Investment vs Total Benefit",
            chart_to_base64(
                "AI Investment vs Total Benefit",
                lambda: (
                    plt.scatter(df["ai_investment_usd"] / 1_000_000, df["total_benefit"] / 1_000_000, alpha=0.6),
                    plt.xlabel("AI Investment USD Millions"),
                    plt.ylabel("Total Benefit USD Millions"),
                    plt.grid(True),
                ),
            ),
        ),
        "automation_productivity": chart_card(
            "Automation Rate vs Productivity Gain",
            chart_to_base64(
                "Automation Rate vs Productivity Gain",
                lambda: (
                    plt.scatter(df["automation_rate"], df["productivity_gain"], alpha=0.6),
                    plt.xlabel("Automation Rate"),
                    plt.ylabel("Productivity Gain"),
                    plt.grid(True),
                ),
            ),
        ),
        "deployment_training": chart_card(
            "Deployment Count by Training Quartile",
            chart_to_base64(
                "Average Deployment Count by Training Quartile",
                lambda: (
                    plt.bar(deployment_by_training.index, deployment_by_training),
                    plt.xlabel("AI Training Hours Quartile"),
                    plt.ylabel("Average Deployment Count"),
                    plt.grid(axis="y"),
                ),
            ),
        ),
        "roi_training_distribution": chart_card(
            "ROI Distribution by Training Quartile",
            chart_to_base64(
                "ROI Distribution by Training Quartile",
                lambda: (
                    df.boxplot(column="roi", by="training_group", showfliers=False),
                    plt.suptitle(""),
                    plt.xlabel("AI Training Hours Quartile"),
                    plt.ylabel("ROI"),
                    plt.grid(axis="y"),
                ),
            ),
        ),
        "productivity_training_distribution": chart_card(
            "Productivity Gain Distribution by Training Quartile",
            chart_to_base64(
                "Productivity Gain Distribution by Training Quartile",
                lambda: (
                    df.boxplot(column="productivity_gain", by="training_group", showfliers=False),
                    plt.suptitle(""),
                    plt.xlabel("AI Training Hours Quartile"),
                    plt.ylabel("Productivity Gain"),
                    plt.grid(axis="y"),
                ),
            ),
        ),
        "roi_investment": chart_card(
            "ROI by Investment Quartile",
            chart_to_base64(
                "ROI by Investment Quartile",
                lambda: (
                    plt.bar(roi_by_investment.index, roi_by_investment),
                    plt.xlabel("AI Investment Quartile"),
                    plt.ylabel("ROI"),
                    plt.grid(axis="y"),
                ),
            ),
        ),
        "roi_investment_distribution": chart_card(
            "ROI Distribution by Investment Quartile",
            chart_to_base64(
                "ROI Distribution by Investment Quartile",
                lambda: (
                    df.boxplot(column="roi", by="invest_group", showfliers=False),
                    plt.suptitle(""),
                    plt.xlabel("AI Investment Quartile"),
                    plt.ylabel("ROI"),
                    plt.grid(axis="y"),
                ),
            ),
        ),
        "industry_net_value": chart_card(
            "Average Net Value by Industry",
            chart_to_base64(
                "Average Net Value by Industry",
                lambda: (
                    plt.barh(industry_net_value.index, industry_net_value),
                    plt.xlabel("Average Net Value USD Millions"),
                    plt.ylabel("Industry"),
                    plt.grid(axis="x"),
                ),
            ),
        ),
        "adoption_productivity": chart_card(
            "AI Adoption vs Productivity Gain",
            chart_to_base64(
                f"AI Adoption vs Productivity Gain | Correlation: {corr_value:.2f}",
                lambda: (
                    plt.scatter(yearly_corr["ai_adoption_level"], yearly_corr["productivity_gain"], alpha=0.8),
                    plt.xlabel("Average AI Adoption Level"),
                    plt.ylabel("Average Productivity Gain"),
                    plt.grid(True),
                ),
            ),
        ),
    }

    summary_cards = "\n".join(
        f"<div class='stat'><span>{label}</span><strong>{value}</strong></div>"
        for label, value in summary.items()
    )
    download_buttons = "\n".join(
        [
            download_link("full_dataset.csv", "Full Dataset"),
            download_link("summary.csv", "Summary"),
            download_link("yearly_adoption.csv", "Yearly Adoption"),
            download_link("industry_adoption.csv", "Industry Adoption"),
            download_link("country_adoption.csv", "Country Adoption"),
            download_link("investment_benefit_by_year.csv", "Investment Benefit"),
            download_link("industry_net_value.csv", "Industry Net Value"),
            download_link("deployment_by_training_quartile.csv", "Deployment Training"),
            download_link("roi_by_investment_quartile.csv", "ROI Investment"),
            download_link("adoption_productivity_by_year.csv", "Adoption Productivity"),
            download_link("industry_ai_profiles.csv", "Industry AI Profiles"),
            download_link("country_ai_profiles.csv", "Country AI Profiles"),
            download_link("industry_country_rates_by_amount.csv", "Industry Country Rates"),
            download_link("country_industry_rates_by_amount.csv", "Country Industry Rates"),
        ]
    )
    industry_years_json = json.dumps(industry_years)
    industry_entities_json = json.dumps(industry_entities)
    industry_values_json = json.dumps(industry_values)
    country_years_json = json.dumps(country_years)
    country_entities_json = json.dumps(country_entities)
    country_values_json = json.dumps(country_values)

    compare_section = """
        <section class="compare-section">
            <h2>Adoption Comparisons</h2>
            <div class="compare-grid">
                <section class="compare-card" id="industry-compare">
                    <h3>Industry Adoption Comparison</h3>
                    <div class="compare-controls">
                        <label>Industry 1<select data-role="first"></select></label>
                        <label>Industry 2<select data-role="second"></select></label>
                    </div>
                    <div class="compare-chart" data-role="chart"></div>
                    <div class="compare-legend" data-role="legend"></div>
                </section>
                <section class="compare-card" id="country-compare">
                    <h3>Country Adoption Comparison</h3>
                    <div class="compare-controls">
                        <label>Country 1<select data-role="first"></select></label>
                        <label>Country 2<select data-role="second"></select></label>
                    </div>
                    <div class="compare-chart" data-role="chart"></div>
                    <div class="compare-legend" data-role="legend"></div>
                </section>
            </div>
        </section>
    """

    def chart_grid(*keys):
        return f"<div class='tab-chart-grid'>{''.join(chart_cards[key] for key in keys)}</div>"

    tab_specs = [
        (
            "overview",
            "Overview",
            f"""
            <section class="stats">{summary_cards}</section>
            {chart_grid("yearly_adoption", "investment_benefit_year", "adoption_productivity")}
            """,
        ),
        (
            "adoption",
            "Adoption",
            f"""
            {compare_section}
            {chart_grid("industry_adoption", "country_adoption")}
            """,
        ),
        (
            "financial",
            "Financial",
            chart_grid("investment_benefit_scatter", "industry_net_value"),
        ),
        (
            "productivity",
            "Productivity",
            chart_grid(
                "automation_productivity",
                "deployment_training",
                "productivity_training_distribution",
            ),
        ),
        (
            "roi",
            "ROI",
            chart_grid(
                "roi_investment",
                "roi_investment_distribution",
                "roi_training_distribution",
            ),
        ),
        (
            "downloads",
            "Downloads",
            f"""
            <section class="downloads">
                <h2>Downloads</h2>
                <div class="download-list">{download_buttons}</div>
            </section>
            """,
        ),
    ]
    tab_buttons = "\n".join(
        f"<button class='dashboard-tab{' active' if index == 0 else ''}' type='button' data-tab='{tab_id}'>{label}</button>"
        for index, (tab_id, label, _content) in enumerate(tab_specs)
    )
    tab_panels = "\n".join(
        f"<section class='tab-panel{' active' if index == 0 else ''}' id='tab-{tab_id}'>{content}</section>"
        for index, (tab_id, _label, content) in enumerate(tab_specs)
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>AI Coverage Dashboard</title>
    <style>
        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            color: #17202a;
        }}
        header {{
            padding: 28px 32px 18px;
            background: #ffffff;
            border-bottom: 1px solid #d8dee4;
        }}
        h1 {{
            margin: 0 0 8px;
            font-size: 32px;
        }}
        p {{
            margin: 0;
            color: #5b6670;
        }}
        .top-links {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 14px;
        }}
        .top-links a {{
            display: inline-block;
            padding: 9px 12px;
            border: 1px solid #b7c2cc;
            border-radius: 6px;
            background: #ffffff;
            color: #17202a;
            text-decoration: none;
            font-size: 14px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            padding: 0;
            margin-bottom: 18px;
        }}
        .dashboard-shell {{
            padding: 20px 32px 32px;
        }}
        .dashboard-tabs {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 18px;
        }}
        .dashboard-tab {{
            border: 1px solid #b7c2cc;
            border-radius: 6px;
            background: #ffffff;
            color: #17202a;
            padding: 10px 13px;
            font-size: 14px;
            cursor: pointer;
        }}
        .dashboard-tab.active {{
            background: #17202a;
            border-color: #17202a;
            color: #ffffff;
        }}
        .tab-panel {{
            display: none;
        }}
        .tab-panel.active {{
            display: block;
        }}
        .downloads {{
            padding: 0;
        }}
        .compare-section {{
            padding: 0 0 18px;
        }}
        .compare-section h2 {{
            margin: 0 0 14px;
            font-size: 22px;
        }}
        .compare-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
            gap: 18px;
        }}
        .compare-card {{
            background: #ffffff;
            border: 1px solid #d8dee4;
            border-radius: 8px;
            padding: 16px;
        }}
        .compare-card h3 {{
            margin: 0 0 12px;
            font-size: 18px;
        }}
        .compare-controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 14px;
        }}
        .compare-controls label {{
            display: grid;
            gap: 5px;
            color: #5b6670;
            font-size: 13px;
        }}
        .compare-controls select {{
            min-width: 170px;
            padding: 8px 10px;
            border: 1px solid #b7c2cc;
            border-radius: 6px;
            background: #ffffff;
            color: #17202a;
        }}
        .compare-chart {{
            min-height: 240px;
            overflow-x: auto;
            padding: 10px 4px 0;
            border-top: 1px solid #d8dee4;
        }}
        .compare-line-svg {{
            min-width: 720px;
            width: 100%;
            height: 260px;
        }}
        .compare-axis-label {{
            color: #5b6670;
            font-size: 12px;
        }}
        .compare-legend {{
            display: flex;
            gap: 14px;
            margin-top: 12px;
            color: #5b6670;
            font-size: 13px;
        }}
        .legend-chip {{
            display: inline-block;
            width: 10px;
            height: 10px;
            margin-right: 6px;
            border-radius: 2px;
        }}
        .downloads h2 {{
            margin: 0 0 12px;
            font-size: 20px;
        }}
        .download-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .download {{
            display: inline-block;
            padding: 10px 12px;
            border: 1px solid #b7c2cc;
            border-radius: 6px;
            background: #ffffff;
            color: #17202a;
            text-decoration: none;
            font-size: 14px;
        }}
        .download:hover {{
            background: #eef3f7;
        }}
        .stat, .card {{
            background: #ffffff;
            border: 1px solid #d8dee4;
            border-radius: 8px;
        }}
        .stat {{
            padding: 14px 16px;
        }}
        .stat span {{
            display: block;
            color: #5b6670;
            font-size: 13px;
            margin-bottom: 6px;
        }}
        .stat strong {{
            font-size: 22px;
        }}
        .tab-chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
            gap: 18px;
        }}
        .card {{
            padding: 16px;
        }}
        .card-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 12px;
        }}
        .card h2 {{
            margin: 0;
            font-size: 18px;
        }}
        .image-download {{
            flex: 0 0 auto;
            padding: 8px 10px;
            border: 1px solid #b7c2cc;
            border-radius: 6px;
            background: #ffffff;
            color: #17202a;
            text-decoration: none;
            font-size: 13px;
        }}
        .image-download:hover {{
            background: #eef3f7;
        }}
        {DASHBOARD_THEME_CSS}
        {agent_widget_styles()}
        img {{
            display: block;
            width: 100%;
            height: auto;
        }}
        @media (max-width: 520px) {{
            header, .dashboard-shell {{
                padding-left: 14px;
                padding-right: 14px;
            }}
            .compare-grid {{
                grid-template-columns: 1fr;
            }}
            .tab-chart-grid {{
                grid-template-columns: 1fr;
            }}
            .card-header {{
                align-items: flex-start;
                flex-direction: column;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>AI Coverage Dashboard</h1>
        <p>One page for adoption, investment, productivity, deployment, ROI, and industry views.</p>
        <div class="top-links">
            <a href="industry_country_profiles.html">Industry and Country Profiles</a>
            <a href="regression_analysis.html">Regression Analysis</a>
        </div>
    </header>
    <div class="dashboard-shell">
        <nav class="dashboard-tabs" aria-label="Dashboard sections">
            {tab_buttons}
        </nav>
        {tab_panels}
    </div>
    {agent_widget_markup()}
    <script>
        {agent_widget_script(os.getenv("AGENT_API_BASE", ""))}
        document.querySelectorAll(".dashboard-tab").forEach((button) => {{
            button.addEventListener("click", () => {{
                document.querySelectorAll(".dashboard-tab").forEach((item) => item.classList.remove("active"));
                document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
                button.classList.add("active");
                document.getElementById(`tab-${{button.dataset.tab}}`).classList.add("active");
            }});
        }});

        function setupCompare(containerId, years, entities, values) {{
            const container = document.getElementById(containerId);
            const firstSelect = container.querySelector('[data-role="first"]');
            const secondSelect = container.querySelector('[data-role="second"]');
            const chart = container.querySelector('[data-role="chart"]');
            const legend = container.querySelector('[data-role="legend"]');

            entities.forEach((entity) => {{
                firstSelect.add(new Option(entity, entity));
                secondSelect.add(new Option(entity, entity));
            }});

            secondSelect.selectedIndex = Math.min(1, entities.length - 1);

            function render() {{
                const first = firstSelect.value;
                const second = secondSelect.value;
                const allValues = [...values[first], ...values[second]].filter((value) => value !== null);
                const maxValue = Math.max(...allValues, 1);
                const width = Math.max(720, years.length * 62);
                const height = 260;
                const left = 42;
                const right = 18;
                const top = 18;
                const bottom = 36;
                const plotWidth = width - left - right;
                const plotHeight = height - top - bottom;

                function pointsFor(entity) {{
                    return values[entity].map((value, index) => {{
                        const x = left + (index / Math.max(years.length - 1, 1)) * plotWidth;
                        const y = top + plotHeight - (((value ?? 0) / maxValue) * plotHeight);
                        return {{ x, y, value: value ?? 0, year: years[index] }};
                    }});
                }}

                function pathFor(points) {{
                    return points.map((point, index) => `${{index === 0 ? "M" : "L"}} ${{point.x}} ${{point.y}}`).join(" ");
                }}

                function dotsFor(points, color, entity) {{
                    return points.map((point) => `
                        <circle cx="${{point.x}}" cy="${{point.y}}" r="5.5" fill="${{color}}">
                            <title>${{entity}} ${{point.year}}: ${{point.value.toFixed(3)}}</title>
                        </circle>
                    `).join("");
                }}

                const firstPoints = pointsFor(first);
                const secondPoints = pointsFor(second);
                const yearLabels = years.map((year, index) => {{
                    const x = left + (index / Math.max(years.length - 1, 1)) * plotWidth;
                    return `<text x="${{x}}" y="${{height - 10}}" text-anchor="middle" class="compare-axis-label">${{year}}</text>`;
                }}).join("");

                chart.innerHTML = `
                    <svg class="compare-line-svg" viewBox="0 0 ${{width}} ${{height}}" role="img">
                        <line x1="${{left}}" y1="${{top}}" x2="${{left}}" y2="${{height - bottom}}" stroke="#b7c2cc" />
                        <line x1="${{left}}" y1="${{height - bottom}}" x2="${{width - right}}" y2="${{height - bottom}}" stroke="#b7c2cc" />
                        <text x="6" y="${{top + 8}}" class="compare-axis-label">${{maxValue.toFixed(2)}}</text>
                        <text x="12" y="${{height - bottom}}" class="compare-axis-label">0</text>
                        <path d="${{pathFor(firstPoints)}}" fill="none" stroke="#2f80ed" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" />
                        <path d="${{pathFor(secondPoints)}}" fill="none" stroke="#f2994a" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" />
                        ${{dotsFor(firstPoints, "#2f80ed", first)}}
                        ${{dotsFor(secondPoints, "#f2994a", second)}}
                        ${{yearLabels}}
                    </svg>
                `;

                legend.innerHTML = `
                    <span><span class="legend-chip" style="background:#2f80ed"></span>${{first}}</span>
                    <span><span class="legend-chip" style="background:#f2994a"></span>${{second}}</span>
                `;
            }}

            firstSelect.addEventListener("change", render);
            secondSelect.addEventListener("change", render);
            render();
        }}

        setupCompare(
            "industry-compare",
            {industry_years_json},
            {industry_entities_json},
            {industry_values_json}
        );
        setupCompare(
            "country-compare",
            {country_years_json},
            {country_entities_json},
            {country_values_json}
        );
        setupAgentWidget();
    </script>
</body>
</html>
"""
    OUTPUT_FILE.write_text(html, encoding="utf-8")
    write_regression_page()
    print(f"Dashboard created: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
