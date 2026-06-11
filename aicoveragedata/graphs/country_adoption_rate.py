import json
from pathlib import Path

import pandas as pd

from data_utils import load_dataset


OUTPUT_FILE = Path(__file__).with_suffix(".html")


def compare_payload(table):
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


df = load_dataset()

# Average AI adoption by country for each year.
country_adoption_table = df.groupby(["year", "country"])["ai_adoption_level"].mean().unstack()
years, countries, values = compare_payload(country_adoption_table)

html = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Country Adoption Comparison</title>
    <style>
        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            color: #17202a;
        }}
        .compare-card {{
            margin: 24px;
            background: #ffffff;
            border: 1px solid #d8dee4;
            border-radius: 8px;
            padding: 16px;
        }}
        h1 {{
            margin: 0 0 14px;
            font-size: 22px;
        }}
        .controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 14px;
        }}
        label {{
            display: grid;
            gap: 5px;
            color: #5b6670;
            font-size: 13px;
        }}
        select {{
            min-width: 180px;
            padding: 8px 10px;
            border: 1px solid #b7c2cc;
            border-radius: 6px;
            background: #ffffff;
            color: #17202a;
        }}
        .chart {{
            min-height: 240px;
            overflow-x: auto;
            padding-top: 10px;
            border-top: 1px solid #d8dee4;
        }}
        .line-svg {{
            min-width: 720px;
            width: 100%;
            height: 260px;
        }}
        .axis-label, .point-label, .legend {{
            color: #5b6670;
            font-size: 13px;
        }}
        .legend {{
            display: flex;
            gap: 14px;
            margin-top: 12px;
        }}
        .chip {{
            display: inline-block;
            width: 10px;
            height: 10px;
            margin-right: 6px;
            border-radius: 2px;
        }}
    </style>
</head>
<body>
    <section class="compare-card">
        <h1>Country Adoption Comparison</h1>
        <div class="controls">
            <label>Country 1<select id="first"></select></label>
            <label>Country 2<select id="second"></select></label>
        </div>
        <div class="chart" id="chart"></div>
        <div class="legend" id="legend"></div>
    </section>
    <script>
        const years = {json.dumps(years)};
        const entities = {json.dumps(countries)};
        const values = {json.dumps(values)};
        const firstSelect = document.getElementById("first");
        const secondSelect = document.getElementById("second");
        const chart = document.getElementById("chart");
        const legend = document.getElementById("legend");

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
                return `<text x="${{x}}" y="${{height - 10}}" text-anchor="middle" class="axis-label">${{year}}</text>`;
            }}).join("");

            chart.innerHTML = `
                <svg class="line-svg" viewBox="0 0 ${{width}} ${{height}}" role="img">
                    <line x1="${{left}}" y1="${{top}}" x2="${{left}}" y2="${{height - bottom}}" stroke="#b7c2cc" />
                    <line x1="${{left}}" y1="${{height - bottom}}" x2="${{width - right}}" y2="${{height - bottom}}" stroke="#b7c2cc" />
                    <text x="6" y="${{top + 8}}" class="axis-label">${{maxValue.toFixed(2)}}</text>
                    <text x="12" y="${{height - bottom}}" class="axis-label">0</text>
                    <path d="${{pathFor(firstPoints)}}" fill="none" stroke="#2f80ed" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" />
                    <path d="${{pathFor(secondPoints)}}" fill="none" stroke="#f2994a" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" />
                    ${{dotsFor(firstPoints, "#2f80ed", first)}}
                    ${{dotsFor(secondPoints, "#f2994a", second)}}
                    ${{yearLabels}}
                </svg>
            `;

            legend.innerHTML = `
                <span><span class="chip" style="background:#2f80ed"></span>${{first}}</span>
                <span><span class="chip" style="background:#f2994a"></span>${{second}}</span>
            `;
        }}

        firstSelect.addEventListener("change", render);
        secondSelect.addEventListener("change", render);
        render();
    </script>
</body>
</html>
"""

OUTPUT_FILE.write_text(html, encoding="utf-8")
print(f"Dashboard chart created: {OUTPUT_FILE}")
