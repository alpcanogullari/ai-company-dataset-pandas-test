# AI Coverage Dashboard Agent Context

## Dataset

- Source: corporate AI adoption and ROI dataset, 2015-2035.
- Rows: 200,000.
- Companies: 8,000.
- Industries: 10.
- Countries: 15.
- Variables: 16 core variables plus generated dashboard buckets.

## Columns

company_id, industry, country, year, ai_adoption_level, ai_investment_usd, automation_rate, cost_savings, revenue_impact, productivity_gain, employee_ai_training_hours, ai_maturity_score, deployment_count, total_benefit, net_value, roi, training_group, invest_group.

## Adoption Trend

- 2015 average AI adoption: 0.1915.
- 2020 average AI adoption: 0.3539.
- 2025 average AI adoption: 0.5295.
- 2030 average AI adoption: 0.7007.
- 2035 average AI adoption: 0.8578.

## Top Industry AI Scores

1. Energy: final AI score 71.4, adoption 0.5316, automation 0.4923, productivity 0.4266, ROI 0.6751.
2. Manufacturing: final AI score 62.2, adoption 0.5280, automation 0.5222, productivity 0.4425, ROI 0.6407.
3. Financial Services: final AI score 60.5, adoption 0.5291, automation 0.4560, productivity 0.4063, ROI 0.7854.
4. Technology: final AI score 60.2, adoption 0.5257, automation 0.4185, productivity 0.3847, ROI 1.3125.
5. Agriculture: final AI score 60.0, adoption 0.5300, automation 0.4564, productivity 0.4064, ROI 0.5771.

## Lower Industry AI Scores

- Telecom: final AI score 46.3.
- Healthcare: final AI score 45.5.
- Education: final AI score 3.2.

## Top Country AI Scores

1. United States: final AI score 94.6, adoption 0.5469, automation 0.4558, productivity 0.4152, ROI 0.8309.
2. United Kingdom: final AI score 80.5, adoption 0.5342, automation 0.4425, productivity 0.4019, ROI 0.8233.
3. Sweden: final AI score 75.7, adoption 0.5290, automation 0.4393, productivity 0.3977, ROI 0.8197.
4. Singapore: final AI score 75.6, adoption 0.5354, automation 0.4407, productivity 0.4020, ROI 0.8193.
5. China: final AI score 65.7, adoption 0.5293, automation 0.4376, productivity 0.3959, ROI 0.8139.

## Lower Country AI Scores

- India: final AI score 37.7.
- UAE: final AI score 27.5.
- Brazil: final AI score 0.1.

## Regression Summary

- Target: revenue_impact.
- Main predictors: ai_investment_usd, automation_rate, cost_savings, employee_ai_training_hours, deployment_count.
- Selected model: Stacked Ensemble Regression.
- Test R Square: 0.5268.
- Test RMSE: 2,790,544.15.
- Test MAE: 1,182,785.13.
- Training observations: 140,000.
- Test observations: 60,000.

## Site Structure

- Dashboard entrypoint: aicoveragedata/site/index.html.
- Industry and country profiles: aicoveragedata/site/industry_country_profiles.html.
- Regression report: aicoveragedata/site/regression_analysis.html.
- AI agent endpoint on Netlify: /api/agent, implemented by netlify/functions/agent.mjs.

## Answering Rules

- Be concise.
- Use exact numbers from this context when available.
- If the user asks for details not present here, say the deployed Netlify function has summary context only.
- Do not claim to run live Python, pandas, DSPy, or local filesystem analysis in Netlify.
