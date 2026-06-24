CASES = [
    {
        "id": "lowest_country_adoption",
        "question": "Which country has the lowest average AI adoption?",
        "expected_route": "dataset_ranking",
        "required_context": ["Lowest", "country", "ai_adoption_level"],
        "required_answer": ["ai_adoption_level"],
    },
    {
        "id": "highest_followup_roi",
        "history": [
            {"role": "user", "content": "Which country has the lowest average ROI?"},
            {"role": "assistant", "content": "The lowest average ROI is Brazil at 0.7634."},
        ],
        "question": "What about the highest?",
        "expected_route": "dataset_ranking",
        "required_context": ["Top", "United States", "roi"],
        "required_answer": ["United States"],
    },
    {
        "id": "adoption_xgboost_target_alias",
        "question": "Explain the supervised XGBoost tree that targets ai_adoption_rate.",
        "expected_route": "regression_model",
        "required_context": ["ai_adoption_rate", "ai_adoption_level", "Adoption XGBoost statistics"],
        "required_answer": ["ai_adoption_level"],
    },
    {
        "id": "dataset_variables",
        "question": "What variables are in the dataset?",
        "expected_route": "dataset_schema",
        "required_context": ["Available dataset columns", "revenue_impact", "ai_adoption_level"],
        "required_answer": ["revenue_impact", "ai_adoption_level"],
    },
    {
        "id": "chatbot_clear_button",
        "question": "How does the chatbot clear button work?",
        "expected_route": "codebase_html",
        "required_context": ["agent_widget.py", "clear", "button"],
        "required_answer": ["clear"],
    },
]

