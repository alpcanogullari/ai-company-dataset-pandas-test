import argparse
from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from aicoveragedata.agent.context.tools import classify_context_task, collect_context
    from aicoveragedata.agent.core.agent import answer_question
    from aicoveragedata.agent.core.config import AgentConfig
    from aicoveragedata.agent.evaluation.cases import CASES
else:
    from ..context.tools import classify_context_task, collect_context
    from ..core.agent import answer_question
    from ..core.config import AgentConfig
    from .cases import CASES


def contains_all(text, required):
    lowered = text.lower()
    return [item for item in required if item.lower() not in lowered]


def check_case(case, live=False):
    question = case["question"]
    history = case.get("history", [])
    route = classify_context_task(question, history)
    context = collect_context(question, history_messages=history, task=route)

    missing_context = contains_all(context, case.get("required_context", []))
    route_ok = route == case["expected_route"]

    answer = ""
    missing_answer = []
    if live:
        answer, _ = answer_question(
            question,
            config=AgentConfig.from_env(),
            use_openai=True,
            session_id=f"eval-{case['id']}",
            remember=False,
        )
        missing_answer = contains_all(answer, case.get("required_answer", []))

    passed = route_ok and not missing_context and not missing_answer
    return {
        "id": case["id"],
        "passed": passed,
        "route": route,
        "expected_route": case["expected_route"],
        "missing_context": missing_context,
        "missing_answer": missing_answer,
        "answer": answer,
    }


def run(live=False):
    results = [check_case(case, live=live) for case in CASES]
    passed = sum(1 for result in results if result["passed"])

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} {result['id']}")
        if result["route"] != result["expected_route"]:
            print(f"  route: {result['route']} expected {result['expected_route']}")
        if result["missing_context"]:
            print(f"  missing context: {', '.join(result['missing_context'])}")
        if result["missing_answer"]:
            print(f"  missing answer: {', '.join(result['missing_answer'])}")
        if live and result["answer"]:
            print(f"  answer: {result['answer'][:240]}")

    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


def main():
    parser = argparse.ArgumentParser(description="Run local evaluation checks for the AI coverage agent.")
    parser.add_argument("--live", action="store_true", help="Call the configured model and check answer text.")
    args = parser.parse_args()
    raise SystemExit(run(live=args.live))


if __name__ == "__main__":
    main()
