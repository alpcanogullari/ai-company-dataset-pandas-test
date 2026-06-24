import argparse
from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from aicoveragedata.agent.core.agent import answer_question
    from aicoveragedata.agent.core.config import AgentConfig
else:
    from ..core.agent import answer_question
    from ..core.config import AgentConfig


def build_parser():
    parser = argparse.ArgumentParser(description="Ask questions about the AI coverage dataset.")
    parser.add_argument("-q", "--question", help="Question to ask. If omitted, starts interactive mode.")
    parser.add_argument("--offline", action="store_true", help="Use local dataset tools only. Do not call OpenAI.")
    parser.add_argument("--show-context", action="store_true", help="Print the local context used for the answer.")
    return parser


def run_once(question, offline=False, show_context=False):
    answer, context = answer_question(
        question,
        config=AgentConfig.from_env(),
        use_openai=not offline,
        include_context=False,
    )
    print(answer)
    if show_context:
        print("\n---\nContext used:\n")
        print(context)


def run_interactive(offline=False, show_context=False):
    print("AI Coverage Data Agent")
    print("Type a question, or type 'exit' to stop.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        run_once(question, offline=offline, show_context=show_context)


def main():
    args = build_parser().parse_args()
    if args.question:
        run_once(args.question, offline=args.offline, show_context=args.show_context)
    else:
        run_interactive(offline=args.offline, show_context=args.show_context)


if __name__ == "__main__":
    main()
