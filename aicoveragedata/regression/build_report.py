import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from aicoveragedata.regression.report.page import write_page


if __name__ == "__main__":
    print(write_page())
