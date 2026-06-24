from aicoveragedata.data import load_dataset


def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def print_dataset_coverage():
    df = load_dataset()

    observations, variables = df.shape
    section("Dataset Dimensions")
    print(f"Observations: {observations}".center(35))
    print(f"Variables: {variables}".center(35))

    section("Coverage")
    print(f"Years Covered: {df['year'].min()} - {df['year'].max()} ".center(20))
    print(f"Individual Companies: {df['company_id'].nunique()}".center(20))
    print(f"Individual Industries: {df['industry'].nunique()}".center(20))
    print(f"Individual Countries: {df['country'].nunique()}".center(20))

    section("Industries Covered")
    for industry in df["industry"].unique().tolist():
        print(f"- {industry}".center(30))

    section("Countries Covered")
    for country in df["country"].unique().tolist():
        print(f"- {country}".center(30))


if __name__ == "__main__":
    print_dataset_coverage()
