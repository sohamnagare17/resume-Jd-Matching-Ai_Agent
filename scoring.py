STATUS_SCORES = {
    "MATCHED": 1.0,
    "PARTIALLY_MATCHED": 0.5,
    "NOT_VERIFIED": 0.25,
    "MISSING": 0.0
}


def calculate_category_score(category_data):
    total_requirements = 0
    total_points = 0.0

    for status, points in STATUS_SCORES.items():
        requirements = category_data.get(
            status.lower(),
            []
        )

        total_requirements += len(requirements)
        total_points += len(requirements) * points

    if total_requirements == 0:
        return 0.0

    return (total_points / total_requirements) * 100


def calculate_overall_score(match_data):

    categories = [
        "skills",
        "experience",
        "domain",
        "education"
    ]

    total_requirements = 0
    total_points = 0.0

    for category in categories:

        category_data = match_data.get(category, {})

        for status, points in STATUS_SCORES.items():

            requirements = category_data.get(
                status.lower(),
                []
            )

            total_requirements += len(requirements)
            total_points += len(requirements) * points

    if total_requirements == 0:
        return 0.0

    return round(
        (total_points / total_requirements) * 100,
        2
    )