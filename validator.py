VALID_STATUSES = {
    "matched",
    "partially_matched",
    "missing",
    "not_verified"
}

REQUIRED_CATEGORIES = {
    "skills",
    "experience",
    "domain",
    "education"
}


def validate_structure(match_data):

    errors = []

    # Check top-level categories
    for category in REQUIRED_CATEGORIES:

        if category not in match_data:
            errors.append(
                f"Missing category: {category}"
            )
            continue

        category_data = match_data[category]

        if not isinstance(category_data, dict):
            errors.append(
                f"{category} must be an object"
            )
            continue

        # Check status fields
        for status in VALID_STATUSES:

            if status not in category_data:
                errors.append(
                    f"Missing field: {category}.{status}"
                )

            elif not isinstance(category_data[status], list):
                errors.append(
                    f"{category}.{status} must be a list"
                )

    # Check evidence
    if "evidence" not in match_data:
        errors.append("Missing field: evidence")
    elif not isinstance(match_data["evidence"], list):
        errors.append("evidence must be a list")

    # Check gaps
    if "gaps" not in match_data:
        errors.append("Missing field: gaps")
    elif not isinstance(match_data["gaps"], list):
        errors.append("gaps must be a list")

    return errors


# ADD THIS FUNCTION HERE
def validate_duplicate_status(match_data):

    errors = []

    for category in REQUIRED_CATEGORIES:

        category_data = match_data.get(category, {})

        requirement_statuses = {}

        for status in VALID_STATUSES:

            requirements = category_data.get(status, [])

            for requirement in requirements:

                if requirement in requirement_statuses:

                    previous_status = requirement_statuses[requirement]

                    errors.append(
                        f"Requirement '{requirement}' appears in "
                        f"both {previous_status} and {status} "
                        f"under {category}"
                    )

                else:
                    requirement_statuses[requirement] = status

    return errors


def validate_match_data(match_data):

    errors = []

    errors.extend(
        validate_structure(match_data)
    )

    errors.extend(
        validate_duplicate_status(match_data)
    )

    return errors