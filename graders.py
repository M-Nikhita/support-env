def grade(task, action):
    score = 0

    # Category (30%)
    if action.get("category") == task.get("expected_category"):
        score += 0.3

    # Priority (20%)
    if "expected_priority" in task:
        if action.get("priority") == task.get("expected_priority"):
            score += 0.2

    # Resolution (20%)
    if "expected_resolution" in task:
        if action.get("resolution") == task.get("expected_resolution"):
            score += 0.2

    # Response quality (30%)
    if "keywords" in task:
        response = action.get("response", "").lower()
        match = sum(1 for word in task["keywords"] if word in response)
        score += (match / len(task["keywords"])) * 0.3

        # bonus for polite tone
        if "sorry" in response or "apologize" in response:
            score += 0.05

    return min(1.0, round(score, 2))