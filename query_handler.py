def generate_queries(user_input):
    topic = user_input.strip().lower()

    # If user doesn't mention market/forecast, enforce it
    market_keywords = ["market", "forecast", "industry", "outlook", "cagr"]
    if not any(k in topic for k in market_keywords):
        topic = topic + " market"

    templates = [
        f"{topic} size forecast report",
        f"{topic} industry outlook CAGR projection",
        f"{topic} market research report filetype:pdf",
        f"{topic} revenue forecast analysis"
    ]

    return templates
