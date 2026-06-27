"""Feed a mixed batch of notifications through the triage model and print the
ranked inbox with priority scores.
"""

from triage import Notification, train_default

BATCH = [
    Notification(
        "Security alert: your account was accessed from a new device. "
        "If this wasn't you, act immediately",
        "security",
    ),
    Notification("Your password reset link expires in 30 minutes", "security"),
    Notification(
        "Weekly newsletter: 5 articles we think you'll enjoy this week",
        "marketing",
    ),
    Notification("Maria liked your photo", "social"),
    Notification("Reminder: dentist appointment tomorrow at 10am", "calendar"),
    Notification(
        "URGENT: production is down, customers cannot check out, respond now",
        "ops",
    ),
    Notification("50% off everything this weekend only, shop the sale", "marketing"),
    Notification("Your package will be delivered today between 2pm and 6pm", "orders"),
]


def main() -> None:
    model = train_default()
    ranked = model.rank(BATCH)

    print("Ranked inbox (highest priority first)\n")
    print(f"{'#':>2}  {'score':>5}  {'tier':<7}  message")
    print("-" * 78)
    for r in ranked:
        text = r.notification.text
        if len(text) > 52:
            text = text[:49] + "..."
        print(
            f"{r.rank:>2}  {r.score:>5.3f}  {r.predicted_priority:<7}  {text}"
        )


if __name__ == "__main__":
    main()
