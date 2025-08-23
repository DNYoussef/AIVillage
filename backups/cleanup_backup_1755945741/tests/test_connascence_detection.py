"""Test file demonstrating connascence detection with our ruff configuration.

This file contains examples of various connascence forms that should be detected
by our enhanced ruff configuration. Each example shows both problematic code
and the improved version.
"""

import datetime

# === CONNASCENCE OF NAME (CoN) EXAMPLES ===
# BAD: Unused import creates unnecessary name dependency (F401)


# BAD: Redefined name (F811)
def calculate_tax(amount):
    return amount * 0.1


def calculate_tax(amount, rate):  # Redefinition should be detected
    return amount * rate


# === CONNASCENCE OF EXECUTION (CoE) EXAMPLES ===


# BAD: Function call in default argument (B008)
def log_message(message: str, timestamp=datetime.datetime.now()):
    """This should trigger B008 - function call in default argument."""
    print(f"{timestamp}: {message}")


# GOOD: Proper default handling
def log_message_good(message: str, timestamp=None):
    """Proper way to handle mutable defaults."""
    if timestamp is None:
        timestamp = datetime.datetime.now()
    print(f"{timestamp}: {message}")


# === CONNASCENCE OF ALGORITHM (CoA) EXAMPLES ===


# BAD: High cyclomatic complexity (C901)
def complex_pricing_logic(item_type, user_level, season, promo_code, quantity):
    """This function should trigger C901 for high complexity."""
    base_price = 100

    if item_type == "premium":
        if user_level == "gold":
            if season == "winter":
                if promo_code:
                    if promo_code.startswith("WINTER"):
                        if quantity > 10:
                            discount = 0.3
                        elif quantity > 5:
                            discount = 0.2
                        else:
                            discount = 0.1
                    elif promo_code.startswith("GOLD"):
                        if quantity > 15:
                            discount = 0.25
                        else:
                            discount = 0.15
                    else:
                        discount = 0.05
                else:
                    discount = 0.02
            elif season == "summer":
                if promo_code:
                    if promo_code.startswith("SUMMER"):
                        discount = 0.15
                    else:
                        discount = 0.08
                else:
                    discount = 0.03
            else:
                discount = 0.01
        elif user_level == "silver":
            # More nested logic...
            if season == "winter":
                discount = 0.05
            else:
                discount = 0.02
        else:
            discount = 0.01
    elif item_type == "standard":
        # Even more logic...
        if user_level == "gold":
            discount = 0.1
        else:
            discount = 0.05
    else:
        discount = 0

    return base_price * (1 - discount)


# GOOD: Simplified with strategy pattern
class PricingStrategy:
    """Better approach using strategy pattern."""

    @staticmethod
    def calculate_base_discount(user_level: str) -> float:
        """Calculate base discount based on user level."""
        discounts = {"gold": 0.1, "silver": 0.05, "bronze": 0.02}
        return discounts.get(user_level, 0.0)

    @staticmethod
    def calculate_seasonal_discount(season: str) -> float:
        """Calculate seasonal discount."""
        discounts = {"winter": 0.05, "summer": 0.03, "spring": 0.02}
        return discounts.get(season, 0.0)


def calculate_price_good(item_type: str, user_level: str, season: str) -> float:
    """Simplified pricing calculation."""
    base_price = 100
    strategy = PricingStrategy()

    base_discount = strategy.calculate_base_discount(user_level)
    seasonal_discount = strategy.calculate_seasonal_discount(season)

    total_discount = min(base_discount + seasonal_discount, 0.3)  # Cap at 30%
    return base_price * (1 - total_discount)


# === CONNASCENCE OF TYPE (CoT) EXAMPLES ===


# BAD: Missing type annotations
def process_items(items):  # Should have type hints
    """Process items without type information."""
    result = []
    for item in items:
        result.append(item.upper())
    return result


# GOOD: Proper type annotations
def process_items_good(items: list[str]) -> list[str]:
    """Process items with proper type information."""
    return [item.upper() for item in items]


# === IMPORT ORGANIZATION (CoN) ===
# The imports at the top should be sorted by isort (I001)


def get_users() -> set[str]:
    """Function using Set type annotation."""
    return {"user1", "user2", "user3"}


if __name__ == "__main__":
    # Test the functions
    print(calculate_price_good("premium", "gold", "winter"))
    print(process_items_good(["hello", "world"]))
    print(get_users())
