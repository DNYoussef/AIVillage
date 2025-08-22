# Connascence-Aware Linting Guide

## Overview

This guide explains how our ruff configuration maps to connascence concepts, helping developers understand why certain rules exist and how to fix violations to achieve weaker coupling.

## Connascence Forms and Detection

### Static Connascence (Compile-time dependencies)

#### 1. Connascence of Name (CoN)
**Definition**: Multiple components must agree on the name of an entity.

**Ruff Rules**:
- `F401` - Unused imports (creates unnecessary name dependencies)
- `F811` - Redefined unused name (name confusion)
- `F821` - Undefined name (broken name dependency)
- `N8*` - Naming conventions (consistent naming reduces connascence)
- `I001` - Import sorting (organized names reduce cognitive load)

**Examples**:
```python
# BAD: Strong connascence of name
from utils import calculate_tax as tax_calc
from billing import calculate_tax as billing_tax
# Multiple modules must agree on these specific names

# GOOD: Weaker connascence through consistent naming
from utils.tax import calculate_standard_tax
from billing.tax import calculate_billing_tax
# Clear, unambiguous names reduce coupling
```

**How to Fix**:
- Use descriptive, unambiguous names
- Follow consistent naming conventions
- Organize imports clearly
- Remove unused imports immediately

#### 2. Connascence of Type (CoT)
**Definition**: Multiple components must agree on the type of an entity.

**Ruff Rules**:
- `ANN*` - Type annotations (explicit types reduce assumptions)
- `FA100` - Missing `from __future__ import annotations`
- `UP006` - Use `typing.Union` instead of `|` for older Python

**Examples**:
```python
# BAD: Implicit type connascence
def process_data(data):  # What type is data?
    return data.transform()

# GOOD: Explicit type reduces connascence
def process_data(data: DataFrame) -> ProcessedData:
    return data.transform()

# BETTER: Generic types for flexibility
from typing import TypeVar, Protocol

T = TypeVar('T', bound='Transformable')

class Transformable(Protocol):
    def transform(self) -> 'ProcessedData': ...

def process_data(data: T) -> ProcessedData:
    return data.transform()
```

#### 3. Connascence of Meaning (CoM)
**Definition**: Multiple components must agree on the meaning of particular values.

**Ruff Rules**:
- `PLR2004` - Magic value used in comparison
- `FBT*` - Boolean trap (boolean parameters have unclear meaning)
- `SIM*` - Simplify expressions (complex expressions create meaning connascence)
- `D*` - Documentation (docstrings clarify meaning)

**Examples**:
```python
# BAD: Magic values create meaning connascence
def create_user(name: str, active: bool = True, role: int = 1):
    # What does role=1 mean? What about role=2?
    pass

# GOOD: Named constants reduce meaning connascence
from enum import Enum

class UserRole(Enum):
    ADMIN = 1
    USER = 2
    GUEST = 3

def create_user(name: str, active: bool = True, role: UserRole = UserRole.USER):
    """Create a user with specified role.

    Args:
        name: User's full name
        active: Whether user account is active
        role: User's permission level
    """
    pass

# BETTER: Remove boolean trap entirely
def create_active_user(name: str, role: UserRole = UserRole.USER):
    pass

def create_inactive_user(name: str, role: UserRole = UserRole.USER):
    pass
```

#### 4. Connascence of Algorithm (CoA)
**Definition**: Multiple components must agree on a particular algorithm.

**Ruff Rules**:
- `C901` - Function is too complex (high cyclomatic complexity)
- `PLR0912` - Too many branches
- `PLR0915` - Too many statements
- `C4*` - Comprehension improvements (consistent algorithms)

**Examples**:
```python
# BAD: Complex algorithm creates tight coupling
def calculate_price(item, user_type, season, discount_code):
    if user_type == "premium":
        base_price = item.price * 0.9
        if season == "winter":
            base_price *= 0.8
            if discount_code:
                if discount_code.startswith("WINTER"):
                    base_price *= 0.7
                elif discount_code.startswith("PREM"):
                    base_price *= 0.85
        elif season == "summer":
            # ... more complex logic
    elif user_type == "standard":
        # ... duplicate algorithm with variations
    return base_price

# GOOD: Separated concerns reduce algorithmic connascence
class PricingStrategy:
    def calculate_base_price(self, item: Item, user: User) -> Decimal:
        return item.price * user.discount_rate

    def apply_seasonal_discount(self, price: Decimal, season: Season) -> Decimal:
        return price * season.discount_multiplier

    def apply_promo_code(self, price: Decimal, code: PromoCode) -> Decimal:
        return price * code.discount_rate

def calculate_price(item: Item, user: User, season: Season,
                   promo: Optional[PromoCode] = None) -> Decimal:
    strategy = PricingStrategy()
    price = strategy.calculate_base_price(item, user)
    price = strategy.apply_seasonal_discount(price, season)
    if promo:
        price = strategy.apply_promo_code(price, promo)
    return price
```

### Dynamic Connascence (Runtime dependencies)

#### 5. Connascence of Execution (CoE)
**Definition**: The order of execution of multiple components is important.

**Ruff Rules**:
- `B026` - Star-arg unpacking after keyword argument
- `B904` - Use `raise ... from ...` to show exception cause
- `A*` - Builtins shadowing (execution order matters)

**Examples**:
```python
# BAD: Execution order creates tight coupling
class DatabaseManager:
    def __init__(self):
        self.connect()  # Must be called before other methods
        self.migrate()  # Must be called after connect
        self.setup_indexes()  # Must be called after migrate

# GOOD: Explicit initialization reduces execution connascence
class DatabaseManager:
    def __init__(self, connection_string: str):
        self._connection_string = connection_string
        self._connection: Optional[Connection] = None

    def initialize(self) -> None:
        """Initialize database connection and schema."""
        self._connect()
        self._migrate()
        self._setup_indexes()

    def _connect(self) -> None:
        if self._connection is None:
            self._connection = create_connection(self._connection_string)

    # Clear dependencies and error handling
```

#### 6. Connascence of Timing (CoTi)
**Definition**: The timing of the execution of multiple components is important.

**Ruff Rules**:
- `DTZ*` - Datetime timezone issues
- `B008` - Do not perform function calls in argument defaults

**Examples**:
```python
# BAD: Timing connascence in default arguments
def log_event(message: str, timestamp: datetime = datetime.now()):
    # timestamp is evaluated at function definition time!
    pass

# GOOD: Timing managed explicitly
def log_event(message: str, timestamp: Optional[datetime] = None):
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    # Clear timing behavior
```

#### 7. Connascence of Values (CoV)
**Definition**: Several values relate to one another and must change together.

**Ruff Rules**:
- `RET*` - Return consistency (related return values)
- `SIM*` - Expression simplification (related value calculations)

**Examples**:
```python
# BAD: Values must be kept in sync manually
class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.area = width * height  # Must update when width/height change
        self.perimeter = 2 * (width + height)  # Must update too

# GOOD: Computed properties remove value connascence
class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def perimeter(self) -> float:
        return 2 * (self.width + self.height)
```

#### 8. Connascence of Identity (CoI)
**Definition**: Multiple components must reference the same entity.

**Ruff Rules**:
- `E711` - Comparison to None should be `is` or `is not`
- `E712` - Comparison to True should be `is` or `is not`

**Examples**:
```python
# BAD: Identity confusion
if user.status == None:  # Should use 'is'
    pass

if user.active == True:  # Should use direct check
    pass

# GOOD: Clear identity handling
if user.status is None:
    pass

if user.active:  # Direct boolean check
    pass
```

## Architectural Layer Rules

### Core/Domain Layer (Strictest)
- **Purpose**: Business logic should have minimal connascence
- **Rules**: All rules apply with minimal exceptions
- **Files**: `packages/core/**/*.py`, `**/interfaces/**/*.py`

### Application Layer (Moderate)
- **Purpose**: Coordinate domain objects, some complexity acceptable
- **Relaxed Rules**: `PLR0913` (more parameters), `PLR0912` (more branches)
- **Files**: `packages/agents/specialized/**/*.py`

### Infrastructure Layer (Relaxed)
- **Purpose**: Handle external concerns, complexity often unavoidable
- **Relaxed Rules**: Security rules (`S*`), complexity rules (`PLR*`)
- **Files**: `packages/automation/**/*.py`, `packages/p2p/**/*.py`

### Test Layer (Most Relaxed)
- **Purpose**: Verify behavior, connascence less critical
- **Relaxed Rules**: Most rules except critical errors
- **Files**: `tests/**/*.py`, `**/test_*.py`

## Common Violation Patterns and Fixes

### Pattern: Too Many Parameters (`PLR0913`)
```python
# BAD: Parameter explosion
def create_user(name, email, phone, address_line1, address_line2,
                city, state, zip_code, country, birth_date, gender):
    pass

# GOOD: Parameter object
@dataclass
class UserInfo:
    name: str
    email: str
    phone: str
    address: Address
    birth_date: date
    gender: Optional[str] = None

@dataclass
class Address:
    line1: str
    line2: Optional[str]
    city: str
    state: str
    zip_code: str
    country: str

def create_user(user_info: UserInfo):
    pass
```

### Pattern: Boolean Trap (`FBT001`)
```python
# BAD: Boolean parameters are unclear
def send_notification(message: str, urgent: bool, email: bool):
    pass

# GOOD: Explicit enums or separate methods
from enum import Enum

class Priority(Enum):
    NORMAL = "normal"
    URGENT = "urgent"

class Channel(Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"

def send_notification(message: str, priority: Priority, channel: Channel):
    pass

# ALTERNATIVE: Separate methods
def send_urgent_email(message: str):
    pass

def send_normal_email(message: str):
    pass
```

### Pattern: Magic Values (`PLR2004`)
```python
# BAD: Magic numbers/strings
def calculate_discount(amount: float, customer_type: int):
    if customer_type == 1:  # What is 1?
        return amount * 0.1  # What is 0.1?
    elif customer_type == 2:
        return amount * 0.15
    return 0

# GOOD: Named constants
from enum import Enum

class CustomerType(Enum):
    STANDARD = 1
    PREMIUM = 2
    VIP = 3

DISCOUNT_RATES = {
    CustomerType.STANDARD: 0.10,
    CustomerType.PREMIUM: 0.15,
    CustomerType.VIP: 0.20,
}

def calculate_discount(amount: float, customer_type: CustomerType) -> float:
    discount_rate = DISCOUNT_RATES.get(customer_type, 0.0)
    return amount * discount_rate
```

## Best Practices Summary

1. **Name Connascence**: Use clear, consistent naming conventions
2. **Type Connascence**: Add type hints and use proper type hierarchies
3. **Meaning Connascence**: Eliminate magic values, add documentation
4. **Algorithm Connascence**: Break down complex functions, use strategy pattern
5. **Execution Connascence**: Make execution order explicit and safe
6. **Timing Connascence**: Handle time-dependent operations carefully
7. **Value Connascence**: Use computed properties and validation
8. **Identity Connascence**: Use proper identity comparisons

## Configuration Testing

Test your configuration with:
```bash
# Check specific file
ruff check packages/core/some_file.py

# Check with explanations
ruff check --explain PLR0913

# Show all rules
ruff linter

# Test configuration
ruff check --select ALL --ignore "" packages/core/
```

Remember: The goal is not to eliminate all connascence (impossible) but to identify and weaken the strongest forms first, moving from dynamic to static, and from broader to narrower scope.
