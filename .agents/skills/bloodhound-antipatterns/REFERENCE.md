# Bloodhound Reference

> Full catalog: 22 code smells + ~50 anti-patterns.
>
> Each entry: Stink (signals) → Why it hurts → Fix → Doc → Example

---

# Part I: Code Smells

From [refactoring.guru](https://refactoring.guru/refactoring/smells)

---

## Bloaters

### Long Method

**Stink:** Method > 10 lines. Nested loops, conditionals, or try/except blocks. Comments explaining "what this block does" — a sign it should be its own method.

**Why it hurts:** Hard to read, test, or reuse. Hides the execution logic behind ceremony.

**Fix:** Extract Method, Replace Temp with Query, Introduce Parameter Object, Preserve Whole Object

**Doc:** https://refactoring.guru/smells/long-method

```python
# Stinky
def process_order(order: dict) -> None:
    total = 0.0
    for item in order["items"]:
        if item["qty"] > 0:
            subtotal = item["price"] * item["qty"]
            if subtotal > 100:
                subtotal *= 0.9  # 10% discount
            total += subtotal
    tax = total * 0.16
    total_with_tax = total + tax
    db.execute("INSERT INTO orders (total, tax) VALUES (?, ?)", (total_with_tax, tax))

# Clean
def process_order(order: dict) -> None:
    total = calculate_total(order["items"])
    tax = calculate_tax(total)
    persist_order(total, tax)

def calculate_total(items: list) -> float: ...
def calculate_tax(total: float) -> float: ...
def persist_order(total: float, tax: float) -> None: ...
```

---

### Large Class

**Stink:** Class > 200 lines. Too many fields (10+). Too many methods (15+). The class name is vague because it does too much (e.g., `Manager`, `Service`, `Handler`).

**Why it hurts:** Violates Single Responsibility Principle. One change reason forces you to touch unrelated code. Testing requires mocking half the universe.

**Fix:** Extract Class, Extract Subclass, Extract Interface

**Doc:** https://refactoring.guru/smells/large-class

```python
# Stinky
class OrderService:
    def create_order(self, ...): ...
    def calculate_taxes(self, ...): ...
    def send_email(self, ...): ...
    def generate_invoice_pdf(self, ...): ...
    def update_inventory(self, ...): ...
    def refund(self, ...): ...
    def apply_coupon(self, ...): ...
    def get_shipping_cost(self, ...): ...

# Clean
class OrderCreator: ...
class TaxCalculator: ...
class InvoiceService: ...
class InventoryUpdater: ...
```

---

### Primitive Obsession

**Stink:** Using `str`, `int`, `float`, `dict` for domain concepts that have constraints, validation rules, or behavior. E.g., `user_id: str`, `amount: float`, `currency: str`, `email: str`.

**Why it hurts:** No type safety — passing args in wrong order is silent. Validation is duplicated everywhere instead of centralized in the type. Makes illegal states representable.

**Fix:** Replace Data Value with Object, Introduce Parameter Object, Replace Type Code with Class, Use NewType/Pydantic branded types

**Doc:** https://refactoring.guru/smells/primitive-obsession

```python
# Stinky
def transfer(sender_id: str, receiver_id: str, amount: float, currency: str): ...

# Clean
from typing import NewType
UserId = NewType("UserId", str)
Amount = NewType("Amount", float)
Currency = NewType("Currency", str)

def transfer(sender_id: UserId, receiver_id: UserId, amount: Amount, currency: Currency): ...
```

---

### Long Parameter List

**Stink:** Method with 3+ parameters. Callers must remember the order. Every new feature adds another parameter.

**Why it hurts:** Hard to read call sites. Hard to add new parameters without breaking callers. Promotes parameter object patterns — but the LLM won't do it unless told.

**Fix:** Introduce Parameter Object, Preserve Whole Object, Replace Parameter with Method Call

**Doc:** https://refactoring.guru/smells/long-parameter-list

```python
# Stinky
def create_user(name, email, age, address, phone, plan, referrer): ...

# Clean
@dataclass
class UserRegistration:
    name: str
    email: str
    age: int
    address: str
    phone: str
    plan: str
    referrer: str | None

def create_user(registration: UserRegistration): ...
```

---

### Data Clumps

**Stink:** The same 2-3 fields appear together in multiple method signatures, data structures, or parameter lists. E.g., `(street, city, zip)` repeated everywhere. Or `(start_date, end_date)`.

**Why it hurts:** Duplication of grouping logic. If you add a field (e.g., `country`), you must update every occurrence.

**Fix:** Extract Class, Introduce Parameter Object, Preserve Whole Object

**Doc:** https://refactoring.guru/smells/data-clumps

```python
# Stinky
def ship_to(street: str, city: str, zip_code: str): ...
def bill_to(street: str, city: str, zip_code: str): ...
def validate_address(street: str, city: str, zip_code: str): ...

# Clean
@dataclass
class Address:
    street: str
    city: str
    zip_code: str

def ship_to(address: Address): ...
def bill_to(address: Address): ...
def validate_address(address: Address): ...
```

---

## Object-Orientation Abusers

### Switch Statements (or long if/elif chains)

**Stink:** Type-checking cascades: `if isinstance(x, A): ... elif isinstance(x, B): ... elif ...` Or long `match/case` blocks dispatching on a type discriminator. Every new type means adding a new branch.

**Why it hurts:** Open/Closed Principle violation — adding a new type requires modifying existing code. Same switch duplicated in multiple places.

**Fix:** Replace Conditional with Polymorphism, Replace Type Code with Strategy/State, Extract Method

**Doc:** https://refactoring.guru/smells/switch-statements

```python
# Stinky
def calculate_payment(method: str, amount: float) -> float:
    if method == "credit_card":
        return amount * 1.03
    elif method == "debit":
        return amount * 1.01
    elif method == "paypal":
        return amount * 1.05
    elif method == "crypto":
        return amount * 1.02

# Clean
from abc import ABC, abstractmethod

class PaymentMethod(ABC):
    @abstractmethod
    def fee(self, amount: float) -> float: ...

class CreditCard(PaymentMethod):
    def fee(self, amount: float) -> float: return amount * 1.03

class Debit(PaymentMethod):
    def fee(self, amount: float) -> float: return amount * 1.01
```

---

### Temporary Field

**Stink:** A field is only set in some code paths. Most of the time it's `None`. E.g., a `RefundService` that sometimes has a `reason` field set, but most methods never use it. Or a `Transaction` with a `fee` field that's only set for certain types.

**Why it hurts:** Confusing — when is the field valid vs not? Null checks everywhere. Impossible to know the invariants.

**Fix:** Extract Class, Introduce Null Object, Remove Setting Method

**Doc:** https://refactoring.guru/smells/temporary-field

```python
# Stinky
class Order:
    def __init__(self):
        self.items = []
        self.discount_code = None        # only set sometimes
        self.discount_amount = 0.0       # only valid if discount_code set

# Clean
@dataclass
class Discount:
    code: str
    amount: float

class Order:
    def __init__(self):
        self.items = []
        self.discount: Discount | None = None
```

---

### Refused Bequest

**Stink:** A subclass inherits from a parent but ignores most of its methods or fields. Or overrides them to raise `NotImplementedError` / `pass`. The subclass "refuses" the inheritance contract.

**Why it hurts:** LSP violation — the subclass cannot substitute the parent. Inheritance was chosen for code reuse, not for an "is-a" relationship. Fragile when the parent changes.

**Fix:** Replace Inheritance with Delegation, Extract Subclass, Push Down Method/Field

**Doc:** https://refactoring.guru/smells/refused-bequest

```python
# Stinky
class Bird:
    def fly(self): ...
    def swim(self): ...
    def eat(self): ...

class Penguin(Bird):
    def fly(self): raise NotImplementedError("Penguins can't fly")  # Refused!

# Clean
class SwimmingBird:
    def swim(self): ...

class FlyingBird:
    def fly(self): ...

class Penguin(SwimmingBird): ...
class Sparrow(FlyingBird): ...
```

---

### Alternative Classes with Different Interfaces

**Stink:** Two classes do the same thing but expose different method names or signatures. E.g., `OrderSaver.save()` and `OrderPersister.persist()` — both write an order to storage.

**Why it hurts:** Callers must know both interfaces. Adding a new alternative means writing yet another adapter. No polymorphism possible.

**Fix:** Rename Method, Move Method, Extract Interface, Adapter pattern

**Doc:** https://refactoring.guru/smells/alternative-classes-with-different-interfaces

```python
# Stinky
class MongoOrders:
    def insert(self, data: dict): ...

class PostgresOrders:
    def save(self, order_id: str, payload: dict): ...

# Clean
class OrdersRepository(ABC):
    @abstractmethod
    def save(self, order: dict): ...

class MongoOrders(OrdersRepository):
    def save(self, order: dict): ...

class PostgresOrders(OrdersRepository):
    def save(self, order: dict): ...
```

---

## Change Preventers

### Divergent Change

**Stink:** One class changes for multiple different reasons. E.g., adding a new field changes the class, changing serialization format changes the class, adding validation changes the class, changing DB dialect changes the class.

**Why it hurts:** Single Responsibility Principle violation. Every new concern touches the same file. Merge conflicts are constant. Testing requires all contexts simultaneously.

**Fix:** Extract Class, Extract Superclass, Separate concerns into dedicated classes

**Doc:** https://refactoring.guru/smells/divergent-change

```python
# Stinky
class User:
    def validate(self): ...
    def to_json(self): ...
    def to_xml(self): ...
    def save(self): ...
    def send_welcome_email(self): ...

# Clean
class User: ...
class UserValidator: ...
class UserSerializer: ...
class UserRepository: ...
class WelcomeEmailSender: ...
```

---

### Shotgun Surgery

**Stink:** One change (e.g., adding a field to an entity) forces you to modify files scattered across the codebase — the model, the DB migration, the serializer, the validator, the form, the test fixtures, the API docs.

**Why it hurts:** Easy to miss one. Hard to onboard new devs. The change is simple conceptually but requires opening 8 files. Open/Closed violation at the module level.

**Fix:** Move Method, Move Field, Inline Class, Consolidate related behavior into one module

**Doc:** https://refactoring.guru/smells/shotgun-surgery

```python
# Stinky — adding 'phone' means editing all 5
# user.py, user_repo.py, user_serializer.py, user_validator.py, user_form.py

# Clean — adding 'phone' means editing just the dataclass
@dataclass
class User:
    name: str
    email: str

    def to_db_row(self): ...
    def validate(self): ...
    def serialize(self): ...
```

---

### Parallel Inheritance Hierarchies

**Stink:** Adding a class to hierarchy A forces you to add a corresponding class to hierarchy B. E.g., adding `CsvOrderExporter` means also adding `CsvOrderExporterTest`. Adding `EnterpriseCustomer` means adding `EnterpriseCustomerValidator`.

**Why it hurts:** Fragile — the two hierarchies must stay in sync or the system breaks. Multiply maintenance effort.

**Fix:** Collapse Hierarchy, Replace Inheritance with Delegation, Remove duplication between the hierarchies

**Doc:** https://refactoring.guru/smells/parallel-inheritance-hierarchies

```python
# Stinky — every new Exporter needs a corresponding Test
class JsonExporter: ...
class XmlExporter: ...
class JsonExporterTest: ...
class XmlExporterTest: ...

# Clean — one test suite parameterized
class ExporterTest(ABC):
    @abstractmethod
    def exporter(self) -> Exporter: ...
    def test_export_basic(self): ...

class JsonExporterTest(ExporterTest):
    def exporter(self): return JsonExporter()
```

---

## Dispensables

### Comments

**Stink:** Comments explaining *what* the code does (not *why*). Stale comments that contradict the code. Comented-out code blocks. "This is a workaround" with no linked issue.

**Why it hurts:** Comments rot faster than code. If a comment explains "what", the code is not self-documenting. Commented-out code clutters and deceives.

**Fix:** Extract Method, Rename Method, Introduce Assertion — make the code express intent. Delete commented-out code (that's what Git is for).

**Doc:** https://refactoring.guru/smells/comments

```python
# Stinky
# Check if user is over 18
if user.age >= 18: ...

# Clean — no comment needed
if user.is_adult(): ...

# Or rename
MINIMUM_ADULT_AGE = 18
if user.age >= MINIMUM_ADULT_AGE: ...
```

---

### Duplicate Code

**Stink:** Same expression, same block, or same algorithm appears in two or more places. Subsets of code that are identical but differ only in variable names.

**Why it hurts:** Fix a bug in one copy and forget the other? Now the bug lives on. DRY principle violation. Increases codebase size and cognitive load.

**Fix:** Extract Method, Pull Up Method, Form Template Method, Substitute Algorithm

**Doc:** https://refactoring.guru/smells/duplicate-code

```python
# Stinky
def apply_discount_standard(items): ...
def apply_discount_premium(items):
    # same logic as standard but with different threshold
    ...

# Clean
def apply_discount(items, *, min_for_discount: float, discount_rate: float): ...
```

---

### Lazy Class

**Stink:** A class that does almost nothing — one or two small methods, no real state. Or a class that was created for "future growth" that never came. A class whose only method is a one-liner delegate.

**Why it hurts:** Adds complexity: understanding, navigating, testing, and maintaining a class that contributes nothing.

**Fix:** Inline Class, Collapse Hierarchy, Remove Middle Man

**Doc:** https://refactoring.guru/smells/lazy-class

```python
# Stinky — does nothing the caller couldn't do directly
class Greeter:
    def greet(self, name: str) -> str:
        return f"Hello, {name}!"

# Clean — inline the function
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

---

### Data Class

**Stink:** A class that only has fields, `__init__`, getters, and setters — no behavior. All logic that operates on this data lives elsewhere (often in a "Service" or "Manager").

**Why it hurts:** Behavior is scattered. Adding a new operation doesn't touch the Data Class — it touches a Service. Over time, the Service grows into a God Object and the Data Class becomes an Anemic Domain Model.

**Fix:** Move Method, Encapsulate Field, Replace Data Value with Object, Extract Method (move behavior into the class)

**Doc:** https://refactoring.guru/smells/data-class

```python
# Stinky
@dataclass
class Order:
    id: str
    items: list
    total: float       # calculated externally

# Clean
@dataclass
class Order:
    id: str
    items: list

    @property
    def total(self) -> float:
        return sum(item.price * item.qty for item in self.items)
```

---

### Dead Code

**Stink:** Variables never read. Methods never called. Imports that are unused. Parameters with a fixed/default value in every call site. `if False:` blocks. Old versions kept "just in case."

**Why it hurts:** Wasted cognitive load — every line of code is a maintenance liability. Future devs wonder if it's used somewhere they can't see. Search results polluted with irrelevant hits.

**Fix:** Remove dead code, Inline Class, Remove Parameter. Git history exists for a reason.

**Doc:** https://refactoring.guru/smells/dead-code

```python
# Stinky
from typing import Optional  # unused import

def process(data: dict, old_param: str = None):  # old_param never used
    ...
    # if False:  # dead code from 2021
    #     legacy_flow(data)
```

---

### Speculative Generality

**Stink:** Abstractions, hooks, interfaces, or parameters created "for future use" that never materialize. An abstract base class with one concrete subclass. A `**kwargs` that's never used. A strategy pattern with one strategy.

**Why it hurts:** YAGNI violation — you pay the complexity cost now for a bet that almost never pays off. Harder to navigate, test, and change.

**Fix:** Collapse Hierarchy, Inline Class, Remove Parameter, Rename Method

**Doc:** https://refactoring.guru/smells/speculative-generality

```python
# Stinky
class PaymentProcessor(ABC):     # one implementation, no plans for second
    @abstractmethod
    def pay(self, ...): ...

class StripePayment(PaymentProcessor):
    def pay(self, ...): ...

# Clean
class StripePayment:
    def pay(self, ...): ...
```

---

## Couplers

### Feature Envy

**Stink:** A method that spends more time with another class's data than its own. Long chains of `other.get_x()`, `other.calc_y()`, `other.set_z()`. The method *envies* the other class.

**Why it hurts:** Low cohesion — the method belongs in the class it envies. If the other class changes, this method breaks too.

**Fix:** Move Method, Move Field, Extract Method

**Doc:** https://refactoring.guru/smells/feature-envy

```python
# Stinky — OrderService reads every field of Customer
class OrderService:
    def send_invoice(self, customer):
        name = customer.first_name + " " + customer.last_name
        email = customer.email
        address = f"{customer.street}, {customer.city} {customer.zip}"
        ...

# Clean — Customer formats itself
class Customer:
    def full_name(self): ...
    def mailing_address(self): ...
    def contact(self): ...
```

---

### Inappropriate Intimacy

**Stink:** One class knows too much about the internals of another. Direct field access (`other._private_field`). Methods that reach through multiple levels (`a.get_b().get_c().do_thing()`). Circular friendship.

**Why it hurts:** Tight coupling — changes to one class ripple to the intimate friend. Encapsulation destroyed. Fragile to refactoring.

**Fix:** Move Method, Move Field, Change Bidirectional Association to Unidirectional, Hide Delegate, Extract Class, Tease Apart Inheritance

**Doc:** https://refactoring.guru/smells/inappropriate-intimacy

```python
# Stinky
class User:
    def is_admin(self) -> bool: ...

class Dashboard:
    def can_access(self, user) -> bool:
        return user._role == "admin" and not user._banned  # reaching into internals

# Clean
class User:
    def is_admin(self) -> bool: ...
    def can_access_dashboard(self) -> bool: ...

class Dashboard:
    def can_access(self, user) -> bool:
        return user.can_access_dashboard()
```

---

### Message Chains

**Stink:** Long chain of method calls: `a.get_b().get_c().get_d().do_thing()`. Intermediate objects are only stepping stones to reach the target.

**Why it hurts:** Tight coupling to the entire navigation path. Any change in the chain breaks every client. The client knows more about the object graph than it should (Law of Demeter violation).

**Fix:** Hide Delegate, Extract Method, Move Method, Introduce Local Extension

**Doc:** https://refactoring.guru/smells/message-chains

```python
# Stinky
user = session.get_tenant().get_user_manager().find_by_id(id)
address = user.get_profile().get_address().get_city()

# Clean
user = session.get_user(id)
city = user.get_city()

# Inside Session:
def get_user(self, id): return self.tenant.user_manager.find_by_id(id)
```

---

### Middle Man

**Stink:** A class that exists only to delegate to another class. Every method on the Middle Man just calls the same method on the Real Class. The Middle Man adds no value.

**Why it hurts:** An extra hop with no benefit. Increases indirection and cognitive load. Makes the codebase feel "enterprisey" without actual encapsulation.

**Fix:** Remove Middle Man, Inline Class, Hide Delegate (only if the delegate is complex), Replace Delegation with Inheritance

**Doc:** https://refactoring.guru/smells/middle-man

```python
# Stinky — ReportsController does nothing but delegate
class ReportsController:
    def __init__(self):
        self.reports = ReportsService()
    def generate(self): return self.reports.generate()
    def list(self): return self.reports.list()

# Clean — call ReportsService directly
```

---

# Part II: Anti-Patterns

From [Wikipedia](https://en.wikipedia.org/wiki/List_of_software_anti-patterns) and industry sources

---

## Software Design Anti-Patterns

### Big Ball of Mud

**Stink:** The entire application is a tangled mess with no clear architecture. Files reference each other in cycles. No layers, no boundaries, no clear responsibilities. Everything depends on everything.

**Why it hurts:** Impossible to change without breaking something. No testability. No team can work on it in parallel. Every refactoring is risky.

**Fix:** Incrementally extract layers. Identify natural boundaries. Extract modules one at a time, writing tests at each boundary.

**Doc:** https://en.wikipedia.org/wiki/Big_ball_of_mud

```python
# Stinky — one file that does everything
# app.py: routes + business logic + DB + email + auth + ...

# Clean — layered architecture
# routes/ → controllers/ → services/ → repositories/ → models/
```

---

### Abstraction Inversion

**Stink:** Building complex, high-level abstractions on top of low-level primitives, and then requiring users of the high-level abstraction to understand the low-level details. E.g., requiring users of an ORM to write raw SQL to join tables.

**Why it hurts:** The abstraction doesn't "pay for itself" — users must drop down to the lower level regularly, defeating the purpose.

**Fix:** Extend the abstraction to cover the common cases. Add convenience methods. Don't expose the underlying mechanism unless absolutely necessary.

**Doc:** https://en.wikipedia.org/wiki/Abstraction_inversion

```python
# Stinky — users must write raw SQL for simple pagination
db.query(User).raw("SELECT * FROM (SELECT *, ROW_NUMBER() ...) WHERE rn > ?", offset)

# Clean
db.query(User).paginate(page=2, per_page=20)
```

---

### Ambiguous Viewpoint

**Stink:** The design or analysis is written from no clear perspective — it mixes user goals, system behavior, component interactions, and implementation details in the same document. Vague phrases like "the system should handle it."

**Why it hurts:** Different readers interpret the same text differently. Leads to miscommunication, wrong implementation, and integration failures.

**Fix:** Use C4 model or similar to separate levels: Context → Container → Component → Code. Be explicit about which viewpoint you're writing from.

**Doc:** https://en.wikipedia.org/wiki/Ambiguous_viewpoint

---

### Database-as-IPC

**Stink:** Using a shared database as a message queue or inter-process communication channel. Polling tables for new rows. Writing status flags that another process reads. Using DB triggers for orchestration.

**Why it hurts:** Couples all services to the same schema. DB becomes a bottleneck. Transaction conflicts. Schema changes require coordinated deploys. The DB is not a message broker.

**Fix:** Use a proper message broker (SQS, RabbitMQ, Kafka) or at least an outbox pattern.

**Doc:** https://en.wikipedia.org/wiki/Database-as-IPC

```python
# Stinky — polling DB for work
def worker_loop():
    while True:
        row = db.query("SELECT * FROM jobs WHERE status='pending' LIMIT 1")
        if row: process(row)
        time.sleep(5)

# Clean — subscribe to a queue
def on_message(msg):
    process(msg)
```

---

### Inner-Platform Effect

**Stink:** Building a system so configurable and customizable that it becomes a poor replica of the platform it runs on. E.g., a CMS that has its own scripting language, its own template engine, its own ORM, its own routing — all reimplemented badly.

**Why it hurts:** Enormous development and maintenance cost for something that already exists. Worse quality, worse performance, worse documentation.

**Fix:** Use the platform's native capabilities. Extend with plugins, don't rebuild. If the platform lacks a feature, contribute upstream.

**Doc:** https://en.wikipedia.org/wiki/Inner-platform_effect

---

### Input Kludge

**Stink:** Ad-hoc, inconsistent handling of input validation scattered across the codebase. Some inputs are validated, some aren't. Error messages are inconsistent. Edge cases are discovered by crash reports.

**Why it hurts:** Security vulnerabilities (injection, overflow). Poor UX (cryptic errors). Untestable — no single validation layer.

**Fix:** Validate at the edge (see `error-handling` skill: Parse, don't validate). Use a single validation layer (Pydantic, marshmallow). Centralize error formatting.

**Doc:** https://en.wikipedia.org/wiki/Input_kludge

---

### Interface Bloat

**Stink:** An interface (protocol/abstract class) with too many methods, many of which are irrelevant to most implementors. Clients depend on methods they never call. Implementors must stub unused methods.

**Why it hurts:** Interface Segregation Principle violation. Changes to the interface force changes in all implementations. Bloat grows over time — no one removes deprecated methods.

**Fix:** Split large interfaces into smaller, focused ones. Prefer many small protocols over one large ABC.

**Doc:** https://en.wikipedia.org/wiki/Interface_bloat

```python
# Stinky — one massive interface
class UserService(ABC):
    @abstractmethod
    def create(self): ...
    @abstractmethod
    def read(self): ...
    @abstractmethod
    def update(self): ...
    @abstractmethod
    def delete(self): ...
    @abstractmethod
    def send_email(self): ...
    @abstractmethod
    def generate_report(self): ...

# Clean — focused interfaces
class UserCRUD(ABC): ...
class EmailSender(ABC): ...
class ReportGenerator(ABC): ...
```

---

### Magic Pushbutton

**Stink:** Writing all business logic directly inside UI event handlers (button click, form submit, keypress). Huge callback functions that do everything: validate, transform, persist, log, navigate.

**Why it hurts:** Untestable — mocking GUI is painful. No reuse — logic is trapped in the handler. Promotes spaghetti code.

**Fix:** Separate concerns: Controller (orchestration) → Service (business logic) → Repository (persistence). Keep UI handlers thin.

**Doc:** https://en.wikipedia.org/wiki/Magic_pushbutton

```python
# Stinky — all logic in the handler
def on_submit(event):
    name = entry_name.get()
    if len(name) > 50: show_error("Name too long")
    db.insert("users", {"name": name})
    email.send(...)
    label_result.config(text="Saved!")

# Clean — thin handler
def on_submit(event):
    result = user_service.register(entry_name.get())
    show_user_feedback(result)
```

---

### Stovepipe System

**Stink:** Systems built as isolated silos — each with its own database, its own auth, its own UI, its own deployment pipeline. They cannot share data or functionality without custom bridges.

**Why it hurts:** Data duplication. No single source of truth. Integration requires expensive point-to-point adapters. Every new system adds another silo.

**Fix:** Define shared interfaces and data formats. Use a service bus or API gateway. Standardise on common infrastructure (auth, monitoring, CI/CD).

**Doc:** https://en.wikipedia.org/wiki/Stovepipe_system

---

### Race Hazard (Race Condition)

**Stink:** The system's correctness depends on the timing of uncontrollable events — two threads writing the same file, two requests updating the same DB row, a read after a write in another process.

**Why it hurts:** Heisenbugs — they disappear when you add logging or breakpoints. Hard to reproduce, harder to fix. Can cause data corruption.

**Fix:** Use locks, transactions, atomic operations, or idempotency keys. Prefer immutable data structures where possible.

**Doc:** https://en.wikipedia.org/wiki/Race_condition

---

## Object-Oriented Anti-Patterns

### God Object

**Stink:** A single class that knows too much or does too much. 30+ fields. 20+ methods. Touches every concern: persistence, validation, serialization, notifications, logging.

**Why it hurts:** Everything depends on it. Single point of failure. Impossible to test in isolation. Any change risks breaking unrelated features. The class cannot be split without touching all its dependents.

**Fix:** Extract Class, Extract Subclass, Move Method. Distribute responsibilities to focused classes.

**Doc:** https://en.wikipedia.org/wiki/God_object

```python
# Stinky — the God class
class Application:
    def __init__(self): ...
    def read_input(self): ...
    def validate(self): ...
    def calculate(self): ...
    def save(self): ...
    def send_email(self): ...
    def generate_report(self): ...
    def log(self): ...
    def render_ui(self): ...

# Clean — split into focused services
```

---

### Anemic Domain Model

**Stink:** Domain objects are just data bags with getters/setters. All business logic lives in "Service" classes. The model is anemic — it has no behavior.

**Why it hurts:** Procedural code disguised as OO. Behavior is scattered across services instead of staying close to the data it operates on. Services grow into God Objects.

**Fix:** Move Method — move business logic into the domain objects. Follow Tell, Don't Ask. If a method only uses one class's data, it belongs on that class.

**Doc:** https://en.wikipedia.org/wiki/Anemic_domain_model

```python
# Stinky
class Order:
    def __init__(self, items, status):
        self.items = items
        self.status = status

class OrderService:
    def can_cancel(self, order):
        return order.status in ("pending", "confirmed")

# Clean — behavior lives on the model
class Order:
    def __init__(self, items, status):
        self.items = items
        self.status = status

    def can_cancel(self) -> bool:
        return self.status in ("pending", "confirmed")
```

---

### Call Super

**Stink:** Every subclass overriding a method must call `super().method()` at the correct point, or the parent breaks. No compile-time enforcement — it's a convention documented in comments.

**Why it hurts:** Easy to forget. Easy to call in the wrong order. Fragile template method. Only discoverable at runtime.

**Fix:** Replace with the Template Method pattern where the parent controls the skeleton and subclasses provide hooks. Or use composition over inheritance.

**Doc:** https://en.wikipedia.org/wiki/Call_super

```python
# Stinky — must remember to call super
class BaseView:
    def render(self):
        self.render_header()
        self.render_body()  # subclass must call super().render() at start

class MyView(BaseView):
    def render(self):
        super().render()     # easy to forget!
        self.render_footer()

# Clean — Template Method
class BaseView:
    def render(self):
        self.render_header()
        self.render_body_hook()
        self.render_footer()

    def render_body_hook(self): ...
```

---

### Circular Dependency

**Stink:** Module A imports module B, and module B imports module A (directly or transitively). A ↔ B. Or A → B → C → A.

**Why it hurts:** Import errors at runtime (`ImportError` in Python). Hard to test in isolation. Tight coupling — you cannot change one without the other. Prevents independent deployment.

**Fix:** Extract the shared dependency into a new module. Use dependency injection. Apply the Dependency Inversion Principle (depend on abstractions, not concretions).

**Doc:** https://en.wikipedia.org/wiki/Circular_dependency

```python
# Stinky
# a.py: from b import B
# b.py: from a import A

# Clean — extract shared interface to a third module
# contracts.py: class EventHandler(ABC): ...
```

---

### Constant Interface

**Stink:** An interface that defines only constants (no methods). Classes implement it just to get access to the constants.

**Why it hurts:** Pollutes the class's namespace. Leaks constants to all subclasses. Not the purpose of interfaces.

**Fix:** Use a module-level constants file or a dedicated `Config` namespace. Python has no interface pollution issue — just use module constants.

**Doc:** https://en.wikipedia.org/wiki/Constant_interface

```python
# Stinky — in Java. In Python, just use module constants
class StatusCodes:
    OK = 200
    NOT_FOUND = 404
```

---

### Object Orgy

**Stink:** Excessive access to object internals from outside. No encapsulation — fields are public, getters expose mutable references, setters allow arbitrary state changes.

**Why it hurts:** No control over invariants. Objects cannot trust their own state. Changes to internal representation break external code.

**Fix:** Encapsulate Field, Hide Method, Remove Setting Method. Return immutable copies or views. Limit mutation to well-defined methods.

**Doc:** https://en.wikipedia.org/wiki/Object_orgy

---

### Poltergeist

**Stink:** Short-lived, stateless objects that are created just to call a single method and then discarded. They have no real identity or state. They "haunt" the code by appearing and disappearing.

**Why it hurts:** Unnecessary object creation. Indirection without benefit. Confusing — what is this class for?

**Fix:** Inline Class — replace with a function call. Use a module-level function instead of a throwaway object.

**Doc:** https://en.wikipedia.org/wiki/Poltergeist_(computer_science)

```python
# Stinky — created just for one call
class InvoiceDispatcher:
    def dispatch(self, invoice):
        email.send(invoice)
        return "sent"

dispatcher = InvoiceDispatcher()
result = dispatcher.dispatch(inv)

# Clean — just a function
def dispatch_invoice(invoice):
    email.send(invoice)
    return "sent"

result = dispatch_invoice(inv)
```

---

### Sequential Coupling

**Stink:** Methods must be called in a specific order or the object breaks. No compile-time enforcement. E.g., `builder.set_name(x).set_age(y).build()` — but `build()` fails if you forget `set_name`. Worse if the order is implicit (must call `open()` before `read()`).

**Why it hurts:** Runtime errors from forgotten setup steps. Hidden temporal coupling. Callers must know internal sequencing.

**Fix:** Use the Builder pattern (with a fluent API that guides the order at compile time), or the Step Builder pattern that encodes the sequence in the type system.

**Doc:** https://en.wikipedia.org/wiki/Sequential_coupling

```python
# Stinky — must remember the order
conn = Database()
conn.connect(host, port)           # step 1
conn.authenticate(user, pass)      # step 2
result = conn.query(sql)           # step 3

# Clean — context manager guarantees sequence
with Database(host, port, user, pass) as conn:
    result = conn.query(sql)
```

---

### Yo-Yo Problem

**Stink:** To understand the class hierarchy, you must constantly navigate up and down — parent → child → grandparent → back to child. Deep inheritance trees (5+ levels) where behavior is scattered across all levels.

**Why it hurts:** Impossible to understand a class without reading the entire ancestor chain. Changes at any level can break any subclass. An over-engineering smell.

**Fix:** Flatten the hierarchy. Prefer composition over inheritance. Collapse Hierarchy, Replace Inheritance with Delegation.

**Doc:** https://en.wikipedia.org/wiki/Yo-yo_problem

---

### Circle-Ellipse Problem

**Stink:** Modeling `Circle extends Ellipse` because a circle "is a" special case of ellipse. But a circle can't independently set width and height — it violates LSP. Mutations inherited from Ellipse break the Circle invariant.

**Why it hurts:** The Liskov Substitution Principle violation is subtle — it works for reads but breaks on mutations. Classic example of modeling real-world "is-a" without considering behavioral subtyping.

**Fix:** Make both shapes immutable, or use a common `Shape` interface with `resize(factor)` instead of individual setters. Or model them as unrelated classes sharing an interface.

**Doc:** https://en.wikipedia.org/wiki/Circle-ellipse_problem

---

### Object Cesspool

**Stink:** An object pool or cache that is never cleaned — it accumulates abandoned or stale objects indefinitely. Memory grows unbounded.

**Why it hurts:** Memory leak. Stale data served from the pool. Debugging is hard because the pool has no eviction policy.

**Fix:** Add eviction policies (TTL, LRU, max size). Use weak references where appropriate. Never pool without a cleanup strategy.

**Doc:** https://en.wikipedia.org/wiki/Object_cesspool

---

## Programming Anti-Patterns

### Spaghetti Code

**Stink:** Code with no discernible structure — GOTO statements, deeply nested conditionals, global state mutation everywhere, functions that span hundreds of lines. Control flow is as tangled as spaghetti.

**Why it hurts:** Impossible to follow, test, or modify safely. A single change can break unrelated features.

**Fix:** Incrementally refactor: Extract Method, Replace Nested Conditional with Guard Clauses, Decompose Conditional. Add tests first to create a safety net.

**Doc:** https://en.wikipedia.org/wiki/Spaghetti_code

### Lasagna Code

**Stink:** The opposite of spaghetti — layers upon layers of abstraction where each layer adds minimal value. To add a one-line feature, you touch 6 files across 4 layers. The code is "structured" but over-abstracted.

**Why it hurts:** High cognitive load to trace a simple code path. Bureaucratic code. Changes are slow because many layers must be coordinated.

**Fix:** Collapse Hierarchy, Inline Class, Remove Middle Man. Question whether each layer is "paying for itself" in reduced complexity.

**Doc:** https://en.wikipedia.org/wiki/Spaghetti_code#Lasagna_code

### Lava Flow

**Stink:** Dead code, experimental code, and half-finished features left in the codebase because "it might be useful later" or "nobody knows if it's used." Often from rapid prototyping with no cleanup.

**Why it hurts:** Codebase grows without bound. No one knows what's safe to delete. Testing and building slows down. Dead code hides bugs by making searches noisy.

**Fix:** When in doubt, delete it. Git history exists. Run coverage tools to identify truly unused code. If it's experimental, tag it clearly and set a cleanup deadline.

**Doc:** https://en.wikipedia.org/wiki/Lava_flow_(programming)

### Boat Anchor

**Stink:** A piece of code, library, or tool that was added to the project but is never used. A heavy ORM when you only need two queries. A test framework you never write tests with. A CI step that always passes trivially.

**Why it hurts:** Adds build time, dependency resolution complexity, maintenance burden, and cognitive load for zero benefit.

**Fix:** Remove it. If it's a library, uninstall it. If it's a build step, delete it. If you need it later, you can re-add it.

**Doc:** https://en.wikipedia.org/wiki/Boat_anchor_(computer_science)

### Cargo Cult Programming

**Stink:** Including code, patterns, or rituals without understanding why — "this is how we did it in the last project" or "everyone imports it this way." Removing something "just in case it's needed" even though nothing breaks without it.

**Why it hurts:** Code bloat. Outdated workarounds for problems that no longer exist. Patterns that made sense in Java/C# cargo-culted into Python where they fight the language.

**Fix:** Question every pattern: "Why are we doing this? Does it solve a current problem?" Prefer idiomatic Python over Cargo Cult Java-isms.

**Doc:** https://en.wikipedia.org/wiki/Cargo_cult_programming

```python
# Stinky — Java-style getters/setters in Python
class User:
    def get_name(self): return self._name
    def set_name(self, value): self._name = value

# Clean — Pythonic
class User:
    @property
    def name(self): return self._name

    @name.setter
    def name(self, value): self._name = value
```

---

### Hard Code

**Stink:** Configuration values, URLs, credentials, magic numbers, or business rules embedded directly in the source code. `timeout = 30` in the middle of a function. `db_url = "localhost"`.

**Why it hurts:** Cannot change without modifying and redeploying code. Different environments require different builds. Secrets leak into source control.

**Fix:** Externalize to environment variables, config files, or feature flags. Use `os.getenv()` or a config library (Pydantic Settings, Dynaconf).

**Doc:** https://en.wikipedia.org/wiki/Hard_code

```python
# Stinky
def connect():
    return psycopg2.connect(host="localhost", port=5432, password="hunter2")

# Clean
def connect():
    return psycopg2.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        password=settings.DB_PASSWORD,
    )
```

---

### Magic Numbers

**Stink:** Literal numbers in code with no explanation: `if x > 86400:` (what is 86400?). `price * 0.9` (what is 0.9?). Numbers that mean something but aren't named.

**Why it hurts:** Obscures intent. Changing the value requires finding all occurrences. Duplicate values that are actually different constants but happen to be equal.

**Fix:** Replace Magic Number with Symbolic Constant. Extract as a named constant or enum.

**Doc:** https://en.wikipedia.org/wiki/Magic_number_(programming)

```python
# Stinky
if elapsed > 86400: expire_session()

# Clean
ONE_DAY_IN_SECONDS = 86400
if elapsed > ONE_DAY_IN_SECONDS: expire_session()
```

---

### Magic Strings

**Stink:** String literals used as keys, status values, or type discriminators: `if status == "active":`, `cache_key = f"user:{id}:{type}"`. No single source of truth for these values.

**Why it hurts:** Typos are silent bugs. Refactoring is impossible — you can't "find all references" for a string. No IDE support.

**Fix:** Replace with enums, module-level constants, or TypedDict keys.

**Doc:** https://en.wikipedia.org/wiki/Magic_string_(programming)

```python
# Stinky
if user.status == "active": send_notification(user)

# Clean
from enum import Enum

class UserStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

if user.status == UserStatus.ACTIVE: send_notification(user)
```

---

### Error Hiding

**Stink:** Catching broad exceptions and doing nothing (or logging only): `try: ... except: pass`, `except Exception: log.warning("failed")`. Swallowing errors that should propagate.

**Why it hurts:** Bugs hide silently. Downstream code receives unexpected `None` or default values. Debugging requires digging through logs. Failures are discovered by users, not by monitoring.

**Fix:** Catch specific exceptions only. If you catch an exception, handle it or re-raise. Never `except: pass`. Consider using the Result pattern (see `error-handling` skill).

**Doc:** https://en.wikipedia.org/wiki/Error_hiding

```python
# Stinky
try:
    process(data)
except Exception:
    pass  # swallowed!

# Clean
try:
    process(data)
except ValidationError as e:
    return Err(e)
except ConnectionError:
    return Err("service unavailable")
```

---

### Accidental Complexity

**Stink:** Adding complexity that is not inherent to the problem but is a side effect of the solution approach. E.g., a `UserFactoryFactory` because "you might need it." XML configs for a 3-line script. Over-engineering.

**Why it hurts:** Increases development and maintenance cost while solving no actual problem. The accidental complexity hides the essential complexity.

**Fix:** YAGNI — build the simplest thing that works. Let the architecture emerge from needs, not from anticipation.

**Doc:** https://en.wikipedia.org/wiki/Accidental_complexity

---

### Action at a Distance

**Stink:** A change in one part of the code affects another part that is far away, with no obvious connection. Global variables, monkey-patching, thread-local state, implicit context managers, deeply nested callbacks.

**Why it hurts:** Non-local reasoning — you cannot understand the code by reading it top-to-bottom. Debugging requires tracing across files. Changes have unpredictable side effects.

**Fix:** Make data flow explicit — pass parameters, use dependency injection. Avoid global state. Prefer pure functions.

**Doc:** https://en.wikipedia.org/wiki/Action_at_a_distance_(computer_science)

---

### Busy Waiting

**Stink:** A loop that continuously checks a condition without yielding control: `while not ready: pass`, `while True: if check(): break`. Wastes CPU cycles.

**Why it hurts:** 100% CPU burn doing nothing. Blocks other threads/processes. Poor battery life on mobile/laptop.

**Fix:** Use events, condition variables, async/await, or sleep with backoff. Let the OS or runtime notify you when the condition is met.

**Doc:** https://en.wikipedia.org/wiki/Busy_waiting

```python
# Stinky
while not job_complete():
    time.sleep(0.01)  # slightly better than bare loop, still wasteful

# Clean — use an event
event.wait()  # blocks until signalled
```

---

### Caching Failure

**Stink:** Caching failed responses (404s, errors, empty results) as if they were successful. The cache remembers "there's nothing here" even after the resource becomes available.

**Why it hurts:** Users see stale failures. Errors become permanent. Hard to clear. Wastes cache space on negative results.

**Fix:** Don't cache error responses, or use very short TTLs for them. Add a "negative cache" with different eviction policy. Use `Cache-Control: no-store` for errors.

**Doc:** https://en.wikipedia.org/wiki/Caching_failure

---

### Coding by Exception

**Stink:** Using exceptions for normal control flow. `try: return get_value() except KeyError: return default`. Throwing exceptions intentionally to break out of loops or to communicate expected results.

**Why it hurts:** Exceptions are for exceptional conditions, not control flow. They're slow (stack unwinding), hard to debug, and obscure the happy path. Every call site must remember which exceptions to catch.

**Fix:** Use option types (`Optional`, `Result[T, E]`) or explicit sentinel values. Reserve exceptions for truly unexpected conditions (network down, disk full, bug).

**Doc:** https://en.wikipedia.org/wiki/Coding_by_exception

```python
# Stinky — exception for control flow
def find_user(users, name):
    try:
        return [u for u in users if u.name == name][0]
    except IndexError:
        return None

# Clean
def find_user(users, name):
    return next((u for u in users if u.name == name), None)
```

---

### Loop-Switch Sequence

**Stink:** A loop containing a switch statement (or if/elif chain) that dispatches based on a value that doesn't change within the loop. The switch runs on every iteration even though the branch is always the same.

**Why it hurts:** Inefficient — the decision is re-evaluated unnecessarily. The intent is unclear — it looks like the branch might change per iteration when it doesn't.

**Fix:** Move the switch before the loop. Use polymorphism or strategy objects selected before the loop.

**Doc:** https://en.wikipedia.org/wiki/Loop-switch_sequence

```python
# Stinky — checks fmt on every row
for row in data:
    if fmt == "csv": write_csv(row)
    elif fmt == "json": write_json(row)

# Clean — choose strategy once
writer = CsvWriter() if fmt == "csv" else JsonWriter()
for row in data:
    writer.write(row)
```

---

### Repeating Yourself (Violating DRY)

**Stink:** The same knowledge expressed in multiple places: same validation logic in frontend and backend, same date formatting in 5 files, same SQL query in 3 repositories, same configuration in 2 formats.

**Why it hurts:** Fix a bug in one place and miss the others? Bug lives on. Inconsistencies creep in over time. Changes take longer.

**Fix:** Extract duplication into a single authoritative source. But beware of premature DRY — if the two occurrences are conceptually different but happen to look alike today, they're not actually repeating yourself.

**Doc:** https://en.wikipedia.org/wiki/Don%27t_repeat_yourself

---

### Shooting the Messenger

**Stink:** When a method receives an object it doesn't like (unexpected data, an error), it throws away the message and blames (deletes/blocks/silences) the messenger instead of handling the problem.

**Why it hurts:** Lost messages mean lost data, lost errors, lost business. Hard to debug — the evidence is destroyed.

**Fix:** Handle unexpected data gracefully. Log and move on. Never silently discard input. Implement dead-letter queues for processing failures.

**Doc:** https://en.wikipedia.org/wiki/Shooting_the_messenger

---

### Shotgun Surgery

*See also code smell — this appears in both catalogs.*

**Stink:** A single logical change requires modifying many files scattered across the codebase. Adding a field means editing model, migration, serializer, validator, form, template, test fixtures, and docs.

**Why it hurts:** High risk of missing one. Hard to review. Slow to implement. The codebase fights against change.

**Fix:** Consolidate related behavior. Apply Open/Closed Principle at the module level. Use code generation for repetitive patterns.

**Doc:** https://en.wikipedia.org/wiki/Shotgun_surgery

---

### Soft Code (Softcoding)

**Stink:** Making *everything* configurable — putting business logic in config files, databases, or UI because "the business might want to change it without redeploying." Ends up with a turing-complete config system that's worse than code.

**Why it hurts:** Harder to test, debug, and version than code. Config becomes the new code — but without the tooling (type checking, tests, reviews). Performance is worse. Inner-platform effect cousin.

**Fix:** Keep business logic in code where it belongs. Use config for genuine variation (environments, credentials, feature flags). Accept that some changes require a deploy.

**Doc:** https://en.wikipedia.org/wiki/Softcoding

---

## Methodological Anti-Patterns

### Copy-Paste Programming

**Stink:** Duplicating code blocks by copy-paste instead of extracting and reusing. Minor variations between copies. "It's faster than refactoring."

**Why it hurts:** Every copy is a maintenance liability. Bug fix must be applied N times. Codebase grows without bound. Variations hide real differences from fake ones.

**Fix:** Extract Method, Pull Up Method, Template Method. Before pasting, ask: "Do I really need a variation, or can I parameterize?"

**Doc:** https://en.wikipedia.org/wiki/Copy_and_paste_programming

---

### Golden Hammer

**Stink:** Using a familiar tool, framework, or pattern for every problem, regardless of fit. "When you have a hammer, everything looks like a nail." Using microservices for a CRUD app. Using Kafka for a simple notification. Using a machine learning model when a simple `if` would do.

**Why it hurts:** Over-engineering. Wrong tool for the job. Higher complexity and cost than necessary. The solution fights the problem.

**Fix:** Match the tool to the problem. Ask: "What's the simplest thing that could work?" Have a diverse toolbox. Learn new patterns.

**Doc:** https://en.wikipedia.org/wiki/Law_of_the_instrument

---

### Not Invented Here (NIH)

**Stink:** Rejecting external solutions in favor of building your own, even when a quality open-source alternative exists. "We need control." "Our needs are unique."

**Why it hurts:** Huge wasted effort rebuilding battle-tested software. Your in-house version has fewer features, more bugs, less documentation, and no community.

**Fix:** Prefer existing solutions unless there's a compelling reason not to. Deduplicate with the community. Contribute upstream rather than fork.

**Doc:** https://en.wikipedia.org/wiki/Not_invented_here

---

### Invented Here

**Stink:** The opposite of NIH — taking pride in everything being built externally, to the point of depending on poorly maintained, low-quality, or abandoned third-party libraries. No willingness to build anything in-house.

**Why it hurts:** Dependency hell. Abandoned libraries create security risks. No strategic differentiation. Every external dependency is a risk.

**Fix:** Balance — build what differentiates you, buy what's commodity. Audit dependencies regularly.

**Doc:** https://en.wikipedia.org/wiki/Invented_here

---

### Premature Optimization

**Stink:** Optimizing code for performance before measuring, before confirming it's a bottleneck. Caching too early. Micro-optimizations. Complex data structures for simple datasets. "Fast is better than correct."

**Why it hurts:** Wasted effort on code that isn't slow. Introduces bugs and complexity in exchange for no measurable benefit. Knuth: "Premature optimization is the root of all evil."

**Fix:** Write clear, correct code first. Profile. Optimize the hot paths. Leave the cold paths alone.

**Doc:** https://en.wikipedia.org/wiki/Program_optimization#When_to_optimize

---

### Programming by Permutation

**Stink:** Making random changes until the code works, without understanding why. "Let me try changing X to Y... nope. How about Z... still nope. What if I add a sleep?" No hypothesis, no debugging.

**Why it hurts:** Produces code that "works" but no one knows why. Fragile — breaks when conditions change. Impossible to review. Teaches nothing.

**Fix:** Understand the problem. Add logging. Write a test that reproduces the bug. Verify the fix with the test. If you don't understand why it works, you don't know it works.

**Doc:** https://en.wikipedia.org/wiki/Programming_by_permutation

---

### Reinventing the Square Wheel

**Stink:** Building a custom solution for a problem that has many existing well-known solutions, but building it worse. Not just NIH — actively doing it worse than the standard approach.

**Why it hurts:** Same as NIH + inferior quality.

**Fix:** Use the standard library. Use well-known patterns. Only innovate where it matters.

**Doc:** https://en.wikipedia.org/wiki/Reinventing_the_square_wheel

---

### Silver Bullet

**Stink:** Believing that a single technology, methodology, or tool will solve all project problems. "If we switch to microservices, our velocity will 10x." "If we use Rust, the bugs will disappear." "If we do Agile right..."

**Why it hurts:** Disappointment when the silver bullet fails. Misses the real systemic issues. Throws away what works.

**Fix:** Brooks's Law: "No silver bullet." Improvement is incremental. Combine multiple approaches. Fix process, people, and culture — not just tools.

**Doc:** https://en.wikipedia.org/wiki/No_Silver_Bullet

---

### Tester-Driven Development

**Stink:** QA or testers define what the developers should build, often through test cases written before or instead of specifications. Tests drive the development rather than verifying it.

**Why it hurts:** Tests become specs. Tests are written before understanding the problem. Brittle tests that break on any refactoring. Devs optimize for "making the test pass" instead of "solving the problem."

**Fix:** Tests should verify, not specify. Write specs first. Use TDD correctly (test *drives* design, not replaces specs). Keep tests at the right level (behavior, not implementation).

**Doc:** https://en.wikipedia.org/wiki/Tester-driven_development

---

## Configuration Management Anti-Patterns

### Dependency Hell

**Stink:** The project depends on library A which requires version 2.x of library C, and library B which requires version 3.x of library C. Incompatible transitive dependencies. "Works on my machine" because of subtly different resolved versions.

**Why it hurts:** Builds break on different machines or at different times. Upgrading one library cascades into upgrading everything. `pip freeze` creates a lockfile that's impossible to review.

**Fix:** Use lockfiles (pip freeze, poetry.lock, Pipfile.lock). Pin versions. Run CI with clean installs. Prefer libraries with minimal dependency trees.

**Doc:** https://en.wikipedia.org/wiki/Dependency_hell

---

### DLL Hell

**Stink:** Windows-specific: shared libraries (DLLs) from different applications conflict because they install different versions to the same system directory. App A works, App B breaks after installing App A.

**Why it hurts:** Unpredictable failures. Hard to diagnose (the DLL is there — wrong version). Regression on unrelated installs.

**Fix:** Use statically linked libraries, side-by-side assemblies, or containers. This is largely solved by modern practices (Docker, NuGet, pip).

**Doc:** https://en.wikipedia.org/wiki/DLL_hell

---

### Extension Conflict

**Stink:** Two browser extensions, IDE plugins, or system extensions that modify the same behavior conflict. Both override the same function. Both add the same keyboard shortcut. Both patch the same prototype.

**Why it hurts:** Unpredictable behavior. One extension silently disables the other. Debugging requires disabling extensions one by one.

**Fix:** Isolate extensions. Use well-defined extension points (hooks, events) instead of monkey-patching. Test combinations.

**Doc:** https://en.wikipedia.org/wiki/Extension_conflict

---

### JAR Hell

**Stink:** Java-specific: multiple versions of the same JAR on the classpath. ClassLoader loads the wrong one. `NoSuchMethodError` at runtime even though the method exists in "your" version.

**Why it hurts:** Same as DLL hell but for Java. Runtime errors not caught at compile time.

**Fix:** Use a build tool (Maven, Gradle) that manages transitive dependencies and deduplication. Use module systems (JPMS, OSGi). Python equivalent: `sys.path` ordering issues.

**Doc:** https://en.wikipedia.org/wiki/JAR_hell