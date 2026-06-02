# Python Classes & OOP — Complete Guide for ML Engineers
> Every topic from the ground up, with Machine Learning examples throughout.

***Author:-Monesh_Patidar_IIT_Kanpur***

## Table of Contents
1. [Getting Started With Python Classes](#1-getting-started-with-python-classes)
2. [Defining a Class in Python](#2-defining-a-class-in-python)
3. [Creating Objects From a Class](#3-creating-objects-from-a-class)
4. [Accessing Attributes and Methods](#4-accessing-attributes-and-methods)
5. [Naming Conventions](#5-naming-conventions-in-python-classes)
6. [Public vs Non-Public Members](#6-public-vs-non-public-members)
7. [Name Mangling](#7-name-mangling)
8. [Benefits of Using Classes](#8-understanding-the-benefits-of-using-classes)
9. [When to Avoid Classes](#9-deciding-when-to-avoid-classes)
10. [Attaching Data — Class vs Instance Attributes](#10-attaching-data-to-classes-and-instances)
11. [The `__dict__` Attribute](#11-the-__dict__-attribute)
12. [Dynamic Attributes](#12-dynamic-class-and-instance-attributes)
13. [Property and Descriptor-Based Attributes](#13-property-and-descriptor-based-attributes)
14. [Lightweight Classes With `__slots__`](#14-lightweight-classes-with-__slots__)
15. [Instance Methods With `self`](#15-instance-methods-with-self)
16. [Special Methods and Protocols (Dunder Methods)](#16-special-methods-and-protocols-dunder-methods)
17. [Class Methods With `@classmethod`](#17-class-methods-with-classmethod)
18. [Static Methods With `@staticmethod`](#18-static-methods-with-staticmethod)
19. [Getter and Setter Methods vs Properties](#19-getter-and-setter-methods-vs-properties)
20. [Complete Example — Summarizing Class Syntax](#20-complete-example--summarizing-class-syntax)
21. [Debugging Python Classes](#21-debugging-python-classes)
22. [Data Classes](#22-data-classes)
23. [Enumerations](#23-enumerations)
24. [Simple Inheritance](#24-simple-inheritance)
25. [Class Hierarchies](#25-class-hierarchies)
26. [Extended vs Overridden Methods](#26-extended-vs-overridden-methods)
27. [Multiple Inheritance](#27-multiple-inheritance)
28. [Method Resolution Order (MRO)](#28-method-resolution-order-mro)
29. [Mixin Classes](#29-mixin-classes)
30. [Benefits of Using Inheritance](#30-benefits-of-using-inheritance)
31. [Composition](#31-composition)
32. [Delegation](#32-delegation)
33. [Dependency Injection](#33-dependency-injection)
34. [Abstract Base Classes (ABCs) and Interfaces](#34-abstract-base-classes-abcs-and-interfaces)
35. [Polymorphism With Common Interfaces](#35-unlocking-polymorphism-with-common-interfaces)
36. [FAQs](#36-frequently-asked-questions)

---

## 1. Getting Started With Python Classes

### What is a Class?

Think of a class as a **blueprint** or **template**. In ML, you train many models — a Linear Regression, a Random Forest, a Neural Network. Each model has:
- **Data** (e.g., weights, learning rate, number of layers)
- **Behavior** (e.g., fit, predict, evaluate)

A class lets you package both together in one clean unit.

### The Real-World Analogy

A cookie cutter = class  
A cookie = object (instance)

You make one cutter (class), and bake hundreds of cookies (objects) from it. Each cookie can have different toppings (data/attributes), but they all share the same shape (structure/methods).

### Why Does ML Need OOP?

Without OOP, your code looks like this:

```python
# Without OOP — messy, hard to manage
lr = 0.01
weights = [0.1, 0.2, 0.3]
epochs = 100

def train(weights, lr, epochs):
    pass

def predict(weights, X):
    pass
```

With OOP, it becomes:

```python
# With OOP — clean, reusable, organized
model = NeuralNetwork(lr=0.01, epochs=100)
model.train(X_train, y_train)
predictions = model.predict(X_test)
```

This is exactly how scikit-learn, PyTorch, and Keras work internally!

---

## 2. Defining a Class in Python

### Syntax

```python
class ClassName:
    # class body goes here
    pass
```

The `class` keyword starts the definition. The body is indented. `pass` means "empty for now."

### Your First ML Class

```python
class LinearRegression:
    """A simple linear regression model."""
    pass
```

### The `__init__` Method (Constructor)

When you create an object, Python automatically calls `__init__`. This is where you set up initial data.

```python
class LinearRegression:
    """A simple linear regression model."""

    def __init__(self, learning_rate=0.01, epochs=1000):
        # 'self' refers to the object being created
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None   # not set yet
        self.bias = None      # not set yet
```

### What is `self`?

`self` is a reference to **the current object**. Every method in a class receives `self` as its first argument automatically. It's how an object "talks about itself."

```
model1.learning_rate → self is model1
model2.learning_rate → self is model2
```

They are separate objects, each with their own data.

---

## 3. Creating Objects From a Class

Once the class (blueprint) is ready, you create objects (instances) from it.

```python
# Creating instances
model1 = LinearRegression()                    # uses defaults
model2 = LinearRegression(learning_rate=0.001) # custom lr
model3 = LinearRegression(0.1, 500)            # positional args
```

### Each Object is Independent

```python
print(model1.learning_rate)  # 0.01
print(model2.learning_rate)  # 0.001
print(model3.learning_rate)  # 0.1
```

Changing one doesn't affect others:

```python
model1.learning_rate = 0.05
print(model1.learning_rate)  # 0.05
print(model2.learning_rate)  # still 0.001 — unaffected
```

### Under the Hood

When you write `LinearRegression()`:
1. Python creates an empty object in memory.
2. Python calls `__init__(self, ...)` with that object as `self`.
3. The object is returned and assigned to your variable.

---

## 4. Accessing Attributes and Methods

### Dot Notation

You access everything inside an object using a dot (`.`):

```python
# Accessing attributes
print(model1.learning_rate)   # 0.05
print(model1.epochs)          # 1000

# Setting attributes
model1.epochs = 500

# Accessing methods (functions inside a class)
model1.train(X, y)
predictions = model1.predict(X_test)
```

### A More Complete Example

```python
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        """Train the model using gradient descent."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y

            # Gradient descent update
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

    def predict(self, X):
        """Make predictions."""
        return np.dot(X, self.weights) + self.bias


# Usage
X_train = np.array([[1, 2], [3, 4], [5, 6]])
y_train = np.array([3, 7, 11])

model = LinearRegression(learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)

X_test = np.array([[7, 8]])
print(model.predict(X_test))     # [15.something]
print(model.weights)             # learned weights
print(model.bias)                # learned bias
```

---

## 5. Naming Conventions in Python Classes

Python has standard naming rules. Breaking these won't cause errors, but everyone will hate you (and your future self will too).

### Classes → PascalCase (CapWords)

Each word starts with a capital letter, no underscores:

```python
# ✅ Correct
class NeuralNetwork: ...
class RandomForestClassifier: ...
class DataPreprocessor: ...

# ❌ Wrong
class neural_network: ...
class randomforest: ...
```

### Methods and Attributes → snake_case

All lowercase, words separated by underscores:

```python
class Model:
    def __init__(self):
        self.learning_rate = 0.01     # ✅ snake_case
        self.numLayers = 3            # ❌ camelCase (bad Python style)

    def train_model(self):            # ✅ snake_case
        pass

    def TrainModel(self):             # ❌ PascalCase (bad for methods)
        pass
```

### Constants → ALL_CAPS

```python
class Config:
    MAX_EPOCHS = 1000       # constant — won't change
    DEFAULT_LR = 0.001
    BATCH_SIZE = 32
```

### Summary Table

| What | Convention | Example |
|------|-----------|---------|
| Class name | PascalCase | `NeuralNetwork` |
| Method | snake_case | `train_model()` |
| Attribute | snake_case | `learning_rate` |
| Constant | ALL_CAPS | `MAX_EPOCHS` |
| "Private" member | `_single_underscore` | `_weights` |
| "Name-mangled" | `__double_underscore` | `__secret` |

---

## 6. Public vs Non-Public Members

Python doesn't have strict `private`/`public` keywords like Java or C++. Instead, it uses naming conventions to **signal intent**.

### Public Members (no underscore)

Accessible from anywhere. The "official API" of your class.

```python
class Model:
    def __init__(self):
        self.accuracy = 0.95    # public — use freely
        self.loss = 0.05        # public

    def predict(self, X):       # public method
        return X @ self.weights
```

### Non-Public / "Private" Members (`_single_underscore`)

A **convention** (not enforcement) that says: "This is for internal use. Don't touch it from outside unless you know what you're doing."

```python
class GradientDescent:
    def __init__(self, lr=0.01):
        self.lr = lr
        self._gradient = None        # internal detail
        self._prev_loss = float('inf') # internal tracking

    def _compute_gradient(self, X, y, y_pred):
        """Internal helper — not meant for outside use."""
        return (2 / len(X)) * X.T @ (y_pred - y)

    def step(self, X, y, y_pred, weights):
        """Public API — this is what users call."""
        self._gradient = self._compute_gradient(X, y, y_pred)
        return weights - self.lr * self._gradient


optimizer = GradientDescent(lr=0.01)

# ✅ OK — public method
new_w = optimizer.step(X, y, y_pred, weights)

# ⚠️ Allowed but frowned upon
print(optimizer._gradient)   # works, but signals "I know I'm doing something unusual"
```

### Key Point

The underscore is a **gentleman's agreement**, not a lock. Python trusts you to respect it.

---

## 7. Name Mangling

### What is Name Mangling?

When you use **double underscore prefix** (`__name`), Python **automatically renames** the attribute to `_ClassName__name`. This makes it harder (but not impossible) to access from outside.

```python
class SecureModel:
    def __init__(self, api_key):
        self.__api_key = api_key    # Python renames this to _SecureModel__api_key
        self.accuracy = 0.0

    def call_api(self):
        # Works fine inside the class
        return f"Calling with key: {self.__api_key}"


model = SecureModel("sk-abc123")

# ❌ This fails
# print(model.__api_key)   → AttributeError

# ✅ Still accessible if you know the mangled name (Python doesn't truly hide it)
print(model._SecureModel__api_key)  # sk-abc123

# ✅ Normal access
print(model.call_api())             # Calling with key: sk-abc123
```

### Why Use Name Mangling?

It's primarily useful in **inheritance** to prevent subclasses from accidentally overriding an attribute.

```python
class BaseModel:
    def __init__(self):
        self.__version = "1.0"     # becomes _BaseModel__version

    def get_version(self):
        return self.__version      # looks up _BaseModel__version


class MyModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.__version = "2.0"     # becomes _MyModel__version — different attribute!

    def get_my_version(self):
        return self.__version      # looks up _MyModel__version


m = MyModel()
print(m.get_version())       # 1.0 — base class sees its own __version
print(m.get_my_version())    # 2.0 — child class sees its own __version
```

Without mangling, `__version` in the child would overwrite the parent's, breaking `get_version()`.

### Quick Summary

| Syntax | Name stored as | Accessible from outside? |
|--------|---------------|--------------------------|
| `name` | `name` | Yes |
| `_name` | `_name` | Yes (but convention says don't) |
| `__name` | `_ClassName__name` | Yes (but harder — must use mangled name) |

---

## 8. Understanding the Benefits of Using Classes

### 1. Encapsulation — Bundle Data + Behavior

```python
# Without OOP — scattered everywhere
weights = None
bias = None
lr = 0.01

def train(X, y, weights, bias, lr): ...
def predict(X, weights, bias): ...
def save(weights, bias, filename): ...

# With OOP — everything together
model = NeuralNetwork(lr=0.01)
model.train(X, y)
model.predict(X_test)
model.save("model.pkl")
```

### 2. Reusability

Write once, use many times with different configurations:

```python
small_model = NeuralNetwork(layers=[32, 16], lr=0.001)
large_model = NeuralNetwork(layers=[512, 256, 128], lr=0.0001)
```

### 3. Maintainability

Change one class → all instances benefit. No need to hunt down scattered functions.

### 4. Modeling the Real World

```python
class Dataset:
    def __init__(self, path):
        self.path = path
        self.data = None

    def load(self): ...
    def split(self, ratio=0.8): ...
    def normalize(self): ...


class Experiment:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.results = {}

    def run(self): ...
    def log(self): ...
```

Real ML pipelines (like MLflow, scikit-learn pipelines) are built exactly this way.

### 5. Extensibility Through Inheritance

```python
class BaseModel: ...
class NeuralNetwork(BaseModel): ...   # extends base
class CNN(NeuralNetwork): ...         # extends further
```

---

## 9. Deciding When to Avoid Classes

Classes are powerful, but not always necessary. Here's when **not** to use them:

### Avoid Classes When...

**1. You just need a one-off function:**
```python
# Overkill — don't do this
class Adder:
    def add(self, a, b):
        return a + b

adder = Adder()
result = adder.add(2, 3)

# Better — just a function
def add(a, b):
    return a + b
result = add(2, 3)
```

**2. You have no state to maintain:**
```python
# No persistent data? → Use a module with functions
# utils.py
def normalize(X):
    return (X - X.mean()) / X.std()

def train_test_split(X, y, ratio=0.8):
    ...
```

**3. You'd create a class with only one method:**
```python
# Useless class
class Predictor:
    def predict(self, X):
        return X @ weights   # weights from where??

# Better — just a function or a closure
def make_predictor(weights):
    def predict(X):
        return X @ weights
    return predict
```

**4. Simple data containers (use `namedtuple` or `dataclass` instead):**
```python
# Overkill
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Better
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(1.0, 2.0)
```

### The Rule of Thumb

> Use a class when you have **both** data **and** behavior that belong together, and you expect to create **multiple instances** or **extend** the class later.

---

## 10. Attaching Data to Classes and Instances

There are two kinds of attributes in Python classes: **class attributes** and **instance attributes**. Understanding the difference is critical.

---

### Class Attributes

Defined **directly in the class body**, outside any method. They are **shared** across all instances.

```python
class NeuralNetwork:
    # Class attributes — shared by ALL instances
    framework = "NumPy"
    version = "1.0"
    instance_count = 0      # track how many models exist

    def __init__(self, layers):
        NeuralNetwork.instance_count += 1   # update class attribute
        self.layers = layers                 # instance attribute
```

```python
m1 = NeuralNetwork([32, 16])
m2 = NeuralNetwork([64, 32])

print(NeuralNetwork.framework)    # "NumPy" — accessed from class
print(m1.framework)               # "NumPy" — accessed from instance (looks up class)
print(m2.framework)               # "NumPy" — same value

print(NeuralNetwork.instance_count)   # 2 — both objects share this
```

### When a Class Attribute Gets "Shadowed"

This is a subtle and important behavior:

```python
m1 = NeuralNetwork([32, 16])
m1.framework = "PyTorch"          # creates a new INSTANCE attribute on m1

print(m1.framework)               # "PyTorch" — instance attribute shadows class attribute
print(m2.framework)               # still "NumPy" — m2's class attribute unchanged
print(NeuralNetwork.framework)    # still "NumPy" — class attribute unchanged
```

Python's lookup order: **instance dict → class dict → parent class dict**

### Mutable Class Attributes — A Common Bug!

```python
class BadModel:
    history = []   # ❌ MUTABLE class attribute — shared by all instances!

    def add_loss(self, loss):
        self.history.append(loss)   # modifies the SHARED list!


m1 = BadModel()
m2 = BadModel()

m1.add_loss(0.5)
print(m2.history)    # [0.5] ← m2 is affected! BUG!
```

**Fix**: put mutable data in `__init__`:

```python
class GoodModel:
    def __init__(self):
        self.history = []   # ✅ each instance gets its OWN list

    def add_loss(self, loss):
        self.history.append(loss)

m1 = GoodModel()
m2 = GoodModel()

m1.add_loss(0.5)
print(m2.history)    # [] ← unaffected ✅
```

---

### Instance Attributes

Defined inside `__init__` (or other methods) using `self.name = value`. Each object gets its **own copy**.

```python
class Model:
    def __init__(self, name, lr, epochs):
        self.name = name        # instance attribute
        self.lr = lr            # instance attribute
        self.epochs = epochs    # instance attribute
        self.weights = None     # instance attribute (not set yet)
        self.trained = False    # instance attribute (state tracker)

    def fit(self, X, y):
        # ... training code ...
        self.weights = [0.1, 0.2]  # set during training
        self.trained = True        # update state


m1 = Model("LR Model", lr=0.01, epochs=100)
m2 = Model("NN Model", lr=0.001, epochs=500)

# m1 and m2 have completely independent attribute values
print(m1.name, m1.lr)    # LR Model 0.01
print(m2.name, m2.lr)    # NN Model 0.001
```

---

## 11. The `__dict__` Attribute

Every Python object has a `__dict__` attribute that is a **dictionary of all instance attributes**.

```python
class Model:
    framework = "NumPy"   # class attribute

    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.weights = None


m = Model(lr=0.01, epochs=100)

# Instance dict — only instance attributes
print(m.__dict__)
# {'lr': 0.01, 'epochs': 100, 'weights': None}

# Class dict — class-level things
print(Model.__dict__)
# {'framework': 'NumPy', '__init__': <function ...>, ...}
```

### Why is This Useful?

```python
# You can dynamically inspect an object's state
def log_model_state(model):
    print("Model State:")
    for key, value in model.__dict__.items():
        print(f"  {key}: {value}")

log_model_state(m)
# Model State:
#   lr: 0.01
#   epochs: 100
#   weights: None
```

Also used for saving/loading models:

```python
import json

def save_params(model, filepath):
    # Only serializable values
    params = {k: v for k, v in model.__dict__.items() if isinstance(v, (int, float, str, bool))}
    with open(filepath, 'w') as f:
        json.dump(params, f)
```

---

## 12. Dynamic Class and Instance Attributes

Python lets you add or remove attributes **at runtime** — not just in `__init__`.

### Adding Attributes Dynamically

```python
class Experiment:
    def __init__(self, name):
        self.name = name


exp = Experiment("ResNet Training")

# Add attributes dynamically — works!
exp.accuracy = 0.95
exp.loss = 0.12
exp.training_time = 3600  # seconds

print(exp.__dict__)
# {'name': 'ResNet Training', 'accuracy': 0.95, 'loss': 0.12, 'training_time': 3600}
```

### Using `setattr` and `getattr`

When the attribute name is a string (e.g., from a config file):

```python
config = {
    "lr": 0.001,
    "batch_size": 32,
    "optimizer": "adam",
    "dropout": 0.3
}

class Model:
    def __init__(self):
        self.weights = None

model = Model()

# Set attributes programmatically
for key, value in config.items():
    setattr(model, key, value)

# Access programmatically
print(getattr(model, "lr"))           # 0.001
print(getattr(model, "hidden", 128))  # 128 (default if not found)

# Check if attribute exists
print(hasattr(model, "optimizer"))    # True
print(hasattr(model, "momentum"))     # False

# Delete an attribute
delattr(model, "dropout")
print(hasattr(model, "dropout"))      # False
```

### Adding Class Attributes Dynamically

```python
class Config:
    pass

Config.max_epochs = 1000      # add to the class itself
Config.default_lr = 0.001

c = Config()
print(c.max_epochs)    # 1000 — instance inherits from class
```

---

## 13. Property and Descriptor-Based Attributes

### The Problem: Validating Attribute Values

What if someone sets a negative learning rate?

```python
model.lr = -5    # Should be an error! But plain attributes allow this.
```

### Solution 1: `@property` (Getter)

A property lets you define a **method that looks like an attribute** from the outside.

```python
class Model:
    def __init__(self, lr):
        self._lr = lr    # store in "private" attribute

    @property
    def lr(self):
        """This method is called when you access model.lr"""
        print("[DEBUG] Getting lr")
        return self._lr


model = Model(lr=0.01)
print(model.lr)       # calls the property getter → prints [DEBUG], returns 0.01
print(model.lr)       # same — it looks like attribute access!
```

### Solution 2: `@property` + `@name.setter` (Getter + Setter)

```python
class Model:
    def __init__(self, lr, epochs):
        self.lr = lr            # uses the setter below
        self.epochs = epochs

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("Learning rate must be a number")
        if value <= 0:
            raise ValueError(f"Learning rate must be > 0, got {value}")
        self._lr = value   # store after validation

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, value):
        if not isinstance(value, int):
            raise TypeError("Epochs must be an integer")
        if value < 1:
            raise ValueError(f"Epochs must be >= 1, got {value}")
        self._epochs = value


model = Model(lr=0.01, epochs=100)
print(model.lr)       # 0.01

model.lr = 0.001      # calls setter — OK
# model.lr = -0.1    # ValueError: Learning rate must be > 0
# model.lr = "fast"  # TypeError: Learning rate must be a number
```

### Solution 3: `@property` + Deleter

```python
class Model:
    def __init__(self, lr):
        self._lr = lr

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, value):
        self._lr = value

    @lr.deleter
    def lr(self):
        print("Deleting lr")
        del self._lr


model = Model(lr=0.01)
del model.lr        # calls the deleter
# print(model.lr)   # AttributeError — _lr no longer exists
```

### Computed Properties

Properties can compute their value on the fly:

```python
class TrainingResults:
    def __init__(self, correct, total):
        self.correct = correct
        self.total = total

    @property
    def accuracy(self):
        """Computed automatically — no need to store separately."""
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    @property
    def error_rate(self):
        return 1 - self.accuracy


results = TrainingResults(correct=950, total=1000)
print(results.accuracy)     # 0.95 — computed on access
print(results.error_rate)   # 0.05 — computed on access

results.correct = 980
print(results.accuracy)     # 0.98 — automatically updated!
```

### Descriptors (Advanced)

A descriptor is a class that implements `__get__`, `__set__`, and/or `__delete__`. Properties are actually descriptors under the hood. You use descriptors to create reusable validation logic.

```python
class PositiveFloat:
    """A descriptor that enforces positive float values."""

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self   # accessing from class, not instance
        return getattr(obj, self.private_name, None)

    def __set__(self, obj, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.name} must be a number, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"{self.name} must be positive, got {value}")
        setattr(obj, self.private_name, float(value))


class NeuralNetwork:
    learning_rate = PositiveFloat()   # descriptor
    dropout = PositiveFloat()         # same descriptor, different attribute

    def __init__(self, lr, dropout):
        self.learning_rate = lr       # validated by descriptor
        self.dropout = dropout        # validated by descriptor


nn = NeuralNetwork(lr=0.001, dropout=0.3)
print(nn.learning_rate)    # 0.001

# nn.learning_rate = -0.1  # ValueError
# nn.dropout = "high"      # TypeError
```

**Why descriptors?** Instead of writing `@property` validation for every attribute, you define the rule once in a descriptor class and reuse it everywhere.

---

## 14. Lightweight Classes With `__slots__`

### The Problem: Memory Overhead

By default, each instance has a `__dict__` (a Python dictionary). Dictionaries have overhead. If you create **millions of objects** (e.g., one per training sample), this adds up.

```python
import sys

class NormalModel:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class SlottedModel:
    __slots__ = ['x', 'y']   # declare all allowed attributes

    def __init__(self, x, y):
        self.x = x
        self.y = y


n = NormalModel(1.0, 2.0)
s = SlottedModel(1.0, 2.0)

print(sys.getsizeof(n.__dict__))    # ~232 bytes (dictionary overhead)
print(hasattr(s, '__dict__'))       # False — no __dict__!
print(sys.getsizeof(s))             # much smaller
```

### Full Example with Slots

```python
class DataPoint:
    """Represents a single training sample — created millions of times."""
    __slots__ = ['features', 'label', 'weight']

    def __init__(self, features, label, weight=1.0):
        self.features = features
        self.label = label
        self.weight = weight


# Creating 1 million points
points = [DataPoint([i, i*2], i % 2) for i in range(1_000_000)]
# Slots version uses ~40% less memory than without slots
```

### Restrictions of `__slots__`

```python
p = DataPoint([1, 2], 0)

# ❌ Cannot add new attributes not in __slots__
# p.name = "sample_1"   # AttributeError!

# ❌ No __dict__
# print(p.__dict__)      # AttributeError!

# ✅ Can only use declared slots
p.features = [3, 4]    # fine
p.weight = 2.0         # fine
```

### When to Use `__slots__`

- Creating **millions** of small objects (e.g., data points, graph nodes, tokens).
- You know the attributes in advance and won't need to add new ones dynamically.
- Memory is a concern (e.g., loading large datasets into objects).

---

## 15. Instance Methods With `self`

An instance method is any function defined inside a class that takes `self` as its first parameter.

```python
class Model:
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.history = {'loss': [], 'val_loss': []}

    def fit(self, X, y, X_val=None, y_val=None):
        """Train the model. 'self' gives access to all instance data."""
        import numpy as np
        n, p = X.shape
        self.weights = np.zeros(p)
        bias = 0.0

        for epoch in range(self.epochs):
            # Forward pass
            y_pred = X @ self.weights + bias

            # Loss
            loss = np.mean((y_pred - y) ** 2)
            self.history['loss'].append(loss)    # 'self' accesses history

            # Gradients
            dw = (2 / n) * X.T @ (y_pred - y)
            db = (2 / n) * np.sum(y_pred - y)

            # Update using self.lr
            self.weights -= self.lr * dw
            bias -= self.lr * db

        self._bias = bias   # save final bias

    def predict(self, X):
        """self.weights was set by fit()"""
        if self.weights is None:
            raise RuntimeError("Call fit() before predict()")
        return X @ self.weights + self._bias

    def score(self, X, y):
        """Uses predict(), which uses self — chaining methods."""
        import numpy as np
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot    # R² score

    def summary(self):
        """Describes the model."""
        print(f"Model: lr={self.lr}, epochs={self.epochs}")
        if self.weights is not None:
            print(f"Weights: {self.weights}")
            print(f"Final loss: {self.history['loss'][-1]:.4f}")
        else:
            print("Not yet trained.")
```

### How `self` Works

When you call `model.fit(X, y)`, Python translates this to:

```python
Model.fit(model, X, y)   # model is passed as 'self'
```

So `self` is literally the object you called the method on. That's all it is.

---

## 16. Special Methods and Protocols (Dunder Methods)

**Dunder** = **D**ouble **under**score. These are methods like `__init__`, `__str__`, `__len__`, etc. They let your class work with Python's built-in syntax and functions.

### `__str__` and `__repr__`

```python
class Model:
    def __init__(self, name, lr, epochs):
        self.name = name
        self.lr = lr
        self.epochs = epochs
        self.weights = None

    def __str__(self):
        """Human-readable string — used by print()"""
        status = "trained" if self.weights is not None else "untrained"
        return f"{self.name} (lr={self.lr}, epochs={self.epochs}, status={status})"

    def __repr__(self):
        """Unambiguous representation — used in the REPL, logs, debugging"""
        return f"Model(name='{self.name}', lr={self.lr}, epochs={self.epochs})"


m = Model("ResNet", 0.001, 100)
print(m)          # ResNet (lr=0.001, epochs=100, status=untrained)
print(str(m))     # same as print(m)
print(repr(m))    # Model(name='ResNet', lr=0.001, epochs=100)
m                 # in REPL, shows repr
```

### `__len__`

```python
class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)


import numpy as np
ds = Dataset(np.random.randn(1000, 10), np.random.randint(0, 2, 1000))
print(len(ds))    # 1000 — uses __len__
```

### `__getitem__` — Makes Objects Subscriptable (Indexable)

This is how PyTorch `Dataset` works!

```python
class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """Allows dataset[0], dataset[1:5], etc."""
        return self.X[idx], self.y[idx]


import numpy as np
X = np.random.randn(100, 4)
y = np.random.randint(0, 3, 100)

ds = Dataset(X, y)
print(len(ds))            # 100
x0, y0 = ds[0]            # first sample
x_batch, y_batch = ds[0:32]  # first 32 samples (slicing)
```

### `__contains__` — The `in` Operator

```python
class Vocabulary:
    def __init__(self, words):
        self.words = set(words)

    def __contains__(self, word):
        return word in self.words


vocab = Vocabulary(["apple", "banana", "cherry"])
print("apple" in vocab)     # True
print("mango" in vocab)     # False
```

### `__iter__` and `__next__` — Make Objects Iterable

```python
class BatchIterator:
    """Iterate over a dataset in batches — like a DataLoader."""

    def __init__(self, X, y, batch_size=32):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self._index = 0

    def __iter__(self):
        self._index = 0     # reset on each new loop
        return self

    def __next__(self):
        if self._index >= len(self.X):
            raise StopIteration
        start = self._index
        end = min(start + self.batch_size, len(self.X))
        self._index = end
        return self.X[start:end], self.y[start:end]


import numpy as np
X = np.random.randn(100, 4)
y = np.random.randint(0, 2, 100)

loader = BatchIterator(X, y, batch_size=32)
for X_batch, y_batch in loader:
    print(X_batch.shape)    # (32, 4), (32, 4), (32, 4), (4, 4)
```

### `__call__` — Make Objects Callable Like Functions

```python
class Predictor:
    """A trained model that can be called like a function."""

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def __call__(self, X):
        """Called when you do predictor(X)"""
        return X @ self.weights + self.bias


import numpy as np
weights = np.array([1.5, -2.0, 0.5])
bias = 0.1
predictor = Predictor(weights, bias)

X = np.random.randn(5, 3)
result = predictor(X)    # calls __call__ — no need to write predictor.predict(X)
print(result.shape)      # (5,)
```

This is how PyTorch `nn.Module` works — you do `model(X)` not `model.forward(X)`.

### `__eq__`, `__lt__`, `__gt__` — Comparison Operators

```python
class ModelResult:
    def __init__(self, name, accuracy):
        self.name = name
        self.accuracy = accuracy

    def __eq__(self, other):
        return self.accuracy == other.accuracy

    def __lt__(self, other):
        return self.accuracy < other.accuracy

    def __gt__(self, other):
        return self.accuracy > other.accuracy

    def __repr__(self):
        return f"ModelResult({self.name}, acc={self.accuracy:.3f})"


r1 = ModelResult("LR", 0.82)
r2 = ModelResult("RF", 0.91)
r3 = ModelResult("NN", 0.95)

print(r2 > r1)           # True
print(r1 < r3)           # True

results = [r3, r1, r2]
print(sorted(results))   # [ModelResult(LR, 0.820), ModelResult(RF, 0.910), ModelResult(NN, 0.950)]
print(max(results))      # ModelResult(NN, 0.950)
```

### `__enter__` and `__exit__` — Context Managers (`with` statement)

```python
class TrainingSession:
    """Use with 'with' to ensure cleanup happens even if training fails."""

    def __init__(self, model_name):
        self.model_name = model_name

    def __enter__(self):
        print(f"[Session] Starting training: {self.model_name}")
        self.log_file = open(f"{self.model_name}_log.txt", "w")
        return self   # what gets assigned to 'as' variable

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"[Session] Closing session for: {self.model_name}")
        self.log_file.close()
        if exc_type:
            print(f"[Session] Training failed with: {exc_type.__name__}: {exc_val}")
        return False   # don't suppress exceptions


with TrainingSession("ResNet") as session:
    session.log_file.write("Epoch 1: loss=0.5\n")
    session.log_file.write("Epoch 2: loss=0.4\n")
    # File is automatically closed after the block — even if an error occurs!
```

### `__add__`, `__mul__` — Arithmetic Operators

```python
class Vector:
    """A feature vector with custom math operators."""

    def __init__(self, data):
        import numpy as np
        self.data = np.array(data)

    def __add__(self, other):
        return Vector(self.data + other.data)

    def __mul__(self, scalar):
        return Vector(self.data * scalar)

    def __repr__(self):
        return f"Vector({self.data})"


v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])

print(v1 + v2)    # Vector([5, 7, 9])
print(v1 * 2)     # Vector([2, 4, 6])
```

---

## 17. Class Methods With `@classmethod`

A class method receives the **class** (`cls`) as its first argument instead of the instance (`self`). It can access/modify class-level data.

```python
class Dataset:
    _registry = {}    # class attribute — tracks all datasets

    def __init__(self, name, X, y):
        self.name = name
        self.X = X
        self.y = y
        Dataset._registry[name] = self

    @classmethod
    def list_all(cls):
        """Access class-level data — no instance needed."""
        print(f"Registered datasets ({len(cls._registry)}):")
        for name in cls._registry:
            print(f"  - {name}")

    @classmethod
    def get(cls, name):
        """Get a dataset by name."""
        return cls._registry.get(name, None)
```

### Alternative Constructors (The Main Use Case!)

This is the most important use of `@classmethod` in ML — creating objects from different input formats:

```python
import numpy as np
import csv

class Dataset:
    def __init__(self, X, y, name="dataset"):
        self.X = np.array(X)
        self.y = np.array(y)
        self.name = name

    @classmethod
    def from_csv(cls, filepath):
        """Alternative constructor: create Dataset from a CSV file."""
        X, y = [], []
        with open(filepath) as f:
            reader = csv.reader(f)
            next(reader)   # skip header
            for row in reader:
                X.append(row[:-1])
                y.append(row[-1])
        return cls(X, y, name=filepath)   # calls __init__

    @classmethod
    def from_dict(cls, data_dict):
        """Alternative constructor: create Dataset from a dictionary."""
        return cls(
            X=data_dict['features'],
            y=data_dict['labels'],
            name=data_dict.get('name', 'unnamed')
        )

    @classmethod
    def make_classification(cls, n_samples=100, n_features=4, random_state=42):
        """Alternative constructor: generate synthetic data."""
        rng = np.random.default_rng(random_state)
        X = rng.standard_normal((n_samples, n_features))
        y = rng.integers(0, 2, n_samples)
        return cls(X, y, name="synthetic")

    def __repr__(self):
        return f"Dataset(name='{self.name}', samples={len(self.X)}, features={self.X.shape[1]})"


# Three ways to create a Dataset
ds1 = Dataset(X=[[1, 2], [3, 4]], y=[0, 1])                    # standard
ds2 = Dataset.from_csv("data.csv")                              # from file
ds3 = Dataset.make_classification(n_samples=500, n_features=10) # synthetic

print(ds3)  # Dataset(name='synthetic', samples=500, features=10)
```

### `@classmethod` vs Regular Method

```python
class Model:
    default_lr = 0.01

    @classmethod
    def get_default_lr(cls):
        return cls.default_lr    # cls = Model

    def get_lr(self):
        return self.lr           # self = instance


# classmethod: no instance needed
print(Model.get_default_lr())    # 0.01

# regular method: needs instance
m = Model(lr=0.001)
print(m.get_lr())                # 0.001
```

---

## 18. Static Methods With `@staticmethod`

A static method doesn't receive `self` or `cls`. It's just a regular function that logically belongs to the class.

```python
class Metrics:
    """A collection of evaluation metrics."""

    @staticmethod
    def accuracy(y_true, y_pred):
        import numpy as np
        return np.mean(y_true == y_pred)

    @staticmethod
    def precision(y_true, y_pred, positive_class=1):
        import numpy as np
        tp = np.sum((y_pred == positive_class) & (y_true == positive_class))
        fp = np.sum((y_pred == positive_class) & (y_true != positive_class))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @staticmethod
    def recall(y_true, y_pred, positive_class=1):
        import numpy as np
        tp = np.sum((y_pred == positive_class) & (y_true == positive_class))
        fn = np.sum((y_pred != positive_class) & (y_true == positive_class))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @staticmethod
    def f1_score(y_true, y_pred):
        p = Metrics.precision(y_true, y_pred)
        r = Metrics.recall(y_true, y_pred)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


import numpy as np
y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 0, 1, 0, 1])

# Call on class — no instance needed
print(Metrics.accuracy(y_true, y_pred))    # 0.833
print(Metrics.f1_score(y_true, y_pred))    # some value

# Can also call on instance (but rarely needed)
m = Metrics()
print(m.accuracy(y_true, y_pred))          # same result
```

### When to Use Each Method Type

| Type | First Arg | Use When |
|------|-----------|----------|
| Instance method | `self` | Needs to access/modify **instance** data |
| `@classmethod` | `cls` | Needs to access/modify **class** data, or alternative constructors |
| `@staticmethod` | nothing | Utility function logically related to the class, but doesn't need instance or class data |

---

## 19. Getter and Setter Methods vs Properties

In other languages (Java, C++), you write explicit `getX()` / `setX()` methods. Python has a better way: **properties**.

### Java-Style (Avoid in Python)

```python
# ❌ Un-Pythonic — don't write getters/setters like Java
class Model:
    def __init__(self, lr):
        self._lr = lr

    def get_lr(self):          # explicit getter
        return self._lr

    def set_lr(self, value):   # explicit setter
        if value > 0:
            self._lr = value


model = Model(0.01)
print(model.get_lr())    # 0.01
model.set_lr(0.001)      # sets lr
```

### Pythonic Way — Use `@property`

```python
# ✅ Pythonic — use @property
class Model:
    def __init__(self, lr):
        self.lr = lr     # uses the setter

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"lr must be a positive number, got {value}")
        self._lr = value


model = Model(0.01)
print(model.lr)       # 0.01 — looks like attribute access
model.lr = 0.001      # looks like attribute assignment — but runs validation!
# model.lr = -1       # ValueError
```

The external API stays clean (`model.lr`) while the implementation adds validation internally.

---

## 20. Complete Example — Summarizing Class Syntax

Let's build a full, real `KNNClassifier` combining everything we've covered:

```python
import numpy as np
from collections import Counter


class KNNClassifier:
    """
    K-Nearest Neighbors Classifier.
    A complete example showing all class features.
    """

    # Class attributes
    _instances = 0
    supported_metrics = ['euclidean', 'manhattan']

    def __init__(self, k=3, metric='euclidean'):
        # Instance attributes with validation
        self.k = k
        self.metric = metric
        self._X_train = None
        self._y_train = None
        self._fitted = False
        KNNClassifier._instances += 1

    # ----- Properties -----

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"k must be a positive integer, got {value}")
        self._k = value

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, value):
        if value not in KNNClassifier.supported_metrics:
            raise ValueError(f"metric must be one of {KNNClassifier.supported_metrics}")
        self._metric = value

    # ----- Instance methods -----

    def fit(self, X, y):
        """Store training data."""
        self._X_train = np.array(X)
        self._y_train = np.array(y)
        self._fitted = True
        return self   # enables method chaining: model.fit(X, y).predict(X_test)

    def predict(self, X):
        """Predict class labels."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        X = np.array(X)
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        """Internal: predict a single sample."""
        distances = self._compute_distances(x, self._X_train)
        k_indices = np.argsort(distances)[:self._k]
        k_labels = self._y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)[0][0]
        return most_common

    def _compute_distances(self, x, X):
        """Internal: compute distances from x to all training points."""
        if self._metric == 'euclidean':
            return np.sqrt(np.sum((X - x) ** 2, axis=1))
        elif self._metric == 'manhattan':
            return np.sum(np.abs(X - x), axis=1)

    def score(self, X, y):
        """Return accuracy on test set."""
        return float(np.mean(self.predict(X) == np.array(y)))

    # ----- Special methods -----

    def __repr__(self):
        return f"KNNClassifier(k={self._k}, metric='{self._metric}')"

    def __str__(self):
        status = "fitted" if self._fitted else "not fitted"
        return f"KNN(k={self._k}, metric={self._metric}, status={status})"

    def __call__(self, X):
        """Allow calling the object like a function."""
        return self.predict(X)

    # ----- Class methods -----

    @classmethod
    def instance_count(cls):
        return cls._instances

    @classmethod
    def from_config(cls, config: dict):
        """Alternative constructor from a config dict."""
        return cls(k=config.get('k', 3), metric=config.get('metric', 'euclidean'))

    # ----- Static methods -----

    @staticmethod
    def _validate_data(X, y):
        X, y = np.array(X), np.array(y)
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length: {len(X)} != {len(y)}")
        return X, y


# ========== Usage ==========
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=4, random_state=42)
X_train, X_test = X[:160], X[160:]
y_train, y_test = y[:160], y[160:]

# Standard creation
knn = KNNClassifier(k=5, metric='euclidean')
print(knn)           # KNN(k=5, metric=euclidean, status=not fitted)

# Fit and predict (method chaining)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)
print(preds)

# Callable syntax
preds2 = knn(X_test)     # calls __call__ → predict

# Score
print(f"Accuracy: {knn.score(X_test, y_test):.3f}")

# Alternative constructor
config = {'k': 3, 'metric': 'manhattan'}
knn2 = KNNClassifier.from_config(config)
knn2.fit(X_train, y_train)
print(f"KNN2 Accuracy: {knn2.score(X_test, y_test):.3f}")

# Class method
print(f"Total instances created: {KNNClassifier.instance_count()}")    # 2

# Property validation
# knn.k = 0        # ValueError
# knn.metric = 'cosine'  # ValueError
```

---

## 21. Debugging Python Classes

### Using `print` and `__repr__`

The simplest debugging tool:

```python
class Model:
    def __repr__(self):
        return f"Model(lr={self.lr}, epochs={self.epochs}, trained={self.weights is not None})"

m = Model(0.01, 100)
print(m)   # Model(lr=0.01, epochs=100, trained=False)
```

### Using `vars()` and `dir()`

```python
m = KNNClassifier(k=3)

# All instance attributes
print(vars(m))     # same as m.__dict__

# All attributes AND methods (including inherited ones)
print(dir(m))      # very long list

# Filter to non-dunder items
public_attrs = [a for a in dir(m) if not a.startswith('__')]
print(public_attrs)
```

### Using `isinstance()` and `type()`

```python
m = KNNClassifier(k=3)

print(type(m))              # <class '__main__.KNNClassifier'>
print(type(m).__name__)     # 'KNNClassifier'
print(isinstance(m, KNNClassifier))   # True
print(isinstance(m, object))          # True — everything is an object!
```

### Using Python's `inspect` Module

```python
import inspect

class Model:
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y): pass
    def predict(self, X): pass

# Get all methods
methods = inspect.getmembers(Model, predicate=inspect.isfunction)
for name, func in methods:
    print(name, inspect.signature(func))

# Get source code
print(inspect.getsource(Model.fit))

# Get class hierarchy
print(inspect.getmro(Model))   # (Model, object)
```

### Using `pdb` (Python Debugger) — Setting Breakpoints

```python
class Trainer:
    def train(self, model, X, y):
        for epoch in range(model.epochs):
            import pdb; pdb.set_trace()   # breakpoint — program pauses here
            loss = self._compute_loss(model, X, y)
            # In pdb: type 'p loss' to print, 'n' for next line, 'c' to continue
```

Or in Python 3.7+:

```python
breakpoint()   # same as pdb.set_trace() — cleaner
```

### Logging Class State

```python
import logging

logging.basicConfig(level=logging.DEBUG)

class Model:
    def __init__(self, lr):
        self.lr = lr
        logging.debug(f"Model created with lr={lr}")

    def fit(self, X, y):
        logging.info(f"Training started: {len(X)} samples")
        # ...
        logging.info("Training complete")
```

---

## 22. Data Classes

A **dataclass** is a class whose main purpose is to hold data. Python's `@dataclass` decorator auto-generates `__init__`, `__repr__`, and `__eq__` for you.

### Without `@dataclass` (tedious)

```python
class TrainingConfig:
    def __init__(self, lr, epochs, batch_size, optimizer):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer

    def __repr__(self):
        return (f"TrainingConfig(lr={self.lr}, epochs={self.epochs}, "
                f"batch_size={self.batch_size}, optimizer={self.optimizer})")

    def __eq__(self, other):
        return (self.lr == other.lr and self.epochs == other.epochs and
                self.batch_size == other.batch_size and self.optimizer == other.optimizer)
```

### With `@dataclass` (clean!)

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TrainingConfig:
    lr: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    optimizer: str = "adam"
    # ✅ mutable default — use field(default_factory=...)
    layers: List[int] = field(default_factory=lambda: [128, 64])
    model_name: Optional[str] = None

    # You can still add methods
    def total_steps(self, dataset_size: int) -> int:
        return (dataset_size // self.batch_size) * self.epochs


# __init__, __repr__, __eq__ are auto-generated!
cfg = TrainingConfig(lr=0.0001, epochs=50)
print(cfg)   # TrainingConfig(lr=0.0001, epochs=50, batch_size=32, ...)
print(cfg.total_steps(10000))   # 15625

cfg2 = TrainingConfig(lr=0.0001, epochs=50)
print(cfg == cfg2)   # True — __eq__ compares all fields
```

### Frozen Dataclasses (Immutable)

```python
@dataclass(frozen=True)
class ModelID:
    """Immutable — use as dict key or in sets."""
    name: str
    version: str
    timestamp: float

    def __hash__(self):
        return hash((self.name, self.version, self.timestamp))


m_id = ModelID("ResNet", "1.0", 1234567890.0)
# m_id.name = "VGG"   # FrozenInstanceError — can't modify!

# Can use as dict key (because it's hashable)
results = {m_id: 0.95}
```

### `@dataclass(order=True)`

```python
@dataclass(order=True)
class Checkpoint:
    """Auto-generates __lt__, __gt__, __le__, __ge__ based on field order."""
    epoch: int
    val_loss: float
    model_path: str = field(compare=False)   # exclude from comparison


c1 = Checkpoint(10, 0.25, "model_epoch10.pkl")
c2 = Checkpoint(20, 0.18, "model_epoch20.pkl")

print(c1 < c2)       # True (epoch 10 < epoch 20)
print(min(c1, c2))   # Checkpoint(epoch=10, ...)
```

### `post_init` — Computed Fields After `__init__`

```python
from dataclasses import dataclass

@dataclass
class Dataset:
    X: object
    y: object
    n_samples: int = field(init=False)
    n_features: int = field(init=False)

    def __post_init__(self):
        """Called after __init__ — compute derived values."""
        import numpy as np
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.n_samples, self.n_features = self.X.shape


import numpy as np
ds = Dataset(X=np.random.randn(100, 5), y=np.random.randint(0, 2, 100))
print(ds.n_samples)    # 100 — computed automatically
print(ds.n_features)   # 5
```

---

## 23. Enumerations

An **Enum** is a set of named constants. In ML, use them for categorical choices that shouldn't be arbitrary strings.

```python
from enum import Enum, auto

class Optimizer(Enum):
    SGD = "sgd"
    ADAM = "adam"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"


class LossFunction(Enum):
    MSE = auto()        # auto() assigns 1, 2, 3, ...
    CROSS_ENTROPY = auto()
    HUBER = auto()
    MAE = auto()


class ActivationFn(Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
```

### Using Enums

```python
# Access
print(Optimizer.ADAM)          # Optimizer.ADAM
print(Optimizer.ADAM.value)    # "adam"
print(Optimizer.ADAM.name)     # "ADAM"

# Comparison — Enums compare by identity
print(Optimizer.ADAM == Optimizer.ADAM)    # True
print(Optimizer.ADAM == Optimizer.SGD)     # False
print(Optimizer.ADAM == "adam")            # False  ← won't accidentally match strings

# Iteration
for opt in Optimizer:
    print(opt.name, "->", opt.value)

# Convert from string (useful when reading configs)
opt = Optimizer("adam")      # Optimizer.ADAM
opt = Optimizer["ADAM"]      # Optimizer.ADAM  (by name)
```

### Using in a Class

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    optimizer: Optimizer = Optimizer.ADAM
    loss: LossFunction = LossFunction.CROSS_ENTROPY
    activation: ActivationFn = ActivationFn.RELU
    lr: float = 0.001


cfg = ModelConfig(optimizer=Optimizer.SGD)

if cfg.optimizer == Optimizer.ADAM:
    print("Use Adam-specific settings")
elif cfg.optimizer == Optimizer.SGD:
    print("Use SGD with momentum")

# No risk of typos like "adm" or "ADAm" — Enum catches it at definition time
```

---

## 24. Simple Inheritance

Inheritance lets one class **inherit** all attributes and methods from another class, then add or change things.

**Syntax:** `class Child(Parent):`

```python
class BaseModel:
    """All ML models share this base."""

    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self._fitted = False

    def fit(self, X, y):
        raise NotImplementedError("Subclasses must implement fit()")

    def predict(self, X):
        raise NotImplementedError("Subclasses must implement predict()")

    def score(self, X, y):
        """This method is shared — subclasses get it for free."""
        import numpy as np
        return float(np.mean(self.predict(X) == y))

    def __repr__(self):
        return f"{self.__class__.__name__}(lr={self.lr}, epochs={self.epochs})"


class LinearClassifier(BaseModel):
    """Inherits from BaseModel — gets __init__, score, __repr__ for free."""

    def fit(self, X, y):
        import numpy as np
        # simple perceptron-like update
        n, p = X.shape
        self.weights = np.zeros(p)
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                pred = 1 if np.dot(self.weights, xi) >= 0 else 0
                self.weights += self.lr * (yi - pred) * xi
        self._fitted = True
        return self

    def predict(self, X):
        import numpy as np
        return (X @ self.weights >= 0).astype(int)


lc = LinearClassifier(lr=0.01, epochs=50)
print(lc)         # LinearClassifier(lr=0.01, epochs=50) — uses BaseModel's __repr__!
```

### `super()` — Calling the Parent's Method

```python
class NeuralNetwork(BaseModel):
    def __init__(self, lr, epochs, hidden_layers):
        super().__init__(lr, epochs)    # call parent's __init__
        self.hidden_layers = hidden_layers   # add new attribute
        self.weights = []

    def fit(self, X, y):
        # ... neural network training ...
        self._fitted = True
        return self

    def predict(self, X):
        # ... forward pass ...
        pass
```

Without `super().__init__(lr, epochs)`, the parent's `__init__` wouldn't run and `self.lr`, `self.epochs` wouldn't be set.

---

## 25. Class Hierarchies

A hierarchy is multiple levels of inheritance:

```python
class BaseModel:
    """Root of all models."""
    def fit(self, X, y): ...
    def predict(self, X): ...
    def score(self, X, y): ...


class SupervisedModel(BaseModel):
    """For labeled data."""
    def cross_validate(self, X, y, folds=5): ...


class UnsupervisedModel(BaseModel):
    """For unlabeled data."""
    def fit_transform(self, X): ...


class Classifier(SupervisedModel):
    """For classification tasks."""
    def predict_proba(self, X): ...
    def confusion_matrix(self, X, y): ...


class Regressor(SupervisedModel):
    """For regression tasks."""
    def predict_interval(self, X, alpha=0.05): ...


class NeuralNetClassifier(Classifier):
    """Specific neural network for classification."""
    def __init__(self, layers, lr):
        self.layers = layers
        self.lr = lr

    def fit(self, X, y): ...
    def predict(self, X): ...
    def predict_proba(self, X): ...


nn_clf = NeuralNetClassifier([128, 64, 32], lr=0.001)
# nn_clf inherits:
# fit, predict, score (BaseModel)
# cross_validate (SupervisedModel)
# predict_proba, confusion_matrix (Classifier)
# Its own: __init__, fit, predict, predict_proba (overrides)
```

### Visualized

```
BaseModel
├── SupervisedModel
│   ├── Classifier
│   │   ├── NeuralNetClassifier ← you are here
│   │   ├── SVMClassifier
│   │   └── RandomForestClassifier
│   └── Regressor
│       ├── LinearRegressor
│       └── RandomForestRegressor
└── UnsupervisedModel
    ├── KMeans
    └── PCA
```

---

## 26. Extended vs Overridden Methods

### Override — Replace Parent's Method Completely

```python
class BaseModel:
    def predict(self, X):
        print("BaseModel: basic prediction")
        return X.mean(axis=1)


class AdvancedModel(BaseModel):
    def predict(self, X):
        # Completely replaces the parent's predict
        print("AdvancedModel: advanced prediction")
        return X @ self.weights + self.bias
```

### Extend — Add to Parent's Method (Using `super()`)

```python
class BaseModel:
    def fit(self, X, y):
        self._fitted = True
        print(f"Base: Training on {len(X)} samples")
        # ... base training ...


class LoggingModel(BaseModel):
    def fit(self, X, y):
        import time
        start = time.time()

        super().fit(X, y)   # ← run the parent's fit first

        # Add extra behavior on top
        elapsed = time.time() - start
        print(f"LoggingModel: Training took {elapsed:.2f}s")
        self._train_time = elapsed
        return self


lm = LoggingModel()
lm.fit(X_train, y_train)
# Base: Training on 100 samples
# LoggingModel: Training took 0.01s
```

### Real ML Example — Override `predict` for Probability Thresholding

```python
class BinaryClassifier(BaseModel):
    def predict_proba(self, X):
        """Returns probabilities between 0 and 1."""
        pass

    def predict(self, X, threshold=0.5):
        """Override: threshold probabilities."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


class CalibratedClassifier(BinaryClassifier):
    def predict_proba(self, X):
        """Override: calibrate probabilities."""
        raw_proba = super().predict_proba(X)   # get base proba
        # Apply calibration (e.g., Platt scaling)
        return self._calibrate(raw_proba)

    def _calibrate(self, proba):
        # Calibration logic
        return proba   # simplified
```

---

## 27. Multiple Inheritance

Python allows a class to inherit from **multiple parents**.

```python
class Saveable:
    """Mixin: gives save/load ability."""
    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Saved to {path}")

    @classmethod
    def load(cls, path):
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)


class Loggable:
    """Mixin: gives logging ability."""
    def log(self, message):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {self.__class__.__name__}: {message}")


class BaseModel:
    def __init__(self, lr):
        self.lr = lr


class SmartModel(BaseModel, Saveable, Loggable):
    """Inherits from three parents."""

    def __init__(self, lr):
        super().__init__(lr)   # calls BaseModel.__init__

    def fit(self, X, y):
        self.log("Training started")      # from Loggable
        # ... training ...
        self.log("Training complete")
        self.save("model.pkl")            # from Saveable


sm = SmartModel(lr=0.01)
sm.fit(X_train, y_train)
# [12:34:56] SmartModel: Training started
# [12:34:56] SmartModel: Training complete
# Saved to model.pkl

loaded = SmartModel.load("model.pkl")    # from Saveable
```

---

## 28. Method Resolution Order (MRO)

When multiple parents define the same method, Python needs to decide which one to use. The **MRO** is the order Python follows.

Python uses the **C3 Linearization algorithm**: roughly, it goes left-to-right through the parents, then up to grandparents.

```python
class A:
    def hello(self):
        print("A.hello")

class B(A):
    def hello(self):
        print("B.hello")

class C(A):
    def hello(self):
        print("C.hello")

class D(B, C):
    pass


d = D()
d.hello()    # B.hello — Python picks B first (left to right)

# Check the MRO
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

print(D.mro())    # same, as a list
```

### MRO in Practice (ML Example)

```python
class Serializable:
    def save(self, path):
        print("Serializable.save: using pickle")

class JSONSerializable(Serializable):
    def save(self, path):
        print("JSONSerializable.save: using JSON")

class BinarySerializable(Serializable):
    def save(self, path):
        print("BinarySerializable.save: using binary")

class MLModel(JSONSerializable, BinarySerializable):
    pass


model = MLModel()
model.save("model")    # JSONSerializable.save — leftmost parent wins!

print(MLModel.__mro__)
# MLModel → JSONSerializable → BinarySerializable → Serializable → object
```

### `super()` Follows the MRO

When you call `super()` in any class in the hierarchy, it doesn't always call the direct parent — it calls the **next class in the MRO**:

```python
class Base:
    def process(self):
        print("Base.process")

class Mixin1(Base):
    def process(self):
        print("Mixin1.process")
        super().process()   # next in MRO

class Mixin2(Base):
    def process(self):
        print("Mixin2.process")
        super().process()   # next in MRO

class FinalModel(Mixin1, Mixin2):
    def process(self):
        print("FinalModel.process")
        super().process()

FinalModel().process()
# FinalModel.process
# Mixin1.process
# Mixin2.process
# Base.process
```

---

## 29. Mixin Classes

A **Mixin** is a class that provides specific functionality to be added to another class, without being meant for standalone use. It's one of the most powerful patterns in ML frameworks.

### Logging Mixin

```python
import logging

class LoggingMixin:
    """Add logging to any model by inheriting this."""

    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger

    def log_info(self, msg):
        self.logger.info(msg)

    def log_debug(self, msg):
        self.logger.debug(msg)

    def log_warning(self, msg):
        self.logger.warning(msg)
```

### Serialization Mixin

```python
import pickle, json

class SerializationMixin:
    """Add save/load to any class."""

    def to_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def to_dict(self):
        """Export hyperparameters as dict (for JSON)."""
        return {k: v for k, v in self.__dict__.items()
                if isinstance(v, (int, float, str, bool, list))}

    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
```

### Evaluation Mixin

```python
import numpy as np
from sklearn.model_selection import cross_val_score

class EvaluationMixin:
    """Add cross-validation and metrics to any model."""

    def cross_validate(self, X, y, cv=5):
        scores = cross_val_score(self, X, y, cv=cv)
        return {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
            'accuracy': np.mean(y_pred == y),
            'samples': len(y)
        }
```

### Combining Mixins

```python
class RandomForestClassifier(LoggingMixin, SerializationMixin, EvaluationMixin, BaseModel):
    """Gets all mixin features for free."""

    def __init__(self, n_trees=100, max_depth=None, lr=0.01, epochs=1):
        super().__init__(lr, epochs)
        self.n_trees = n_trees
        self.max_depth = max_depth

    def fit(self, X, y):
        self.log_info(f"Training {self.n_trees} trees...")
        # ... tree building code ...
        self.log_info("Training complete")
        return self

    def predict(self, X):
        # ... ensemble prediction ...
        pass


rf = RandomForestClassifier(n_trees=100)
rf.fit(X_train, y_train)
# INFO: Training 100 trees...
# INFO: Training complete

# Evaluation
cv_results = rf.cross_validate(X_train, y_train, cv=5)
print(f"CV Accuracy: {cv_results['mean']:.3f} ± {cv_results['std']:.3f}")

# Save/load
rf.to_pickle("rf_model.pkl")
loaded_rf = RandomForestClassifier.from_pickle("rf_model.pkl")
```

---

## 30. Benefits of Using Inheritance

1. **Code Reuse**: Write `score()`, `cross_validate()`, `save()` once in a base class; all models get them.
2. **Consistency**: All models follow the same interface (`fit`, `predict`, `score`).
3. **Easy Extension**: Add a new model by subclassing and implementing only what's different.
4. **Polymorphism**: A function that accepts a `BaseModel` works with any subclass.

```python
def run_experiment(model: BaseModel, X_train, y_train, X_test, y_test):
    """Works with ANY model that inherits BaseModel — you don't need to know the specific type."""
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

# Works with ALL of these:
run_experiment(LinearClassifier(), ...)
run_experiment(NeuralNetClassifier([128, 64], 0.001), ...)
run_experiment(KNNClassifier(k=5), ...)
```

---

## 31. Composition

**Composition** means building a class by **having** other objects as attributes, instead of **being** a subclass of them.

> **Inheritance = "is a"** (NeuralNetwork IS A BaseModel)
> **Composition = "has a"** (Trainer HAS A model and HAS A dataset)

```python
class Model:
    def fit(self, X, y): ...
    def predict(self, X): ...


class Optimizer:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, weights, gradients):
        return weights - self.lr * gradients


class LossFunction:
    def compute(self, y_true, y_pred): ...
    def gradient(self, y_true, y_pred): ...

class MSELoss(LossFunction):
    def compute(self, y_true, y_pred):
        import numpy as np
        return np.mean((y_true - y_pred) ** 2)

    def gradient(self, y_true, y_pred):
        import numpy as np
        return 2 * (y_pred - y_true) / len(y_true)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self._best_loss = float('inf')
        self._counter = 0

    def should_stop(self, val_loss):
        if val_loss < self._best_loss - self.min_delta:
            self._best_loss = val_loss
            self._counter = 0
        else:
            self._counter += 1
        return self._counter >= self.patience


class Trainer:
    """Has-a model, optimizer, loss, and early stopping — Composition."""

    def __init__(self, model, optimizer, loss_fn, early_stopping=None):
        self.model = model              # has a Model
        self.optimizer = optimizer      # has an Optimizer
        self.loss_fn = loss_fn          # has a LossFunction
        self.early_stopping = early_stopping  # has an EarlyStopping (optional)
        self.history = {'loss': [], 'val_loss': []}

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100):
        for epoch in range(epochs):
            y_pred = self.model.predict(X_train)
            loss = self.loss_fn.compute(y_train, y_pred)
            grad = self.loss_fn.gradient(y_train, y_pred)

            self.model.weights = self.optimizer.step(self.model.weights, grad)
            self.history['loss'].append(loss)

            if X_val is not None:
                val_pred = self.model.predict(X_val)
                val_loss = self.loss_fn.compute(y_val, val_pred)
                self.history['val_loss'].append(val_loss)

                if self.early_stopping and self.early_stopping.should_stop(val_loss):
                    print(f"Early stopping at epoch {epoch}")
                    break

        return self


# Usage — very flexible, mix and match components
trainer = Trainer(
    model=LinearRegression(),
    optimizer=Optimizer(lr=0.001),
    loss_fn=MSELoss(),
    early_stopping=EarlyStopping(patience=10)
)
trainer.train(X_train, y_train, X_val, y_val, epochs=1000)
```

### Composition vs Inheritance

```python
# ❌ Bad design — inheritance when composition is better
class TrainerWithSGD(SGDOptimizer, MSELoss, BaseModel):
    pass   # a Trainer IS NOT an optimizer or a loss

# ✅ Good design — composition
class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model       # HAS a model
        self.optimizer = optimizer  # HAS an optimizer
        self.loss_fn = loss_fn   # HAS a loss function
```

**Favor composition over inheritance** when the relationship is "has-a" not "is-a".

---

## 32. Delegation

**Delegation** means forwarding method calls to an attribute object, effectively "delegating" work to it.

```python
class ModelPipeline:
    """Delegates preprocessing to a transformer and prediction to a model."""

    def __init__(self, transformer, model):
        self.transformer = transformer
        self.model = model

    def fit(self, X, y):
        # Delegate to transformer
        X_transformed = self.transformer.fit_transform(X)
        # Delegate to model
        self.model.fit(X_transformed, y)
        return self

    def predict(self, X):
        # Delegate to transformer, then model
        X_transformed = self.transformer.transform(X)
        return self.model.predict(X_transformed)

    def score(self, X, y):
        return self.model.score(self.predict(X), y)


# Real usage (with sklearn components)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = ModelPipeline(
    transformer=StandardScaler(),
    model=LogisticRegression()
)
pipeline.fit(X_train, y_train)
print(pipeline.score(X_test, y_test))
```

### `__getattr__` — Automatic Delegation

```python
class WrappedModel:
    """Wraps any model and automatically delegates unknown attributes to it."""

    def __init__(self, model):
        self._model = model
        self.call_count = 0

    def predict(self, X):
        self.call_count += 1   # track calls
        return self._model.predict(X)

    def __getattr__(self, name):
        """Called when attribute is NOT found on WrappedModel.
           Automatically delegate to the inner model."""
        return getattr(self._model, name)


from sklearn.ensemble import RandomForestClassifier as SKLearnRF

wrapped = WrappedModel(SKLearnRF(n_estimators=100))
wrapped.fit(X_train, y_train)      # delegates to _model.fit (via __getattr__)
preds = wrapped.predict(X_test)    # uses WrappedModel.predict (tracked)
print(wrapped.n_estimators)        # delegates to _model.n_estimators (100)
print(wrapped.call_count)          # 1
```

---

## 33. Dependency Injection

**Dependency Injection (DI)** means giving an object its dependencies from the outside rather than creating them internally.

```python
# ❌ Without DI — hard-coded dependencies
class Trainer:
    def __init__(self):
        self.optimizer = AdamOptimizer(lr=0.001)   # hard-coded!
        self.loss = CrossEntropyLoss()              # hard-coded!
        self.logger = FileLogger("training.log")   # hard-coded!

# Can't easily test, can't swap optimizer, can't change logger


# ✅ With DI — inject dependencies
class Trainer:
    def __init__(self, model, optimizer, loss_fn, logger=None):
        self.model = model
        self.optimizer = optimizer    # injected
        self.loss_fn = loss_fn        # injected
        self.logger = logger or PrintLogger()  # injected with default


# Flexible — swap any component
trainer_adam = Trainer(
    model=NeuralNetwork([128, 64]),
    optimizer=AdamOptimizer(lr=0.001),
    loss_fn=CrossEntropyLoss(),
    logger=WandbLogger()
)

trainer_sgd = Trainer(
    model=NeuralNetwork([128, 64]),
    optimizer=SGDOptimizer(lr=0.01, momentum=0.9),  # different optimizer!
    loss_fn=FocalLoss(gamma=2),                       # different loss!
    logger=TensorboardLogger()                         # different logger!
)

# For tests — inject mocks
trainer_test = Trainer(
    model=MockModel(),
    optimizer=MockOptimizer(),
    loss_fn=MockLoss(),
    logger=None    # no logging in tests
)
```

### DI with Factory Functions

```python
def create_trainer(config: dict) -> Trainer:
    """Factory: builds a Trainer from a config dict — full DI."""

    optimizers = {
        'adam': AdamOptimizer,
        'sgd': SGDOptimizer,
        'rmsprop': RMSPropOptimizer
    }
    losses = {
        'cross_entropy': CrossEntropyLoss,
        'mse': MSELoss,
        'focal': FocalLoss
    }

    model = NeuralNetwork(layers=config['layers'])
    optimizer = optimizers[config['optimizer']](lr=config['lr'])
    loss_fn = losses[config['loss']]()

    return Trainer(model, optimizer, loss_fn)


config = {'layers': [128, 64], 'optimizer': 'adam', 'lr': 0.001, 'loss': 'cross_entropy'}
trainer = create_trainer(config)
```

---

## 34. Abstract Base Classes (ABCs) and Interfaces

An **Abstract Base Class** defines a **contract**: subclasses must implement certain methods, or they get an error.

```python
from abc import ABC, abstractmethod

class BaseEstimator(ABC):
    """All ML estimators must implement these methods."""

    @abstractmethod
    def fit(self, X, y):
        """Train the model — MUST be implemented by all subclasses."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions — MUST be implemented."""
        pass

    # Non-abstract — shared default implementation
    def score(self, X, y):
        import numpy as np
        return float(np.mean(self.predict(X) == y))

    def fit_predict(self, X, y, X_test):
        """Uses abstract methods — works because they're guaranteed to exist."""
        self.fit(X, y)
        return self.predict(X_test)


# ❌ Cannot instantiate an ABC directly
# model = BaseEstimator()    # TypeError!

# ❌ Subclass that doesn't implement all abstract methods
class IncompleteModel(BaseEstimator):
    def fit(self, X, y):
        pass
    # forgot predict!

# incomplete = IncompleteModel()   # TypeError: Can't instantiate abstract class IncompleteModel
#                                  # with abstract method predict


# ✅ Complete implementation — works fine
class ConcreteModel(BaseEstimator):
    def fit(self, X, y):
        self._X, self._y = X, y
        return self

    def predict(self, X):
        # simplest model: predict most common class
        import numpy as np
        from collections import Counter
        most_common = Counter(self._y).most_common(1)[0][0]
        return np.full(len(X), most_common)


m = ConcreteModel()  # works!
m.fit(X_train, y_train)
print(m.score(X_test, y_test))  # uses BaseEstimator's score
```

### Abstract Properties

```python
from abc import ABC, abstractmethod

class BaseTransformer(ABC):

    @property
    @abstractmethod
    def n_features_in(self):
        """Must expose input feature count after fitting."""
        pass

    @abstractmethod
    def fit(self, X): ...

    @abstractmethod
    def transform(self, X): ...

    def fit_transform(self, X):
        """Template method — uses abstract methods."""
        self.fit(X)
        return self.transform(X)


class Normalizer(BaseTransformer):
    def fit(self, X):
        import numpy as np
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._n_features = X.shape[1]
        return self

    def transform(self, X):
        return (X - self._mean) / self._std

    @property
    def n_features_in(self):
        return self._n_features
```

### Abstract Classes as Interfaces

Python doesn't have a separate `interface` keyword. An ABC with all abstract methods acts as one:

```python
class ModelInterface(ABC):
    """Interface contract — all methods are abstract."""

    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def predict(self, X): ...

    @abstractmethod
    def predict_proba(self, X): ...

    @abstractmethod
    def get_params(self): ...

    @abstractmethod
    def set_params(self, **params): ...
```

Any class implementing all these methods satisfies the interface — regardless of inheritance!

### `__subclasshook__` — Virtual Subclasses (Advanced)

```python
from abc import ABC, abstractmethod

class Fittable(ABC):
    @abstractmethod
    def fit(self, X, y): ...

    @classmethod
    def __subclasshook__(cls, C):
        """A class is considered a Fittable if it has a 'fit' method."""
        if cls is Fittable:
            if any("fit" in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented


class SklearnModel:
    def fit(self, X, y): ...
    def predict(self, X): ...


# SklearnModel is considered a Fittable even without explicit inheritance!
print(isinstance(SklearnModel(), Fittable))   # True
```

---

## 35. Unlocking Polymorphism With Common Interfaces

**Polymorphism** means "many forms". One function/code works with different types of objects, as long as they follow the same interface.

```python
from abc import ABC, abstractmethod
import numpy as np

class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def predict(self, X): ...

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class KNNClassifier(BaseClassifier):
    def __init__(self, k=5):
        self.k = k
    def fit(self, X, y):
        self._X, self._y = X, y
        return self
    def predict(self, X):
        from collections import Counter
        preds = []
        for x in X:
            dists = np.sqrt(np.sum((self._X - x)**2, axis=1))
            idx = np.argsort(dists)[:self.k]
            preds.append(Counter(self._y[idx]).most_common(1)[0][0])
        return np.array(preds)


class NaiveBayes(BaseClassifier):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.means_ = {c: X[y==c].mean(axis=0) for c in self.classes_}
        self.vars_ = {c: X[y==c].var(axis=0) + 1e-9 for c in self.classes_}
        self.priors_ = {c: np.mean(y==c) for c in self.classes_}
        return self
    def predict(self, X):
        def log_likelihood(x, c):
            return np.sum(-0.5*np.log(2*np.pi*self.vars_[c]) -
                         (x - self.means_[c])**2 / (2*self.vars_[c]))
        preds = []
        for x in X:
            scores = {c: log_likelihood(x, c) + np.log(self.priors_[c])
                      for c in self.classes_}
            preds.append(max(scores, key=scores.get))
        return np.array(preds)


class DummyClassifier(BaseClassifier):
    def fit(self, X, y):
        from collections import Counter
        self.majority_ = Counter(y).most_common(1)[0][0]
        return self
    def predict(self, X):
        return np.full(len(X), self.majority_)
```

### Polymorphism in Action

```python
def benchmark_models(models, X_train, y_train, X_test, y_test):
    """
    This function works with ANY object that is a BaseClassifier.
    That's polymorphism — same code, many types.
    """
    results = {}
    for model in models:
        model.fit(X_train, y_train)                    # same interface!
        acc = model.score(X_test, y_test)             # same interface!
        results[model.__class__.__name__] = acc
        print(f"{model.__class__.__name__:25s} → Accuracy: {acc:.4f}")
    return results


# Completely different algorithms — same function handles all of them
models = [
    KNNClassifier(k=5),
    NaiveBayes(),
    DummyClassifier()
]

results = benchmark_models(models, X_train, y_train, X_test, y_test)
best_model_name = max(results, key=results.get)
print(f"\nBest model: {best_model_name}")
```

### Duck Typing — Polymorphism Without Inheritance

Python also supports polymorphism through "duck typing": if it has `fit` and `predict`, treat it as a model — no inheritance needed.

```python
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """Works with ANY object that has fit() and predict() — no ABC required."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return np.mean(preds == y_test)


# Works with our custom classes
print(train_and_evaluate(KNNClassifier(k=3), X_train, y_train, X_test, y_test))

# Also works with sklearn models — no shared parent class needed!
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
print(train_and_evaluate(LogisticRegression(), X_train, y_train, X_test, y_test))
print(train_and_evaluate(SVC(), X_train, y_train, X_test, y_test))
```

---

## 36. Frequently Asked Questions

### Q1: What's the difference between a class and an object?
A class is the blueprint (definition). An object (instance) is a concrete thing created from that blueprint. `Model` is a class. `model = Model()` creates an object.

### Q2: Can I have a class without `__init__`?
Yes. Python will use `object.__init__` (the base of all classes), which does nothing. You'd just create instances with no initial data: `m = MyClass()`.

### Q3: Why does `self` need to be the first parameter?
Python requires it by convention. When you call `model.fit(X, y)`, Python translates it to `Model.fit(model, X, y)` — the instance is always passed first. The name `self` is a convention, not enforced (you could technically use `this` or `me`), but never do — everyone uses `self`.

### Q4: When should I use `@property` vs just setting the attribute directly?
Use `@property` when you need **validation**, **computed values**, or want to **hide implementation details**. For simple data storage without validation, a plain attribute is fine.

### Q5: Is multiple inheritance bad?
Not inherently. The problems arise when the hierarchy is confusing (the "diamond problem"). The key rule: use multiple inheritance primarily for **mixins** — small, focused classes that add one capability. Avoid inheriting from two large, complex classes.

### Q6: `@classmethod` vs `@staticmethod` — when to use which?
- `@classmethod` when you need access to the **class** itself (e.g., class attributes, alternative constructors that call `cls(...)`).
- `@staticmethod` when the method is logically related to the class but doesn't need access to class or instance data — a utility function.

### Q7: What is the MRO and why does it matter?
MRO (Method Resolution Order) determines which class's method is called when multiple parent classes define the same method. Use `ClassName.__mro__` to inspect it. It matters in multiple inheritance to understand which `fit()` or `predict()` actually runs.

### Q8: Composition vs Inheritance — which to prefer?
Start with composition ("has-a"). Use inheritance ("is-a") only when there's a genuine hierarchical relationship. A good test: say the sentence out loud — "Trainer IS A BaseModel" — if it sounds wrong, use composition.

### Q9: What are `__slots__` good for in ML?
Creating **millions of small objects** (one per data point, token, graph node) where memory is critical. Slots eliminate the per-instance `__dict__`, saving ~40% memory. Trade-off: you can't add new attributes dynamically.

### Q10: Are ABCs necessary?
Not strictly — Python's duck typing means any object with the right methods works. But ABCs add **explicit documentation** of required methods, **immediate errors** when a required method is missing, and better IDE support. For production ML libraries, ABCs are strongly recommended.

---

## Quick Reference Card

```
Class definition:        class Name:
Constructor:             def __init__(self, ...):
Instance attribute:      self.attr = value
Class attribute:         attr = value  (outside methods)
Instance method:         def method(self, ...):
Class method:            @classmethod / def method(cls, ...):
Static method:           @staticmethod / def method(...):
Property getter:         @property / def attr(self):
Property setter:         @attr.setter / def attr(self, value):
String repr:             __str__ (human) / __repr__ (debug)
Callable objects:        __call__
Indexing:                __getitem__ / __setitem__
Length:                  __len__
Iteration:               __iter__ / __next__
Context manager:         __enter__ / __exit__
Comparison:              __eq__, __lt__, __gt__
Arithmetic:              __add__, __mul__, etc.

Inheritance:             class Child(Parent):
Call parent method:      super().method(...)
Check type:              isinstance(obj, Class)
Get MRO:                 Class.__mro__
Abstract class:          class Name(ABC): + @abstractmethod
Dataclass:               @dataclass
Enum:                    class Name(Enum):
Slots:                   __slots__ = ['a', 'b']
All instance attrs:      obj.__dict__
All attrs + methods:     dir(obj)
```

---

***Author:-Monesh_Patidar_IIT_Kanpur***
