### Python Lists vs. Tuples
Media: Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/star.svg

| Feature | List | Tuple |
| :--- | :--- | :--- |
| **Mutability** | Mutable (elements can be added, removed, or modified) | Immutable (elements cannot be changed after creation) |
| **Syntax** | Defined using square brackets: `[1, 2, 3]` | Defined using parentheses: `(1, 2, 3)` |
| **Memory & Performance** | Uses more memory; slightly slower iteration | Uses less memory; faster iteration and creation |
| **Dictionary Keys** | Cannot be used as dictionary keys (unhashable) | Can be used as dictionary keys (if all elements are hashable) |
| **Primary Use Case** | Best for homogenous collections of data that change dynamically | Best for heterogeneous, fixed collections or read-only data records |

---

### Code Examples
Media: Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/laptop.svg

#### List Example (Mutable)
Media: Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/star.svg
```python
# Creating and modifying a list
fruits = ["apple", "banana"]
https://fruits.append("cherry")
fruits[0] = "avocado"
print(fruits)  # Output: ['avocado', 'banana', 'cherry']
```

#### Tuple Example (Immutable)
```python
# Creating and accessing a tuple
point = (10, 20)
print(point[0])  # Output: 10

# point[0] = 15  # Raises TypeError: 'tuple' object does not support item assignment
```
