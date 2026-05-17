# API Reference

## Main Functions

### encode()

```python
encode(embedding, method="auto", validate=True)
```

Converts an embedding vector into a position-safe string.

**Parameters:**
- **embedding** (*array-like*): The embedding vector to encode. Accepts numpy arrays, lists, or bytes.
- **method** (*str*): Encoding method. Options: "auto", "eq64", "shq64", "t8q64", "zoq64"
- **validate** (*bool*): Validates that input values are within [0, 255]

**Returns:**
- **str**: Position-safe encoded string

**Example:**
```python
import numpy as np
from uubed import encode

embedding = np.random.rand(384).astype(np.float32)
encoded = encode(embedding, method="eq64")
```

### decode()

```python
decode(encoded_string)
```

Reverses Eq64 encoding back to the original embedding vector.

**Parameters:**
- **encoded_string** (*str*): The string to decode

**Returns:**
- **bytes**: Decoded embedding

**Raises:**
- **ValueError**: If the string wasn't encoded with Eq64

**Example:**
```python
decoded = decode(encoded_string)
embedding = np.frombuffer(decoded, dtype=np.uint8)
```

## Encoding Methods

### Eq64 (Full Precision)

```python
from uubed.encoders.eq64 import eq64_encode, eq64_decode

encoded = eq64_encode(embedding_bytes)
decoded = eq64_decode(encoded)
```

Standard base encoding with dot separators every 8 characters for readability.

### Shq64 (SimHash)

```python
from uubed.encoders.shq64 import simhash_q64

encoded = simhash_q64(embedding_array)
```

Locality-sensitive hashing that preserves similarity relationships.

### T8q64 (Top-k)

```python
from uubed.encoders.t8q64 import top_k_q64

encoded = top_k_q64(embedding_array, k=8)
```

Encodes indices of the top-k largest values.

### Zoq64 (Z-order)

```python
from uubed.encoders.zoq64 import z_order_q64

encoded = z_order_q64(embedding_array)
```

Spatial encoding using Morton codes for multi-dimensional queries.

## Low-Level Q64 Codec

### q64_encode()

```python
from uubed.encoders.q64 import q64_encode

encoded = q64_encode(data_bytes)
```

Basic Q64 encoding without dots. Used internally by other encoders.

**Parameters:**
- **data_bytes** (*bytes*): Raw bytes to encode

**Returns:**
- **str**: Q64 encoded string

### q64_decode()

```python
from uubed.encoders.q64 import q64_decode

decoded = q64_decode(encoded_string)
```

Basic Q64 decoding.

**Parameters:**
- **encoded_string** (*str*): Q64 encoded string

**Returns:**
- **bytes**: Decoded bytes

## Constants

### Q64_ALPHABETS

```python
from uubed.encoders.q64 import Q64_ALPHABETS

# Position-dependent alphabets
print(Q64_ALPHABETS[0])  # 'ABCD...XYZ'  (uppercase)
print(Q64_ALPHABETS[1])  # 'abcd...xyz'  (lowercase)
print(Q64_ALPHABETS[2])  # 'AaBb...YyZz' (mixed case)
print(Q64_ALPHABETS[3])  # '0123...+/-_' (digits/symbols)
```

### Q64_REVERSE

```python
from uubed.encoders.q64 import Q64_REVERSE

# Reverse lookup table for decoding
char_value = Q64_REVERSE[position][character]
```

## Native Module Detection

```python
import uubed

if uubed._has_native:
    print("Native Rust acceleration available")
else:
    print("Using pure Python implementation")
```

## Type Hints

uubed includes full type hinting for better IDE support:

```python
from typing import Union, Literal
import numpy as np
from numpy.typing import NDArray

def encode(
    embedding: Union[NDArray[np.float32], list, bytes],
    method: Literal["auto", "eq64", "shq64", "t8q64", "zoq64"] = "auto",
    validate: bool = True
) -> str: ...
```

## Error Handling

Functions raise standard exceptions:

- **ValueError**: Invalid input values or parameters
- **TypeError**: Wrong input types
- **IndexError**: Invalid positions in encoded strings

Example:
```python
try:
    encoded = encode(embedding, method="invalid")
except ValueError as e:
    print(f"Encoding error: {e}")
```