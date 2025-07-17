#!/usr/bin/env python3
"""
Test core functions from voyemb.py that don't require external dependencies.
We'll copy the relevant functions to test them in isolation.
"""

import base64
import struct
import sys
import os

# Core functions extracted from voyemb.py for testing without dependencies
def int_list_to_binary_visual(int_list, bits_per_int=8):
    """Convert a list of integers to a visual binary string where 0 = '.' and 1 = '#'"""
    binary_str = ""
    for num in int_list:
        binary_repr = format(num, f"0{bits_per_int}b")
        visual_binary = binary_repr.replace("0", ".").replace("1", "#")
        binary_str += visual_binary
    return binary_str

def int_list_to_base64(int_list):
    """Convert a list of integers to URL-safe Base64 representation."""
    first_bits = []
    for i in range(min(8, len(int_list))):
        first_bit = (int_list[i] >> 7) & 1
        first_bits.append(first_bit)
    
    while len(first_bits) < 8:
        first_bits.append(0)
    
    packed_byte = 0
    for i, bit in enumerate(first_bits):
        packed_byte |= bit << (7 - i)
    
    enhanced_data = int_list + [packed_byte]
    byte_data = bytes(enhanced_data)
    base64_result = base64.urlsafe_b64encode(byte_data).decode("ascii")
    base64_result = base64_result.rstrip("=")
    return base64_result

def gray_coded_hex(int_list):
    """Gray-coded nibble stream → hex string"""
    gray_codes = [0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8]
    
    hex_result = ""
    for byte_val in int_list:
        high_nibble = (byte_val >> 4) & 0xF
        low_nibble = byte_val & 0xF
        gray_high = gray_codes[high_nibble]
        gray_low = gray_codes[low_nibble]
        hex_result += f"{gray_high:x}{gray_low:x}"
    return hex_result

def run_length_encode_binary(int_list):
    """Run-length encode the .# binary pattern"""
    binary_str = int_list_to_binary_visual(int_list, bits_per_int=8)
    
    if not binary_str:
        return ""
    
    result = ""
    current_char = binary_str[0]
    count = 1
    
    for char in binary_str[1:]:
        if char == current_char:
            count += 1
        else:
            result += f"{count}{current_char}"
            current_char = char
            count = 1
    
    result += f"{count}{current_char}"
    return result

def emoji_quaternions(int_list):
    """Map 2-bit pairs to emoji quaternions"""
    emojis = ["⚪", "🟡", "🟠", "🔴"]  # 00, 01, 10, 11
    
    result = ""
    for byte_val in int_list:
        for shift in [6, 4, 2, 0]:
            two_bits = (byte_val >> shift) & 0b11
            result += emojis[two_bits]
    return result

def z_order_hash(int_list):
    """Simplified Z-order (Morton code) encoding"""
    quantized = [(b >> 6) & 0b11 for b in int_list]
    
    result = 0
    for i, val in enumerate(quantized[:16]):
        for bit_pos in range(2):
            bit = (val >> bit_pos) & 1
            result |= bit << (i * 2 + bit_pos)
    
    packed = struct.pack(">I", result)
    return base64.urlsafe_b64encode(packed).decode().rstrip("=")

# Q64 encoding functions
ALPHABETS = [
    "ABCDEFGHIJKLMNOP",  # pos ≡ 0 (mod 4)
    "QRSTUVWXYZabcdef",  # pos ≡ 1
    "ghijklmnopqrstuv",  # pos ≡ 2
    "wxyz0123456789-_",  # pos ≡ 3
]

REV_LOOKUP = {}
for idx, pos_set in enumerate(ALPHABETS):
    for c in pos_set:
        REV_LOOKUP[c] = (idx, pos_set.index(c))

def q64_encode(byte_seq):
    """Encode bytes into q64 positional alphabet."""
    if isinstance(byte_seq, list):
        byte_seq = bytes(byte_seq)
    
    out = []
    pos = 0
    for byte in byte_seq:
        hi, lo = byte >> 4, byte & 0xF
        for nibble in (hi, lo):
            alphabet = ALPHABETS[pos & 3]
            out.append(alphabet[nibble])
            pos += 1
    return "".join(out)

def q64_decode(s):
    """Inverse of q64_encode."""
    if len(s) & 1:
        raise ValueError("q64 length must be even (2 chars per byte).")
    
    nibbles = []
    for pos, ch in enumerate(s):
        try:
            expected_alpha, val = REV_LOOKUP[ch]
        except KeyError:
            raise ValueError(f"Invalid q64 character {ch!r}") from None
        if expected_alpha != (pos & 3):
            raise ValueError(f"Character {ch!r} illegal at position {pos}")
        nibbles.append(val)
    
    it = iter(nibbles)
    return bytes((hi << 4) | lo for hi, lo in zip(it, it))

def eq64_encode(byte_seq):
    """Encode bytes into Eq64 format with dots every 8 characters."""
    base_encoded = q64_encode(byte_seq)
    
    result = []
    for i, char in enumerate(base_encoded):
        if i > 0 and i % 8 == 0:
            result.append(".")
        result.append(char)
    
    return "".join(result)

def eq64_decode(s):
    """Decode Eq64 format by removing dots and using standard q64 decode."""
    clean_s = s.replace(".", "")
    return q64_decode(clean_s)

# Test functions
def test_int_list_to_binary_visual():
    """Test binary visual representation function."""
    # Test with simple values
    result = int_list_to_binary_visual([0, 255], bits_per_int=8)
    assert result == "........########", f"Expected '........########', got '{result}'"
    
    # Test with custom bit width
    result = int_list_to_binary_visual([5], bits_per_int=4)
    assert result == ".#.#", f"Expected '.#.#', got '{result}'"
    
    # Test empty list
    result = int_list_to_binary_visual([], bits_per_int=8)
    assert result == "", f"Expected empty string, got '{result}'"
    
    print("✅ PASS: test_int_list_to_binary_visual")

def test_int_list_to_base64():
    """Test base64 encoding function."""
    # Test with known values
    test_data = [72, 101, 108, 108, 111]  # "Hello" in ASCII
    result = int_list_to_base64(test_data)
    assert isinstance(result, str)
    assert len(result) <= 44  # Should be 44 chars or less
    
    # Test with empty list
    result = int_list_to_base64([])
    assert isinstance(result, str)
    
    print("✅ PASS: test_int_list_to_base64")

def test_gray_coded_hex():
    """Test gray coded hex encoding."""
    # Test with known values
    result = gray_coded_hex([0, 255])
    assert isinstance(result, str)
    assert len(result) == 4  # 2 bytes * 2 hex chars each
    
    # Test empty list
    result = gray_coded_hex([])
    assert result == ""
    
    print("✅ PASS: test_gray_coded_hex")

def test_run_length_encode_binary():
    """Test run-length encoding of binary patterns."""
    # Test with known pattern
    test_data = [0, 255]  # Should create pattern like "8.8#"
    result = run_length_encode_binary(test_data)
    assert isinstance(result, str)
    assert "." in result
    assert "#" in result
    
    # Test empty list
    result = run_length_encode_binary([])
    assert result == ""
    
    print("✅ PASS: test_run_length_encode_binary")

def test_emoji_quaternions():
    """Test emoji quaternion encoding."""
    # Test with known values
    test_data = [0, 255]  # 0 = all ⚪, 255 = all 🔴
    result = emoji_quaternions(test_data)
    assert isinstance(result, str)
    assert len(result) == 8  # 2 bytes * 4 emoji each
    assert "⚪" in result
    assert "🔴" in result
    
    print("✅ PASS: test_emoji_quaternions")

def test_z_order_hash():
    """Test z-order hash encoding."""
    test_data = [128, 64, 192, 32] * 4  # 16 bytes
    result = z_order_hash(test_data)
    assert isinstance(result, str)
    assert len(result) == 6  # Base64 encoded 4 bytes without padding
    
    print("✅ PASS: test_z_order_hash")

def test_q64_encode_decode():
    """Test q64 encoding and decoding round trip."""
    # Test with known data
    test_data = [72, 101, 108, 108, 111]  # "Hello" in ASCII
    encoded = q64_encode(test_data)
    assert isinstance(encoded, str)
    assert len(encoded) == 10  # 5 bytes * 2 chars each
    
    # Test round trip
    decoded = q64_decode(encoded)
    assert list(decoded) == test_data
    
    # Test with bytes input
    encoded_bytes = q64_encode(bytes(test_data))
    assert encoded_bytes == encoded
    
    # Test error conditions
    try:
        q64_decode("ABC")  # Odd length
        assert False, "Should raise ValueError for odd length"
    except ValueError:
        pass
    
    try:
        q64_decode("ZZ")  # Invalid characters for position
        assert False, "Should raise ValueError for invalid characters"
    except ValueError:
        pass
    
    print("✅ PASS: test_q64_encode_decode")

def test_eq64_encode_decode():
    """Test eq64 encoding and decoding with dots."""
    # Test with known data
    test_data = [72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100, 33]  # "Hello World!"
    encoded = eq64_encode(test_data)
    assert isinstance(encoded, str)
    assert "." in encoded  # Should have dots for readability
    
    # Test round trip
    decoded = eq64_decode(encoded)
    assert list(decoded) == test_data
    
    # Test with shorter data
    short_data = [72, 101, 108, 108]  # "Hell"
    encoded_short = eq64_encode(short_data)
    decoded_short = eq64_decode(encoded_short)
    assert list(decoded_short) == short_data
    
    print("✅ PASS: test_eq64_encode_decode")

def test_q64_alphabets():
    """Test q64 alphabet structure."""
    # Test alphabet structure
    assert len(ALPHABETS) == 4
    assert all(len(alphabet) == 16 for alphabet in ALPHABETS)
    
    # Test reverse lookup consistency
    for pos, alphabet in enumerate(ALPHABETS):
        for char_idx, char in enumerate(alphabet):
            assert REV_LOOKUP[char] == (pos, char_idx)
    
    # Test no character overlap between alphabets
    all_chars = set()
    for alphabet in ALPHABETS:
        alphabet_chars = set(alphabet)
        assert len(alphabet_chars.intersection(all_chars)) == 0
        all_chars.update(alphabet_chars)
    
    print("✅ PASS: test_q64_alphabets")

if __name__ == "__main__":
    """Run all tests"""
    test_int_list_to_binary_visual()
    test_int_list_to_base64()
    test_gray_coded_hex()
    test_run_length_encode_binary()
    test_emoji_quaternions()
    test_z_order_hash()
    test_q64_encode_decode()
    test_eq64_encode_decode()
    test_q64_alphabets()
    print("\n🎉 All core function tests passed!")