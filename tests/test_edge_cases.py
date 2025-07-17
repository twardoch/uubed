#!/usr/bin/env python3
"""
Test edge cases and error handling for uubed research functions.
"""

import sys
import os

# Add the test directory to the path to import test_core_functions
sys.path.insert(0, os.path.dirname(__file__))

from test_core_functions import (
    int_list_to_binary_visual,
    int_list_to_base64,
    gray_coded_hex,
    run_length_encode_binary,
    emoji_quaternions,
    z_order_hash,
    q64_encode,
    q64_decode,
    eq64_encode,
    eq64_decode,
    ALPHABETS,
    REV_LOOKUP
)

def test_edge_case_empty_inputs():
    """Test functions with empty inputs."""
    # Test empty list handling
    assert int_list_to_binary_visual([]) == ""
    assert int_list_to_base64([]) != ""  # Should still produce output due to padding
    assert gray_coded_hex([]) == ""
    assert run_length_encode_binary([]) == ""
    assert emoji_quaternions([]) == ""
    assert z_order_hash([]) != ""  # Should still produce output due to struct packing
    
    # Test empty bytes
    assert q64_encode(b'') == ""
    assert q64_encode([]) == ""
    
    print("✅ PASS: test_edge_case_empty_inputs")

def test_edge_case_single_values():
    """Test functions with single values."""
    # Test single byte
    single_byte = [42]
    
    result = int_list_to_binary_visual(single_byte)
    assert len(result) == 8  # 1 byte * 8 bits
    
    result = int_list_to_base64(single_byte)
    assert isinstance(result, str)
    assert len(result) > 0
    
    result = gray_coded_hex(single_byte)
    assert len(result) == 2  # 1 byte * 2 hex chars
    
    result = emoji_quaternions(single_byte)
    assert len(result) == 4  # 1 byte * 4 emoji
    
    result = q64_encode(single_byte)
    assert len(result) == 2  # 1 byte * 2 chars
    
    print("✅ PASS: test_edge_case_single_values")

def test_edge_case_extreme_values():
    """Test functions with extreme values (0 and 255)."""
    extreme_values = [0, 255]
    
    # Test binary visual
    result = int_list_to_binary_visual(extreme_values)
    assert "........" in result  # All zeros
    assert "########" in result  # All ones
    
    # Test gray coded hex
    result = gray_coded_hex(extreme_values)
    assert len(result) == 4  # 2 bytes * 2 hex chars each
    
    # Test emoji quaternions
    result = emoji_quaternions(extreme_values)
    assert "⚪" in result  # From 0 (00)
    assert "🔴" in result  # From 255 (11)
    
    print("✅ PASS: test_edge_case_extreme_values")

def test_edge_case_large_inputs():
    """Test functions with large inputs."""
    # Test with 256 bytes (larger than typical use cases)
    large_input = list(range(256))
    
    result = int_list_to_binary_visual(large_input)
    assert len(result) == 256 * 8  # 256 bytes * 8 bits each
    
    result = gray_coded_hex(large_input)
    assert len(result) == 256 * 2  # 256 bytes * 2 hex chars each
    
    result = emoji_quaternions(large_input)
    assert len(result) == 256 * 4  # 256 bytes * 4 emoji each
    
    result = q64_encode(large_input)
    assert len(result) == 256 * 2  # 256 bytes * 2 chars each
    
    print("✅ PASS: test_edge_case_large_inputs")

def test_q64_decode_error_cases():
    """Test q64_decode error handling."""
    # Test odd length
    try:
        q64_decode("ABC")
        assert False, "Should raise ValueError for odd length"
    except ValueError as e:
        assert "even" in str(e)
    
    # Test invalid character
    try:
        q64_decode("ZZ")  # Z is not in any alphabet
        assert False, "Should raise ValueError for invalid character"
    except ValueError as e:
        assert "illegal at position" in str(e) or "Invalid" in str(e)
    
    # Test character in wrong position
    try:
        q64_decode("QA")  # Q is from alphabet 1, should be at position 1,5,9... not 0
        assert False, "Should raise ValueError for character in wrong position"
    except ValueError as e:
        assert "illegal at position" in str(e)
    
    print("✅ PASS: test_q64_decode_error_cases")

def test_eq64_round_trip_various_sizes():
    """Test eq64 round trip with various data sizes."""
    # Test different sizes to ensure dot placement works correctly
    test_sizes = [1, 4, 8, 12, 16, 20, 32, 64]
    
    for size in test_sizes:
        test_data = list(range(size))
        encoded = eq64_encode(test_data)
        decoded = eq64_decode(encoded)
        
        assert list(decoded) == test_data, f"Round trip failed for size {size}"
        
        # Check that dots are inserted correctly (every 8 characters)
        if size > 4:  # eq64 produces 2 chars per byte, so 4 bytes = 8 chars
            assert "." in encoded, f"Expected dots in encoded string for size {size}"
    
    print("✅ PASS: test_eq64_round_trip_various_sizes")

def test_binary_visual_bit_widths():
    """Test binary visual with different bit widths."""
    # Test different bit widths with appropriate values
    test_cases = [
        (1, [1]),    # 1 bit: can represent 0,1
        (2, [3]),    # 2 bits: can represent 0-3
        (3, [5]),    # 3 bits: can represent 0-7
        (4, [5]),    # 4 bits: can represent 0-15
        (8, [5]),    # 8 bits: can represent 0-255
        (16, [5]),   # 16 bits: can represent 0-65535
    ]
    
    for bits, test_value in test_cases:
        result = int_list_to_binary_visual(test_value, bits_per_int=bits)
        expected_length = bits * len(test_value)
        assert len(result) == expected_length, f"Expected length {expected_length}, got {len(result)} for {bits} bits"
        
        # Check that result contains only . and #
        assert all(c in ".#" for c in result), f"Invalid characters in result: {result}"
    
    print("✅ PASS: test_binary_visual_bit_widths")

def test_run_length_encode_patterns():
    """Test run-length encoding with specific patterns."""
    # Test alternating pattern
    alternating = [0, 255, 0, 255]  # Should produce alternating . and #
    result = run_length_encode_binary(alternating)
    assert "." in result
    assert "#" in result
    
    # Test all zeros
    all_zeros = [0, 0, 0, 0]
    result = run_length_encode_binary(all_zeros)
    assert result.endswith(".")
    
    # Test all ones
    all_ones = [255, 255, 255, 255]
    result = run_length_encode_binary(all_ones)
    assert result.endswith("#")
    
    print("✅ PASS: test_run_length_encode_patterns")

def test_alphabet_coverage():
    """Test that all characters in alphabets are unique and covered."""
    # Check that we can encode/decode all possible nibble values
    for nibble in range(16):
        # Create test data with this nibble value
        test_data = [nibble]
        encoded = q64_encode(test_data)
        decoded = q64_decode(encoded)
        assert list(decoded) == test_data
        
        # Test with nibble in high position
        test_data = [nibble << 4]
        encoded = q64_encode(test_data)
        decoded = q64_decode(encoded)
        assert list(decoded) == test_data
    
    print("✅ PASS: test_alphabet_coverage")

if __name__ == "__main__":
    """Run all edge case tests"""
    test_edge_case_empty_inputs()
    test_edge_case_single_values()
    test_edge_case_extreme_values()
    test_edge_case_large_inputs()
    test_q64_decode_error_cases()
    test_eq64_round_trip_various_sizes()
    test_binary_visual_bit_widths()
    test_run_length_encode_patterns()
    test_alphabet_coverage()
    print("\n🎉 All edge case tests passed!")