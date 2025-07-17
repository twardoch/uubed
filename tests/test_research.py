#!/usr/bin/env python3
"""
Test for research code in the uubed project.
This tests the voyemb.py research script to ensure it works correctly.
"""

import subprocess
import sys
import os

def test_voyemb_execution():
    """Test that voyemb.py executes without errors using uv run."""
    script_path = os.path.join(os.path.dirname(__file__), "..", "research", "voyemb.py")
    
    assert os.path.exists(script_path), f"Script {script_path} not found"
    
    try:
        # First try with uv run -s (which handles dependencies automatically)
        result = subprocess.run(["uv", "run", "-s", script_path], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            assert len(result.stdout) > 1000, "Script output too short, may not be working correctly"
            print("✅ PASS: voyemb.py executed successfully with uv run")
            print(f"Output length: {len(result.stdout)} characters")
            return
            
        # If uv run fails, try with direct python execution and skip if missing deps
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0 and "ModuleNotFoundError" in result.stderr:
            print("⚠️  SKIP: voyemb.py requires dependencies not available in test environment")
            return
            
        assert result.returncode == 0, f"Script returned exit code {result.returncode}, stderr: {result.stderr}"
        assert len(result.stdout) > 1000, "Script output too short, may not be working correctly"
        
        print("✅ PASS: voyemb.py executed successfully")
        print(f"Output length: {len(result.stdout)} characters")
            
    except subprocess.TimeoutExpired:
        assert False, "Script timed out"
    except FileNotFoundError:
        # If uv is not available, skip the test
        print("⚠️  SKIP: uv not available, cannot test voyemb.py")
        return
    except Exception as e:
        assert False, f"Error running voyemb.py: {e}"


# Import the functions to test them directly
# Add the research directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'research'))

try:
    # Try importing the functions - if dependencies are missing, skip these tests
    from voyemb import (
        int_list_to_binary_visual,
        int_list_to_base64,
        simhash_bits,
        simhash_slug,
        gray_coded_hex,
        top_k_indices,
        run_length_encode_binary,
        emoji_quaternions,
        z_order_hash,
        q64_encode,
        q64_decode,
        simhash_q64,
        top_k_q64,
        z_order_q64,
        eq64_encode,
        eq64_decode,
        ALPHABETS,
        REV_LOOKUP
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  SKIP: Cannot import functions due to missing dependencies: {e}")
    IMPORTS_AVAILABLE = False


def test_int_list_to_binary_visual():
    """Test binary visual representation function."""
    if not IMPORTS_AVAILABLE:
        print("⚠️  SKIP: test_int_list_to_binary_visual - dependencies not available")
        return
    
    # Test with simple values
    result = int_list_to_binary_visual([0, 255], bits_per_int=8)
    assert result == "........########"
    
    # Test with custom bit width
    result = int_list_to_binary_visual([5], bits_per_int=4)
    assert result == ".#.#"
    
    # Test empty list
    result = int_list_to_binary_visual([], bits_per_int=8)
    assert result == ""


def test_int_list_to_base64():
    """Test base64 encoding function."""
    if not IMPORTS_AVAILABLE:
        print("⚠️  SKIP: test_int_list_to_base64 - dependencies not available")
        return
    
    # Test with known values
    test_data = [72, 101, 108, 108, 111]  # "Hello" in ASCII
    result = int_list_to_base64(test_data)
    assert isinstance(result, str)
    assert len(result) <= 44  # Should be 44 chars or less
    
    # Test with empty list
    result = int_list_to_base64([])
    assert isinstance(result, str)


def test_simhash_bits():
    """Test simhash bits generation."""
    if not IMPORTS_AVAILABLE:
        print("⚠️  SKIP: test_simhash_bits - dependencies not available")
        return
    
    # Test with known data
    test_data = [128, 64, 192, 32]
    result = simhash_bits(test_data, planes=64)
    assert len(result) == 64
    assert all(bit in [0, 1] for bit in result)
    
    # Test reproducibility (should be deterministic with same seed)
    result2 = simhash_bits(test_data, planes=64)
    assert (result == result2).all()


def test_simhash_slug():
    """Test simhash slug generation."""
    if not IMPORTS_AVAILABLE:
        print("⚠️  SKIP: test_simhash_slug - dependencies not available")
        return
    
    test_data = [128, 64, 192, 32]
    result = simhash_slug(test_data)
    assert isinstance(result, str)
    assert len(result) == 11  # Should be 11 chars
    
    # Test reproducibility
    result2 = simhash_slug(test_data)
    assert result == result2


def test_gray_coded_hex():
    """Test gray coded hex encoding."""
    if not IMPORTS_AVAILABLE:
        print("⚠️  SKIP: test_gray_coded_hex - dependencies not available")
        return
    
    # Test with known values
    result = gray_coded_hex([0, 255])
    assert isinstance(result, str)
    assert len(result) == 4  # 2 bytes * 2 hex chars each
    
    # Test empty list
    result = gray_coded_hex([])
    assert result == ""


def test_top_k_indices():
    """Test top-k indices extraction."""
    if not IMPORTS_AVAILABLE:
        print("⚠️  SKIP: test_top_k_indices - dependencies not available")
        return
    
    # Test with known data
    test_data = [10, 200, 50, 255, 5, 100, 150, 75, 25, 175]
    result = top_k_indices(test_data, k=3)
    assert isinstance(result, str)
    assert "-" in result  # Should be dash-separated
    
    # Test with k=1
    result = top_k_indices(test_data, k=1)
    assert isinstance(result, str)
    assert "03" in result  # Index 3 has value 255 (highest)


def test_run_length_encode_binary():
    """Test run-length encoding of binary patterns."""
    if not IMPORTS_AVAILABLE:
        print("⚠️  SKIP: test_run_length_encode_binary - dependencies not available")
        return
    
    # Test with known pattern
    test_data = [0, 255]  # Should create pattern like "8.8#"
    result = run_length_encode_binary(test_data)
    assert isinstance(result, str)
    assert "." in result
    assert "#" in result
    
    # Test empty list
    result = run_length_encode_binary([])
    assert result == ""


def test_emoji_quaternions():
    """Test emoji quaternion encoding."""
    if not IMPORTS_AVAILABLE:
        print("⚠️  SKIP: test_emoji_quaternions - dependencies not available")
        return
    
    # Test with known values
    test_data = [0, 255]  # 0 = all ⚪, 255 = all 🔴
    result = emoji_quaternions(test_data)
    assert isinstance(result, str)
    assert len(result) == 8  # 2 bytes * 4 emoji each
    assert "⚪" in result
    assert "🔴" in result


def test_z_order_hash():
    """Test z-order hash encoding."""
    if not IMPORTS_AVAILABLE:
        print("⚠️  SKIP: test_z_order_hash - dependencies not available")
        return
    
    test_data = [128, 64, 192, 32] * 4  # 16 bytes
    result = z_order_hash(test_data)
    assert isinstance(result, str)
    assert len(result) == 6  # Base64 encoded 4 bytes without padding


def test_q64_encode_decode():
    """Test q64 encoding and decoding round trip."""
    if not IMPORTS_AVAILABLE:
        print("⚠️  SKIP: test_q64_encode_decode - dependencies not available")
        return
    
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


def test_simhash_q64():
    """Test simhash with q64 encoding."""
    if not IMPORTS_AVAILABLE:
        print("⚠️  SKIP: test_simhash_q64 - dependencies not available")
        return
    
    test_data = [128, 64, 192, 32]
    result = simhash_q64(test_data)
    assert isinstance(result, str)
    assert len(result) == 16  # 8 bytes * 2 chars each
    
    # Test reproducibility
    result2 = simhash_q64(test_data)
    assert result == result2


def test_top_k_q64():
    """Test top-k with q64 encoding."""
    if not IMPORTS_AVAILABLE:
        print("⚠️  SKIP: test_top_k_q64 - dependencies not available")
        return
    
    test_data = [10, 200, 50, 255, 5, 100, 150, 75, 25, 175]
    result = top_k_q64(test_data, k=8)
    assert isinstance(result, str)
    assert len(result) == 16  # 8 bytes * 2 chars each


def test_z_order_q64():
    """Test z-order with q64 encoding."""
    if not IMPORTS_AVAILABLE:
        print("⚠️  SKIP: test_z_order_q64 - dependencies not available")
        return
    
    test_data = [128, 64, 192, 32] * 4  # 16 bytes
    result = z_order_q64(test_data)
    assert isinstance(result, str)
    assert len(result) == 8  # 4 bytes * 2 chars each


def test_eq64_encode_decode():
    """Test eq64 encoding and decoding with dots."""
    if not IMPORTS_AVAILABLE:
        print("⚠️  SKIP: test_eq64_encode_decode - dependencies not available")
        return
    
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


def test_q64_alphabets():
    """Test q64 alphabet structure."""
    if not IMPORTS_AVAILABLE:
        print("⚠️  SKIP: test_q64_alphabets - dependencies not available")
        return
    
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