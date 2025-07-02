#!/usr/bin/env python3
# this_file: benchmarks/bench_encoders.py
"""Basic benchmark script for encoding performance."""

import time
import numpy as np
from uubed import encode
from uubed.encoders import q64, eq64, shq64, t8q64, zoq64


def benchmark_function(func, *args, iterations=1000):
    """Time a function over multiple iterations."""
    start = time.perf_counter()
    for _ in range(iterations):
        func(*args)
    end = time.perf_counter()
    return (end - start) / iterations


def main():
    """Run benchmarks."""
    # Generate test data
    sizes = [32, 256, 1024]
    
    print("uubed Encoding Performance Benchmarks")
    print("=" * 60)
    print(f"{'Size':<10} {'Method':<15} {'Time (μs)':<15} {'Throughput (MB/s)':<15}")
    print("-" * 60)
    
    for size in sizes:
        data = np.random.randint(0, 256, size, dtype=np.uint8).tolist()
        
        # Benchmark each encoding method
        methods = [
            ("q64", lambda: q64.q64_encode(data)),
            ("eq64", lambda: eq64.eq64_encode(data)),
            ("shq64", lambda: shq64.simhash_q64(data)),
            ("t8q64", lambda: t8q64.top_k_q64(data)),
            ("zoq64", lambda: zoq64.z_order_q64(data)),
        ]
        
        for method_name, method_func in methods:
            # Use fewer iterations for larger sizes
            iters = 1000 if size <= 256 else 100
            
            time_per_op = benchmark_function(method_func, iterations=iters)
            throughput = size / (time_per_op * 1e6)  # MB/s
            
            print(f"{size:<10} {method_name:<15} {time_per_op*1e6:<15.2f} {throughput:<15.2f}")
        
        print()


if __name__ == "__main__":
    main()