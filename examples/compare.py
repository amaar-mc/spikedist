"""Compare two spike trains with both metrics, and show how the parameters act.

Run with: python examples/compare.py
"""

from spikedist import van_rossum, victor_purpura

a = [0.010, 0.025, 0.090]  # spike times in seconds
b = [0.012, 0.030, 0.095]  # a slightly jittered copy of a

print(f"train a: {a}")
print(f"train b: {b}\n")

print("Victor-Purpura distance as the shift cost q increases:")
for cost in (0.0, 50.0, 200.0, 1000.0):
    print(f"  cost={cost:>7.1f}  ->  {victor_purpura(a, b, cost=cost):.4f}")

print("\nvan Rossum distance as the time constant tau increases:")
for tau in (0.002, 0.010, 0.050, 0.200):
    print(f"  tau={tau:>6.3f}   ->  {van_rossum(a, b, tau=tau):.4f}")

print("\nA spike train is distance 0 from itself:")
print(f"  victor_purpura(a, a) = {victor_purpura(a, a, cost=200.0):.4f}")
print(f"  van_rossum(a, a)     = {van_rossum(a, a, tau=0.01):.4f}")
