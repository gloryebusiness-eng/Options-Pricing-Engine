# Design decisions

Format: what was decided, what alternatives were considered, why this one.

## D-001: src/ layout with an installable package
Alternative: flat scripts at repo root.
Chose src/ so tests import the installed package, not stray local files.
Prevents "works on my machine" import-shadowing bugs.

## D-002: Pricing kernels are pure functions with no I/O
Alternative: classes holding market data as state.
Pure kernels vectorize over NumPy arrays, test in microseconds, and let
numerical methods be swapped behind one interface.