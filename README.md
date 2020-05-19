# Symbolic differentiation
Python library for calculating formulas for derivatives. Covers parsing, differentiation, formula simplification and pretty printing.
## Usage
```
import symbolic_differentiation as sd
expression = sd.parse("4*x^2+sin(log(tan(1/x)))")
derivative = sd.derivative(expression)
simplified = sd.simplify(derivative)
print(sd.to_pretty_string(simplified))
```

No libraries needed. See main.py for demo.
# What works
Arithmetic operations: ``+``, ``*``, ``/``, ``^``, ``-``.

Functions: ``sin``, ``cos``, ``tan``, ``log`` (natural log).

Negative numbers don't. (I mean they just don't parse - can write 0-a). But have you ever seen -5 of something? - so I thought ;)

Be explicit with ``*`` sign - can't write ``4(x+3)``.

Simplify will simplify for example: 
1. ``sin(x^2)-(3*sin(x^2)+5)`` to ``-2*sin(x^2)-5``
2. ``2^(x^3/x-x^2)`` to 1
3. ``5*sin(x)*log(5)/sin(x)^2`` to ``5*log(5)/sin(x)``

But will not simplify:
1. ``x/(x^2+x)`` to ``1/(x+1)``
2. ``sin(0)`` to ``0``
