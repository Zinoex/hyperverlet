import sympy
from sympy import diff

k, l, qax, qay, qbx, qby = sympy.symbols('k l qax qay qbx qby')

dist = sympy.sqrt((qax - qbx) ** 2 + (qay - qby) ** 2)
ext = dist - l

potential_energy = (k / 2) * ext ** 2

force_qax = diff(potential_energy, qax)
force_qay = diff(potential_energy, qay)
force_qbx = diff(potential_energy, qbx)
force_qby = diff(potential_energy, qby)

print(force_qax)
print(force_qay)
print(force_qbx)
print(force_qby)
