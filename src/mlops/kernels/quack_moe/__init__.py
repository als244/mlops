"""Local grouped expert math shared by SonicMoE DP and Quack EP adapters.

No routing policy or communication lives here. Import exact modules so optional
Quack/CuTe dependencies remain lazy. Providers own custom-op registration.
"""
