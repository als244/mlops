"""Version-checked optional dependency patches; installed sources stay untouched."""


def initialize_moonep():
    """Apply the shared MoonEP compatibility fixes before either backend runs."""
    from . import moonep_compiler, moonep_rank1

    moonep_rank1.apply()
    moonep_compiler.apply()
