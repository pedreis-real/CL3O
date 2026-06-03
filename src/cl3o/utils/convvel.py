'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Velocity Conversion Module.

Unit conversion utility for velocity quantities, supporting feet per
second, meters per second, kilometers per hour, miles per hour, knots
and feet per minute. Conversion is performed via a fixed
factors-to-m/s table, so adding a new unit requires only one entry
in _TO_MPS.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import numpy as np

# ================ Module imports ================

# Constants
from cl3o.Constants import TO_MPS


# ========================================================================
# PUBLIC API - Velocity conversion
# ========================================================================

def convvel(
    v:  float | list | np.ndarray,
    ui: str,
    uo: str,
) -> np.ndarray:
    '''
    Convert velocity values from one unit to another.

    Args:
        v:  Input velocity value(s).
        ui: Input unit string. Supported: 'ft/s', 'm/s', 'km/s',
            'in/s', 'km/h', 'mph', 'kts', 'ft/min'.
        uo: Output unit string (same options as ui).

    Returns:
        Converted velocity value(s) as a float ndarray.
    '''
    ui = ui.strip().lower()
    uo = uo.strip().lower()

    if ui not in TO_MPS:
        raise ValueError(
            f'[CL3O] Unknown input velocity unit.\n'
            f'| Unit      : {ui}\n'
            f'| Supported : {list(TO_MPS.keys())}'
        )
    if uo not in TO_MPS:
        raise ValueError(
            f'[CL3O] Unknown output velocity unit.\n'
            f'| Unit      : {uo}\n'
            f'| Supported : {list(TO_MPS.keys())}'
        )

    factor = TO_MPS[ui] / TO_MPS[uo]
    return np.asarray(v, dtype=float) * factor


if __name__ == '__main__':
    result = convvel([30, 100, 250], 'ft/min', 'm/s')
    print("convvel([30, 100, 250], 'ft/min', 'm/s') =", result)
