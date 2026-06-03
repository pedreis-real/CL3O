'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Length Conversion Module.

Unit conversion utility for length quantities, supporting feet, meters,
kilometers, inches, statute miles and nautical miles. Conversion is
performed via a fixed factors-to-meters table, so adding a new unit
requires only one entry in _TO_METERS.

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import numpy as np

# ================ Module imports ================

# Constants
from cl3o.Constants import TO_METERS


# ========================================================================
# PUBLIC API - Length conversion
# ========================================================================

def convlength(
    v:  float | list | np.ndarray,
    ui: str,
    uo: str,
) -> np.ndarray:
    '''
    Convert length values from one unit to another.

    Args:
        v:  Input length value(s).
        ui: Input unit string. Supported: 'ft', 'm', 'km', 'in',
            'mi', 'naut mi'.
        uo: Output unit string (same options as ui).

    Returns:
        Converted length value(s) as a float ndarray.
    '''
    ui = ui.strip().lower()
    uo = uo.strip().lower()

    if ui not in TO_METERS:
        raise ValueError(
            f'[CL3O] Unknown input length unit.\n'
            f'| Unit      : {ui}\n'
            f'| Supported : {list(TO_METERS.keys())}'
        )
    if uo not in TO_METERS:
        raise ValueError(
            f'[CL3O] Unknown output length unit.\n'
            f'| Unit      : {uo}\n'
            f'| Supported : {list(TO_METERS.keys())}'
        )

    factor = TO_METERS[ui] / TO_METERS[uo]
    return np.asarray(v, dtype=float) * factor


if __name__ == '__main__':
    result = convlength([3, 10, 20], 'ft', 'm')
    print("convlength([3, 10, 20], 'ft', 'm') =", result)
