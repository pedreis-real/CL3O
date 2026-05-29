'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
I / O Utilities Module - Centralized file reading and writing

Single point of file I/O for the entire CL3O project, handling the
following API:
- read_json:
- write_json:
- read_dat_file:
- read_xlsx:
- 

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
import logging
from pathlib import Path
from typing import Optional

import json
import numpy as np
import re   # Regex

# ================ Paths bootstrap ================



# ================ Global variables ================
_ENABLE_INNER_DEBUG_LOGGING = True


# ================ Module logger ================
io_logger = logging.getLogger("cl3o.utils.io")
io_logger.setLevel(logging.DEBUG if _ENABLE_INNER_DEBUG_LOGGING else logging.CRITICAL)

if _ENABLE_INNER_DEBUG_LOGGING and not io_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    io_logger.addHandler(_handler)
    io_logger.propagate = False


# =============================================================================
# Internal helpers
# =============================================================================

# -------- Pathing Helpers --------
def _resolve_path(filepath: str | Path) -> Path:
    '''
    Converts a string or Path to an absolute Path and verifies existence.

    Args:
        filepath: Relative or absolute path to the file.

    Returns:
        Resolved absolute Path object.
    '''
    p = Path(filepath).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p

# -------- Data Converter Helpers --------
def _to_dict(dcls_obj: object) -> dict[str, np.ndarray]:
    '''
    Centralized dataclass object to dictionary converter.

    Converts data as follows:

    @dataclass
    class ClassName():
        ...
        Arg1: np.ndarray = value1
        Arg2: np.ndarray = value2
        Arg3: np.ndarray = value3
    
    --> dict_classname = {
            str(Arg1): value1,
            str(Arg2): value2,
            str(Arg3): value3
        }
    '''
    io_logger.info(
        f"Converting {dcls_obj.__class__.__name__} instance into dictionary"
        f" to serialize JSON file."
    )
    return dcls_obj.__dict__


def _to_dataclass(data: dict, dcls: type) -> object:
    '''
    Centralized dicionary to dataclass object converter.

    Convertion follows the opposite of :_to_dict: API
    '''
    return dcls(**data)


# ---------------- Airfoil Helpers ----------------
def _parse_afl_raw_data(lines: list[str]) -> tuple[np.ndarray, np.ndarray, str]:
    '''
    Parses a .dat file in Selig format.

    Args:
        lines: Raw lines read from the .dat file.

    Returns:
        Tuple (x, y, name) where x and y are 1-D coordinate.
    '''
    data_lines = [
        l.strip() for l in lines
        if l.strip() and not l.strip().startswith(("#", "!"))
    ]

    name = ""
    xy_pairs = []
    for i, line in enumerate(data_lines):
        tokens = line.split()
        if len(tokens) == 2:
            try:
                xy_pairs.append((float(tokens[0]), float(tokens[1])))
            except ValueError:
                if i == 0:
                    name = line
        else:
            if i == 0:
                name = line

    if not xy_pairs:
        raise ValueError("No valid coordinates found in Selig file.")

    arr = np.array(xy_pairs)
    return arr[:, 0], arr[:, 1], name


def _normalize_afl(
    x: np.ndarray,
    y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    chord = np.max(x) - np.min(x)
    if chord < 1e-12:
        raise ValueError(
            f"Computed chord is zero or negative ({chord}). Check the .dat file."
        )

    x_norm = (x - np.min(x)) / chord
    y_norm = y / chord

    return x_norm, y_norm


# ---------------- ExLoads Helpers ----------------
def _parse_txt_xflr5(
    lines: list[str]
) -> tuple[list[str], list[list]]:
    '''
    Parses lines from an XFLR5 .txt export, locating the data table
    by its header sentinel ("y-span" and "Chord") and collecting all
    numeric rows until the next blank line or section header.

    Args:
        lines: Raw lines read from the .txt file.

    Returns:
        Tuple (headers, rows) where headers is a list of column name
        strings and rows is a list of lists of floats.
    '''
    HEADERS = [
        "y-span", "Chord", "Ai", "Cl", "PCd", "ICd",
        "CmGeom", "CmAirf@chord/4", "XTrtop", "XTrBot",
        "XCP", "BM",
    ]
    N_COLS = len(HEADERS)

    # Locate the first data row (line after the header sentinel)
    start_idx = None
    for i, line in enumerate(lines):
        if "y-span" in line and "Chord" in line:
            start_idx = i + 1
            break

    if start_idx is None:
        raise ValueError(
            "Main datatable sentinel ('y-span' / 'Chord') not found in file."
        )

    rows: list[list] = []
    for line in lines[start_idx:]:
        # Stop at blank lines or next named section
        if line.strip() == "" or "Main Wing Cp Coefficients" in line:
            break

        parts = re.split(r'\s+', line.strip())
        if len(parts) < N_COLS:
            continue                        # skip malformed rows silently

        rows.append([float(p) for p in parts[:N_COLS]])

    return HEADERS, rows


# =============================================================================
# Public API - logging utilities
# =============================================================================

# Shared disabled logger returned whenever enable_logging=False.
# Created once at import time; avoids per-instance setLevel / _clear_cache calls
# that would otherwise cost ~14 ms per candidate in the DE hot path.
_NULL_LOGGER: logging.Logger = logging.getLogger("cl3o._null")
_NULL_LOGGER.setLevel(logging.CRITICAL + 1)
_NULL_LOGGER.propagate = False


def setup_logger(
        obj: str | type,
        enable_logging: bool
) -> logging.Logger:
    '''
    Sets up logger for the 'obj' class.

    Args:
        enable_logging: Enable (True) / Disable (False) logging.

    Returns:
        Configured logger instance.
    '''
    if not enable_logging:
        return _NULL_LOGGER

    if isinstance(obj, str):                    # string
        name = obj
    elif isinstance(obj, type):                 # class
        name = obj.__name__
    elif hasattr(obj, "__class__"):             # instance
        name = obj.__class__.__name__
    else:
        raise TypeError("obj must be a class, instance, or string")

    logger = logging.getLogger(f"{name}_{hex(id(obj))}")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def log(
        obj: str | type,
        msg: str
) -> None:
    '''Emit *msg* through the instance logger (INFO level).'''
    if obj.logger:
        obj.logger.info(msg)



# =============================================================================
# Public API - .dat files
# =============================================================================

def read_dat_file(filepath: str | Path) -> dict[str, str | np.ndarray]:
    '''
    Reads an airfoil .dat file and returns normalized coordinate data.

    The coordinates are normalized so that the chord equals 1.0,
    regardless of the original file scaling.

    Args:
        filepath: Path to the .dat file.

    Returns:
        Dictionary with keys:
            'name' - airfoil name, if exists fileheader,
            'x' - normalized x-coordinates (0 to 1),
            'y' - corresponding y-coordinates
    '''
    path = _resolve_path(filepath)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    
    x, y, name = _parse_afl_raw_data(lines)
    x_norm, y_norm = _normalize_afl(x, y)

    return {
        "name": name,
        "x": x_norm,
        "y": y_norm
    }



# =============================================================================
# Public API - JSON reader and writer
# =============================================================================

def read_json(
    filepath: str | Path,
    dcls: type
) -> object:
    '''
    Reads a JSON file and returns the corresponding Python dictionary.

    Args:
        filepath: Path (relative or absolute) to the .json file.
        dcls: The dataclass, not an instance.

    Returns:
        Dictionary with the parsed JSON contents.
    '''
    path = _resolve_path(filepath)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    io_logger.info(f"Converting {path} into {dcls.__name__}.")
    return _to_dataclass(data, dcls)


def write_json(
    obj: dict | type,
    filepath: str | Path,
    indent: Optional[int] = 2
) -> None:
    '''
    Serializes a Python dictionary to a JSON file.

    Parent directories are created automatically. NumPy arrays and scalar
    types are converted to native Python types before serialization.

    Args:
        dcls: The dataclass, not instance.
        filepath: Destination file path.
        indent: Number of indentation spaces.
    '''
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Internal converter helper
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError(f"Non-serializable type: {type(obj)}")

    if isinstance(obj, dict):
        data = obj
    else:
        data = _to_dict(obj)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=_convert, ensure_ascii=False)
        
        io_logger.info(f"Database saved in filepath: {path}.")



# =============================================================================
# Public API - read .XLSX
# =============================================================================

def read_xlsx(
    filepath: str | Path,
    sheet_name: Optional[str] = None,
) -> dict:
    '''
    Reads spreadsheet and returns columnar arrays.

    The spreadsheet must have column headers on the first row. Each column
    becomes a dictionary key whose value is a np.ndarray of floats (or
    objects if the column contains non-numeric data).

    Args:
        filepath: Path to the .xlsx file.
        sheet_name: Worksheet name. If None, the active (first) sheet is used.

    Returns:
        Dictionary with [key = column header, value = np.ndarray].
    '''
    import openpyxl  # optional heavy dep; imported here to avoid breaking headless envs
    path = _resolve_path(filepath)

    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet_name] if sheet_name else wb.active

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        raise ValueError(f"Empty worksheet in {path}.")

    headers = [
        str(h).strip() if h is not None else f"col_{i}"
        for i, h in enumerate(rows[0])
    ]

    data: dict = {h: [] for h in headers}
    for row in rows[1:]:
        for h, val in zip(headers, row):
            data[h].append(val)

    result: dict = {}
    for h, vals in data.items():
        try:
            result[h] = np.array(vals, dtype=float)
        except (TypeError, ValueError):
            result[h] = np.array(vals, dtype=object)

    return result


def read_txt(
    filepath: str | Path,
) -> dict:
    '''
    Reads a .txt file exported from XFLR5 and returns columnar arrays.

    The file is expected to contain a spanwise aero table preceded by a
    sentinel header line that includes the strings "y-span" and "Chord".
    Each column becomes a dictionary key whose value is a np.ndarray of
    floats (or objects if conversion fails).

    Args:
        filepath: Path to the XFLR5 .txt export file.

    Returns:
        Dictionary with [key = column header, value = np.ndarray], i.e.
        the same structure returned by :read_xlsx:.
    '''
    path = _resolve_path(filepath)

    with open(path, "r", encoding="latin-1", errors="replace") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Empty file: {path}.")

    headers, rows = _parse_txt_xflr5(lines)

    # Pivot rows → columns, then mirror the read_xlsx conversion logic
    data: dict = {h: [] for h in headers}
    for row in rows:
        for h, val in zip(headers, row):
            data[h].append(val)

    result: dict = {}
    for h, vals in data.items():
        try:
            result[h] = np.array(vals, dtype=float)
        except (TypeError, ValueError):
            result[h] = np.array(vals, dtype=object)

    return result


# ================================================================================
# Module-level usage
# ================================================================================

if __name__ == '__main__':
    pass
