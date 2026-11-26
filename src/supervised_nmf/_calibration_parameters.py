"""Check candidate parameter displays against their saved calibration identities."""

import hashlib
import json
import math
import re
import struct

import numpy as np


def _path_type(node, names):
    module, _, cls = node["type"].rpartition(".")
    return module in ("pathlib", "pathlib._local") and cls in names


def _unicode_format(dtype):
    digits = dtype[2:]
    if len(digits) > 19:
        raise ValueError("Invalid fixed-width Unicode dtype.")
    return int(digits), "utf-32-le" if dtype[0] == "<" else "utf-32-be"


def _unicode_storage(value, width, encoding):
    if type(value) is not str or len(value) > width:
        raise ValueError("NumPy text parameter exceeds its declared width or has the wrong type.")
    padding = width - len(value)
    # JSON can join a UTF-32 surrogate pair into one astral character. That
    # alternative needs a spare code unit. Cap implicit padding per element
    # so that a short archive cannot request unbounded hashing or allocation.
    if padding > 256 or padding and re.search(r"[\U00010000-\U0010ffff]", value):
        return None
    return value.encode(encoding, errors="surrogatepass") + bytes(4 * padding)


def _legacy_unicode_key(dtype, display, cache, content):
    # Match UTF-32 bytes to the actual JSON names. Never allocate storage
    # using an archive's declared width. A padded historical scalar or an
    # escaped surrogate pair can lose information in its public spelling.
    width, endian = _unicode_format(dtype)
    if endian not in cache:
        names = {}
        for name in display:
            raw = name.encode(endian, errors="surrogatepass")
            names[(str(len(name)), hashlib.sha256(raw).hexdigest())] = name
            if not name:
                names[("0", hashlib.sha256(bytes(4)).hexdigest())] = name
        cache[endian] = names, min(map(len, display))
    names, minimum = cache[endian]
    name = names.get((dtype[2:], content))
    if name is not None and not name.endswith("\0"):
        return name
    if name is None and minimum >= width:
        raise ValueError("NumPy text dictionary key disagrees with its display.")
    # Ordinary exact-width names took a single pass over the mapping. Limit
    # extra passes for legacy mixtures of different padded widths.
    used = cache.setdefault(("padded_widths", endian), {})
    if width not in used:
        if len(used) >= 4:
            return None
        matches, ambiguous = {}, False
        for candidate in display:
            if len(candidate) > width:
                continue
            raw = _unicode_storage(candidate, width, endian)
            if raw is None:
                ambiguous = True
            else:
                digest = hashlib.sha256(raw).hexdigest()
                matches[digest] = None if digest in matches else candidate
        used[width] = matches, ambiguous
    matches, ambiguous = used[width]
    if ambiguous:
        return None
    if content not in matches:
        raise ValueError("NumPy text dictionary key disagrees with its display.")
    return matches[content]


def _string_items(node):
    """Read a structural dictionary whose keys have a lossless string display."""
    if node["kind"] != "dict" or type(node["items"]) is not list:
        raise ValueError("Invalid structural parameter dictionary.")
    items = {}
    for key, value in node["items"]:
        if key["kind"] != "str":
            return None
        name = key["value"]
        if type(name) is not str or name in items:
            raise ValueError("Invalid structural parameter name.")
        items[name] = value
    return items


def _numpy_bool(node):
    # NumPy treats any nonzero storage byte as true. Most scalars have the
    # canonical representation, but an array view can contain other bytes.
    digest = node["content"]
    for byte in range(256):
        if hashlib.sha256(bytes([byte])).hexdigest() == digest:
            return bool(byte)
    raise ValueError("Invalid NumPy boolean parameter digest.")


def _finite_float_key_spellings(value, dtype):
    scalar = np.dtype(dtype).type(value)
    # NumPy has changed the cutoffs between positional and scientific
    # notation. Both use a shortest round trip for the original dtype, which
    # matters for subnormals whose decimal can be far from the exact value.
    forms = {np.format_float_positional(scalar, trim="0")}
    if value and (abs(value) < 0.001 or abs(value) >= 1000):
        forms.add(np.format_float_scientific(scalar, trim="-", exp_digits=2))
    # Legacy scalar printing can round to fewer significant digits, e.g.
    # 12 for a double under legacy='1.13'. Preserve the actual emitted form
    # of the decimal, including signed zero and a signed, padded exponent.
    minimum_digits = {"f2": 3, "f4": 6, "f8": 12}[dtype[1:]]
    for digits in range(minimum_digits, 18):
        spelling = format(value, f".{digits}g")
        if "." not in spelling and "e" not in spelling:
            spelling += ".0"
        forms.add(spelling)
    return forms


def _display_key(key, display, cache, class_labels):
    """Recover the name produced by the writer's str(key), where possible."""
    kind = key["kind"]
    if kind == "str" or kind == "path" and _path_type(key, ("WindowsPath", "PosixPath", "PureWindowsPath", "PurePosixPath")):
        name = key["value"]
        if type(name) is not str:
            raise ValueError("Invalid structural dictionary key.")
        return name
    if kind == "none":
        return "None"
    if kind == "bool":
        if type(key["value"]) is not bool:
            raise ValueError("Invalid structural boolean dictionary key.")
        return str(key["value"])
    if kind == "int":
        text = key["value"]
        if re.fullmatch(r"0|-?[1-9][0-9]*", text):
            return text
        if re.fullmatch(r"-?0x[1-9a-f][0-9a-f]*", text):
            return str(int(text, 16))
        raise ValueError("Invalid structural integer dictionary key.")
    if kind == "float":
        bits = bytes.fromhex(key["bits"])
        if len(bits) != 8:
            raise ValueError("Invalid structural floating-point dictionary key.")
        return str(struct.unpack("!d", bits)[0])
    if kind == "numpy_scalar":
        dtype = key["dtype"]["str"]
        if "key_text" in key:
            if type(key["key_text"]) is not str:
                raise ValueError("Invalid NumPy dictionary key spelling.")
            return key["key_text"]
        if re.fullmatch(r"[<>]U[0-9]+", dtype):
            return _legacy_unicode_key(dtype, display, cache, key["content"])
        if dtype == "|b1":
            return str(_numpy_bool(key))
        if dtype[1:] in ("f2", "f4", "f8") and key["dtype"].get("format", [None])[0] != "g":
            if dtype not in cache:
                names = {}
                packer = struct.Struct((">" if dtype[0] == ">" else "<") + {"f2": "e", "f4": "f", "f8": "d"}[dtype[1:]])
                for name in display:
                    try:
                        number = float(name)
                        raw = packer.pack(number)
                    except (ValueError, OverflowError, struct.error):
                        continue
                    if math.isnan(number):
                        continue  # String conversion discards NaN payloads.
                    digest = hashlib.sha256(raw).hexdigest()
                    # Several names may round to the same scalar.
                    names[digest] = None if digest in names else name
                # A class label can independently identify the original
                # scalar by its bytes even if the genuine key was renamed.
                identified = {}
                numeric_labels = [k for k in class_labels if type(k) in (bool, int, float)]
                if 0 in numeric_labels:
                    # Both signs of zero are equal dictionary keys for this class.
                    numeric_labels.extend([0.0, -0.0])
                for label in numeric_labels:
                    try:
                        raw = packer.pack(label)
                    except (OverflowError, struct.error):
                        continue
                    original = packer.unpack(raw)[0]
                    if math.isfinite(original):
                        label_digest = hashlib.sha256(raw).hexdigest()
                        identified[label_digest] = _finite_float_key_spellings(original, dtype).intersection(display)
                cache[dtype] = names, identified
            names, identified = cache[dtype]
            possible = identified.get(key["content"])
            if possible is None:
                return names.get(key["content"])
            if not possible:
                raise ValueError("NumPy floating-point dictionary key disagrees with its class label and display.")
            return next(iter(possible)) if len(possible) == 1 else None
        if re.fullmatch(r"[<>|][iu][1248]", dtype):
            # A legacy integer key carries its byte digest. Match the decimal
            # names actually present, once per integer dtype in this mapping.
            if dtype not in cache:
                names = {}
                for name in display:
                    if re.fullmatch(r"0|-?[1-9][0-9]{0,19}", name):
                        try:
                            raw = int(name).to_bytes(int(dtype[2:]), "big" if dtype[0] == ">" else "little",
                                                     signed=dtype[1] == "i")
                        except OverflowError:
                            continue
                        names[hashlib.sha256(raw).hexdigest()] = name
                cache[dtype] = names
            if key["content"] not in cache[dtype]:
                raise ValueError("NumPy integer dictionary key disagrees with its display.")
            return cache[dtype][key["content"]]
    return None


def _check_dict_display(node, display, class_labels, float_finiteness):
    if (type(display) is not dict or type(node["items"]) is not list
            or bool(display) != bool(node["items"]) or len(display) > len(node["items"])):
        raise ValueError("Invalid candidate parameter dictionary.")
    cache = {}
    declarations = [(_display_key(key, display, cache, class_labels), value) for key, value in node["items"]]
    known_names = {name for name, _ in declarations if name is not None}
    max_names = len(known_names) + sum(name is None for name, _ in declarations)
    if len(display) > max_names or not known_names.issubset(display):
        raise ValueError("Candidate parameter names disagree with the declared grid.")
    bindings = {}
    for name, value in declarations:
        if name is None:
            # An opaque key could overwrite a preceding value only if at
            # least two declared names collapsed to one displayed name.
            if len(display) < max_names:
                bindings.clear()
        else:
            bindings[name] = value
    for name, value in bindings.items():
        _check_display(value, display[name], class_labels, float_finiteness)


def _exact_numpy_text(key, name):
    if key["kind"] != "numpy_scalar" or not re.fullmatch(r"[<>]U[0-9]+", key["dtype"]["str"]):
        return False
    width, encoding = _unicode_format(key["dtype"]["str"])
    raw = _unicode_storage(name, width, encoding)
    return raw is not None and hashlib.sha256(raw).hexdigest() == key["content"]


def _refinement_key(key, display, cache):
    """Recover an original parameter name eligible for a raw string lookup."""
    if key["kind"] == "str":
        name = key["value"]
    elif key["kind"] == "numpy_scalar" and re.fullmatch(r"[<>]U[0-9]+", key["dtype"]["str"]):
        name = _display_key(key, display, cache, ())
    else:
        return None
    # Only retained UTF-32 bytes can resolve the original spelling when JSON
    # may have combined a pair of surrogate code points.
    if name is not None and re.search(r"[\U00010000-\U0010ffff]", name) and not _exact_numpy_text(key, name):
        return None
    return name


def _extended_display(display, binary64):
    """Validate a stored dyadic and return its correctly rounded binary64 coordinate."""
    if (type(display) is not dict or set(display) != {"__supervised_nmf_scalar__", "significand", "exponent"}
            or display["__supervised_nmf_scalar__"] != "longdouble-v1"
            or type(display["significand"]) is not str
            or not re.fullmatch(r"-?(?:0|[1-9][0-9]{0,632})", display["significand"])
            or type(display["exponent"]) is not int):
        raise ValueError("Invalid extended-float parameter display.")
    text, exponent = display["significand"], display["exponent"]
    significand = int(text)
    if not significand:
        if exponent != 0:
            raise ValueError("Invalid zero exponent in extended-float parameter.")
        return float(text)
    bits = significand.bit_length()
    if not significand % 2:
        raise ValueError("Extended-float significand must be odd.")
    if binary64:
        if bits > 53 or exponent < -1074 or exponent + bits > 1024:
            raise ValueError("Extended-float parameter is not exactly representable in binary64.")
        return math.ldexp(float(significand), exponent)
    # Allow the range of IEEE quad/80-bit extended, or a sum of two double
    # values. The latter can have a gap between its significant components.
    if not ((bits <= 113 and -16494 <= exponent and exponent + bits <= 16384)
            or (-1074 <= exponent and exponent + bits <= 1025)):
        raise ValueError("Extended-float parameter is outside supported storage formats.")
    # Converting the significand to float before scaling can round twice
    # at the subnormal boundary. Parse the bounded dyadic in one rounding.
    try:
        return float.fromhex(f"{significand:#x}p{exponent}")
    except OverflowError:
        return -math.inf if significand < 0 else math.inf


def _check_numpy_display(node, display, float_finiteness):
    # NumPy numeric grids often contain scalar objects. Ordinary integers and
    # finite IEEE floats can be checked against their original storage bytes.
    dtype = node["dtype"]["str"]
    if "finite" in node and node["finite"] is not (display is not None):
        raise ValueError("NumPy floating-point display disagrees with its recorded finiteness.")
    if "display" in node and json.dumps(node["display"], sort_keys=True) != json.dumps(display, sort_keys=True):
        raise ValueError("NumPy parameter display disagrees with its request identity.")
    if re.fullmatch(r"[<>]U[0-9]+", dtype):
        width, encoding = _unicode_format(dtype)
        raw = _unicode_storage(display, width, encoding)
        if raw is not None and hashlib.sha256(raw).hexdigest() != node["content"]:
            # An empty scalar can retain one zero code unit with dtype U0.
            if width or hashlib.sha256(bytes(4)).hexdigest() != node["content"]:
                raise ValueError("NumPy text parameter disagrees with its byte digest.")
        return
    if dtype == "|b1":
        if display is not _numpy_bool(node):
            raise ValueError("NumPy boolean disagrees with the declared grid.")
        return
    format_code = node["dtype"].get("format", [None])[0]
    if dtype[1:2] == "c":
        coordinate = _complex_coordinate(node, display)
        if coordinate is not NotImplemented and float_finiteness is not None:
            # Matching scalar bytes on the writer's ABI give the same repr
            # or the same pair of tags, even when NaN payloads/padding are lost.
            key = (dtype, node["content"], "tagged" if type(display) is dict else "repr")
            witness = json.dumps(display, sort_keys=True)
            if float_finiteness.setdefault(key, witness) != witness:
                raise ValueError("Candidate displays disagree on the same complex parameter.")
        return
    wide = re.fullmatch(r"[<>]f(?:10|12|16)", dtype) is not None
    if display is not None and dtype[1:2] == "f" and (wide or format_code == "g" or type(display) is dict):
        if format_code not in (None, "g") or dtype[2:] not in ("8", "10", "12", "16"):
            raise ValueError("Invalid extended-float parameter dtype.")
        decoded = _extended_display(display, binary64=dtype[2:] == "8")
        if not wide:
            display = decoded
    if wide:
        # Identical bytes within one archive came from the same ABI. Padding
        # can hide a match but cannot change the value of an exact match.
        witness = None if display is None else (display["significand"], display["exponent"])
        key = (dtype, node["content"])
        if float_finiteness is not None and float_finiteness.setdefault(key, witness) != witness:
            raise ValueError("Candidate displays disagree on the same extended floating-point parameter.")
        return
    if not re.fullmatch(r"[<>|](?:[iu][1248]|f[248])", dtype):
        return
    floating = dtype[1] == "f"
    if floating and float_finiteness is not None:
        # A finite IEEE display must reproduce the saved bytes below. A null
        # alone cannot identify those bytes, but it cannot share them with a
        # verified finite value elsewhere in this archive, in either order.
        key = (dtype, node["content"])
        finite = display is not None
        if float_finiteness.setdefault(key, finite) != finite:
            raise ValueError("Candidate displays disagree on the finiteness of the same NumPy parameter.")
    if floating and display is None:
        # The writer discards the bit pattern of a nonfinite value.
        return
    if type(display) is not (float if floating else int):
        raise ValueError("NumPy parameter has the wrong display type.")
    dtype = np.dtype(dtype)
    if not floating:
        bounds = np.iinfo(dtype)
        if not bounds.min <= display <= bounds.max:
            raise ValueError("NumPy integer parameter is outside its dtype range.")
    with np.errstate(over="ignore", invalid="ignore"):
        scalar = np.array(display, dtype=dtype)
    if scalar.item() != display or hashlib.sha256(scalar.tobytes()).hexdigest() != node["content"]:
        raise ValueError("NumPy parameter value disagrees with the declared grid.")


def _complex_coordinate(node, display):
    """Recover float(scalar), or None for an unresolved nonfinite real part."""
    dtype = node["dtype"]["str"]
    if not re.fullmatch(r"[<>]c(?:8|16|20|24|32)", dtype):
        return NotImplemented
    format_code = node["dtype"].get("format", [None])[0]
    if type(display) is dict or format_code == "G" or dtype[2:] not in ("8", "16"):
        if (dtype[2:] == "8" or format_code not in (None, "G") or type(display) is not dict
                or set(display) != {"__supervised_nmf_scalar__", "real", "imag"}
                or display["__supervised_nmf_scalar__"] != "clongdouble-v1"):
            raise ValueError("Invalid extended-complex grid scalar display.")
        binary64 = dtype[2:] == "16"
        real, imag = [None if part is None else _extended_display(part, binary64)
                      for part in (display["real"], display["imag"])]
        if binary64:
            pack = struct.Struct(dtype[0] + "dd").pack
            if real is not None and imag is not None:
                if hashlib.sha256(pack(real, imag)).hexdigest() != node["content"]:
                    raise ValueError("Complex grid scalar disagrees with its declared bytes.")
            elif real is None:
                # A null may be a signed infinity or any NaN payload. Only
                # compare reconstructable encodings of the complete scalar.
                for sign in (-math.inf, math.inf):
                    for other in ((imag,) if imag is not None else (-math.inf, math.inf)):
                        if hashlib.sha256(pack(sign, other)).hexdigest() == node["content"]:
                            return sign
                if imag is not None:
                    return math.nan
        return real
    # An ordinary scalar's Python complex repr spells both components,
    # including signed infinities. It omits the payload of each NaN.
    if format_code not in (None, "F" if dtype[2:] == "8" else "D") or type(display) is not str or len(display) > 128:
        raise ValueError("Invalid complex grid scalar display.")
    value = complex(display)
    if repr(value) != display:
        raise ValueError("Invalid complex grid scalar display.")
    component = struct.Struct(dtype[0] + ("f" if dtype[2:] == "8" else "d"))
    recovered = []
    for part in (value.real, value.imag):
        if not math.isnan(part):
            try:
                raw = component.pack(part)
            except OverflowError as error:
                raise ValueError("Complex component exceeds its declared dtype.") from error
            if component.unpack(raw)[0] != part:
                raise ValueError("Complex component disagrees with its declared dtype.")
            recovered.append(raw)
    if len(recovered) == 2 and hashlib.sha256(b"".join(recovered)).hexdigest() != node["content"]:
        raise ValueError("Complex grid scalar disagrees with its declared bytes.")
    return value.real


def _matches_infinity(node, value):
    """Check a signed infinity against its ordinary IEEE byte encoding."""
    if node["kind"] == "float":
        return node["bits"] == struct.pack(">d", value).hex()
    if node["kind"] == "numpy_scalar":
        dtype = node["dtype"]["str"]
        if re.fullmatch(r"[<>]f[248]", dtype):
            code = {"2": "e", "4": "f", "8": "d"}[dtype[2:]]
            return node["content"] == hashlib.sha256(struct.pack(dtype[0] + code, value)).hexdigest()
    return None  # A historical wide-float digest can include platform-specific padding.


def _array_values(display, shape, axis=0):
    if axis == len(shape):
        yield display
    else:
        if type(display) is not list or len(display) != shape[axis]:
            raise ValueError("NumPy array parameter shape disagrees with its display.")
        for item in display:
            yield from _array_values(item, shape, axis + 1)


def _check_array(node, display, class_labels, float_finiteness):
    dtype, shape = node["dtype"]["str"], node["shape"]
    if type(shape) is not list or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("Invalid structural array shape.")
    values = _array_values(display, shape)
    if re.fullmatch(r"[<>]U[0-9]+", dtype):
        width, encoding = _unicode_format(dtype)
        digest = hashlib.sha256()
        reversible = True
        for value in values:
            raw = _unicode_storage(value, width, encoding)
            if raw is None:
                reversible = False
            else:
                digest.update(raw)
        if reversible and digest.hexdigest() != node["content"]:
            raise ValueError("NumPy text array parameter disagrees with its byte digest.")
        return
    if dtype in ("|O", "StringDType", "|T8", "|T16"):
        if type(node["content"]) is not list:
            raise ValueError("Missing array parameter element declarations.")
        sentinel = node["dtype"].get("na_object") if dtype == "StringDType" else None
        for element, value in zip(node["content"], values, strict=True):
            # StringDType encodes a missing text sentinel as None in its
            # identity. Its public tolist() returns the sentinel itself.
            if sentinel and sentinel["kind"] == "str" and element["kind"] == "none":
                element = sentinel
            _check_display(element, value, class_labels, float_finiteness)
        return
    # Only decode fixed-size, ordinary numeric dtypes. The stream length is
    # determined by the existing JSON display, never an untrusted dimension.
    codes = {"i1": "b", "u1": "B", "i2": "h", "u2": "H", "i4": "i", "u4": "I",
             "i8": "q", "u8": "Q", "f2": "e", "f4": "f", "f8": "d", "b1": "?"}
    code = codes.get(dtype[1:])
    if not code or dtype[:1] not in ("|", "<", ">"):
        # Even if the leaf representation is opaque, tolist preserves the
        # dimensions up to and including the first empty axis.
        for _ in values:
            pass
        return
    packer = struct.Struct((">" if dtype[0] == ">" else "<") + code)
    digest = hashlib.sha256()
    lossy = False
    for value in values:
        if dtype[1] == "f":
            if value is None:
                lossy = True  # IEEE NaN payloads are absent from tolist()/JSON.
                digest.update(packer.pack(math.nan))
                continue
            if type(value) is dict and dtype[1:] == "f8" and node["dtype"].get("format", [None])[0] in ("g", None):
                value = _extended_display(value, binary64=True)
            valid_type = type(value) is float
        elif dtype[1] == "b":
            valid_type = type(value) is bool
            lossy |= value is True  # Any nonzero storage byte can mean true.
        else:
            valid_type = type(value) is int
        if not valid_type:
            raise ValueError("Invalid NumPy array parameter element type.")
        try:
            raw = packer.pack(value)
        except (struct.error, OverflowError) as error:
            raise ValueError("NumPy array parameter element is out of range.") from error
        if packer.unpack(raw)[0] != value:
            raise ValueError("NumPy array parameter element differs after conversion to its dtype.")
        digest.update(raw)
    actual = digest.hexdigest()
    if ((not lossy and actual != node["content"])
            or ("display_content" in node and actual != node["display_content"])):
        raise ValueError("NumPy array parameter disagrees with its byte digest.")


def _check_display(node, display, class_labels=(), float_finiteness=None):
    """Bind reversible display values without constructing historical objects."""
    kind = node["kind"]
    if kind == "dict":
        _check_dict_display(node, display, class_labels, float_finiteness)
        return
    if kind in ("list", "tuple"):
        items = node["items"]
        if type(items) is not list or type(display) is not list or len(items) != len(display):
            raise ValueError("Candidate parameter sequence disagrees with the declared grid.")
        for value, item in zip(items, display):
            _check_display(value, item, class_labels, float_finiteness)
        return
    if kind == "numpy_scalar":
        _check_numpy_display(node, display, float_finiteness)
        return
    if kind == "ndarray":
        _check_array(node, display, class_labels, float_finiteness)
        return
    if kind == "none":
        matches = display is None
    elif kind in ("bool", "str"):
        expected_type = bool if kind == "bool" else str
        matches = type(node["value"]) is expected_type and type(display) is expected_type and display == node["value"]
    elif kind == "path" and _path_type(node, ("PosixPath", "WindowsPath")):
        matches = type(display) is str and display == node["value"]
    elif kind == "int":
        # Early schema-2 writers used decimal for all Python integers.
        matches = type(display) is int and (node["value"] == str(display)
                                            or display.bit_length() > 2048 and node["value"] == hex(display))
    elif kind == "float":
        bits = bytes.fromhex(node["bits"])
        if len(bits) != 8:
            raise ValueError("Invalid structural floating-point parameter.")
        value = struct.unpack("!d", bits)[0]
        matches = (type(display) is float and struct.pack("!d", display) == bits
                   if math.isfinite(value) else display is None)
    else:
        # In particular, an estimator's repr and a callback's display name do
        # not uniquely represent their structural identity.
        return
    if not matches:
        raise ValueError("Candidate parameter value disagrees with the declared grid.")
