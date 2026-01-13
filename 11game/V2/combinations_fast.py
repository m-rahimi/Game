import os
import ctypes

# Resolve library path relative to this file or c++ directory
_here = os.path.dirname(os.path.abspath(__file__))
_cpp_dir = os.path.join(os.path.dirname(os.path.dirname(_here)), 'c++')
_candidates = [
    os.path.join(_cpp_dir, 'libcombinations.dylib'),
    os.path.join(_here, 'libcombinations.dylib'),
]
_lib_path = next((p for p in _candidates if os.path.exists(p)), None)

if _lib_path is None:
    raise FileNotFoundError('libcombinations.dylib not found. Build it in game/11/c++')

lib = ctypes.CDLL(_lib_path)

class Card(ctypes.Structure):
    _fields_ = [
        ("rank", ctypes.c_char * 10),
        ("suit", ctypes.c_char * 10),
        ("score", ctypes.c_int),
        ("value", ctypes.c_int),
        ("prob", ctypes.c_double),
    ]

class CombosResult(ctypes.Structure):
    _fields_ = [
        ("indices", ctypes.POINTER(ctypes.c_int)),
        ("counts", ctypes.POINTER(ctypes.c_int)),
        ("num_combos", ctypes.c_int),
    ]

# Function signatures
lib.find_combinations_c.argtypes = [
    ctypes.POINTER(Card), ctypes.c_int, ctypes.POINTER(Card), ctypes.c_int
]
lib.find_combinations_c.restype = CombosResult

lib.free_combinations_arrays.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
lib.free_combinations_arrays.restype = None


def _to_card_struct(py_card):
    # py_card as tuple: (rank, suit, score, value, prob)
    rank, suit, score, value, prob = py_card
    # ensure bytes for C char[] fields
    rank_b = rank.encode('utf-8') if isinstance(rank, str) else rank
    suit_b = suit.encode('utf-8') if isinstance(suit, str) else suit
    c = Card()
    c.rank[:len(rank_b)] = rank_b
    c.suit[:len(suit_b)] = suit_b
    c.score = int(score)
    c.value = int(value)
    c.prob = float(prob)
    return c


def find_combinations_fast(floor, card, target=11):
    # Build ctypes arrays
    floor_cards = [_to_card_struct(c) for c in floor]
    floor_array = (Card * len(floor_cards))(*floor_cards)
    card_struct = _to_card_struct(card)

    res = lib.find_combinations_c(floor_array, len(floor_cards), ctypes.byref(card_struct), int(target))

    # Extract results
    num = res.num_combos
    if num == 0:
        # Nothing to free (indices/counts may be nullptr)
        return []

    counts = [res.counts[i] for i in range(num)]
    total = sum(counts)
    indices = [res.indices[i] for i in range(total)]

    # Free C-allocated memory
    lib.free_combinations_arrays(res.indices, res.counts)

    # Rebuild Python combos as lists of card tuples + the played card
    combos = []
    cursor = 0
    for ccount in counts:
        idxs = indices[cursor:cursor+ccount]
        combo = [floor[i] for i in idxs] + [card]
        combos.append(combo)
        cursor += ccount

    return combos
