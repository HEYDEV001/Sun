"""Country-agnostic text normalisation for business names and addresses.

Design goals
  * No country-specific hard-coding of *matching logic* (France is unseen in train).
    Abbreviation tables are generic street/legal words for US/IN/FR; everything
    else (which tokens are "generic") is learned from data via document frequency.
  * Fold accents / non-Latin scripts to ASCII with `unidecode` so that
    "Société", "société" and "SOCIETE" collapse together.
  * Strip address junk that only the noisy sources add (PO BOX, PMB, N/A).
"""
import re
from unidecode import unidecode

# ---- generic abbreviation expansion (applied token-wise after lowercasing) ----
ADDR_ABBR = {
    # English street types
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "pl": "place", "hwy": "highway", "pkwy": "parkway", "cir": "circle", "ter": "terrace",
    "trl": "trail", "sq": "square", "mt": "mount", "ft": "fort", "apt": "apartment",
    "bldg": "building", "flr": "floor", "ste": "suite", "no": "number", "nr": "near",
    "opp": "opposite", "cross": "cross", "nagar": "nagar", "n": "north", "s": "south",
    "e": "east", "w": "west", "ne": "northeast", "nw": "northwest", "se": "southeast",
    "sw": "southwest",
    # French street types
    "r": "rue", "bd": "boulevard", "bld": "boulevard", "chem": "chemin", "che": "chemin",
    "imp": "impasse", "all": "allee", "pce": "place", "fg": "faubourg", "fbg": "faubourg",
    "crs": "cours", "qu": "quai", "rte": "route", "sq.": "square", "av.": "avenue",
}
NAME_ABBR = {
    "pvt": "private", "ltd": "limited", "corp": "corporation", "co": "company",
    "inc": "incorporated", "intl": "international", "svc": "service", "svcs": "services",
    "mfg": "manufacturing", "assoc": "associates", "bros": "brothers", "dept": "department",
    "univ": "university", "ctr": "center", "centre": "center", "grp": "group",
}
# tokens that are pure noise in addresses of the noisy sources
_ADDR_JUNK = re.compile(
    r"\b(p\.?\s?o\.?\s?box\s*\w*|pmb\s*\w+|n/?a|null|none|nan)\b", re.I)
_DOMAIN = re.compile(r"^(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|net|org|in|co\.in|fr|co|biz|info|io)$")
_NONWORD = re.compile(r"[^a-z0-9]+")
_BRACKET = re.compile(r"[\[\(][^\]\)]*[\]\)]")   # "[INC]", "(PC)" tags

def _fold(s: str) -> str:
    if not s:
        return ""
    if not s.isascii():
        s = unidecode(s)
    return s.lower()

_VOWELS = re.compile(r"[aeiouy]")
_DOUBLE = re.compile(r"(.)\1+")

def skeleton(squashed: str) -> str:
    """Vowel-free, de-duplicated consonant skeleton. Cross-script robust:
    'al ttek praaivett limittedd' and 'al tech private limited' both shrink
    to similar consonant strings, which lets transliterated names partly match."""
    return _DOUBLE.sub(r"\1", _VOWELS.sub("", squashed))

def norm_name(s: str):
    """Return (tokens_str, squashed, is_domain, nonlatin_flag)."""
    raw = s if isinstance(s, str) else ""   # guard NaN/None from parquet nulls
    nonlatin = 0 if raw.isascii() else 1
    t = _fold(raw).replace("&", " and ")
    t = _BRACKET.sub(" ", t)
    dom = 0
    m = _DOMAIN.match(t.strip())
    if m:
        dom = 1
        t = m.group(1).replace("-", " ")
    t = _NONWORD.sub(" ", t)
    toks = [NAME_ABBR.get(w, w) for w in t.split()]
    out = " ".join(toks)
    return out, out.replace(" ", ""), dom, nonlatin

def norm_addr(s: str):
    """Return (tokens_str, numeric_tokens_str)."""
    if not isinstance(s, str) or not s:   # guard NaN/None from parquet nulls
        return "", ""
    t = _fold(s)
    t = _ADDR_JUNK.sub(" ", t)
    t = _NONWORD.sub(" ", t)
    toks = [ADDR_ABBR.get(w, w) for w in t.split()]
    out = " ".join(toks)
    nums = " ".join(w for w in toks if any(c.isdigit() for c in w))
    return out, nums
