"""Chain-aware address case policy.

Lowercasing is canonical ONLY where the address alphabet is case-insensitive:
  - EVM ``0x`` + 40 hex (EIP-55 checksum case is display-only);
  - bech32/bech32m (``bc1``/``tb1``/``ltc1`` — single-case by spec, canonical
    form is lowercase).
Everywhere else (base58check and friends: BTC legacy ``1…``/``3…``, LTC
``L…``/``M…``, DOGE ``D…``, DASH ``X…``, XRP ``r…``, TRON ``T…``, SOL, XMR
``4…``/``8…``, ZEC ``t1…``/``t3…``) case IS payload: lowercasing destroys the
address irreversibly.

A blanket ``.lower()`` in a label-ingest path therefore silently destroys
every base58check-family address it touches. Every ingest path MUST
normalize through :func:`canonical_case` and every label-set gate MUST
count :func:`is_case_destroyed`.

Caveat on detection: a genuine base58 string CAN be naturally all-lowercase
(no uppercase letters drawn), at a small but nonzero rate.
``is_case_destroyed`` is therefore a high-precision heuristic for
COUNTING/flagging, not a proof for any single address.
"""

from __future__ import annotations

import re

_EVM_RE = re.compile(r"0x[0-9a-fA-F]{40}")
_BECH32_PREFIXES = ("bc1", "tb1", "ltc1")

# Chains whose canonical address alphabet is case-SENSITIVE. Chain tags as they
# appear across label sources (short and long forms).
CASE_SENSITIVE_CHAINS = frozenset(
    {
        "btc",
        "bitcoin",
        "ltc",
        "litecoin",
        "doge",
        "dogecoin",
        "dash",
        "bch",
        "bitcoincash",
        "xrp",
        "ripple",
        "sol",
        "solana",
        "xmr",
        "monero",
        "zec",
        "zcash",
        "tron",
        "trx",
    }
)


def _is_bech32_form(a: str) -> bool:
    return a.lower().startswith(_BECH32_PREFIXES)


def _single_case(a: str) -> bool:
    return not (any(c.isupper() for c in a) and any(c.islower() for c in a))


def canonical_case(addr: str) -> str:
    """Return the canonical-case form of *addr* without destroying payload.

    EVM-form → lowercase; single-case bech32-form → lowercase; everything else
    (base58check etc., plus mixed-case bech32 which is invalid by spec and kept
    intact for forensics) → returned byte-for-byte with only whitespace
    stripped.
    """
    a = addr.strip()
    if _EVM_RE.fullmatch(a):
        return a.lower()
    if _is_bech32_form(a) and _single_case(a):
        return a.lower()
    return a


def is_case_sensitive_form(addr: str, chain: str | None = None) -> bool:
    """True when *addr* sits on a case-SENSITIVE (base58check-family) alphabet:
    not EVM-form, not bech32-form, and — when *chain* is given — on a chain in
    :data:`CASE_SENSITIVE_CHAINS`.

    These are exactly the addresses whose payload a blanket ``.lower()`` would
    destroy, so this is the denominator for an artefact case-integrity audit:
    :func:`is_case_destroyed` (all-lowercase-with-letters) is the numerator.
    """
    a = addr.strip()
    if not a or _EVM_RE.fullmatch(a) or _is_bech32_form(a):
        return False
    if chain is not None and chain.strip().lower() not in CASE_SENSITIVE_CHAINS:
        return False
    return True


def is_case_destroyed(addr: str, chain: str | None = None) -> bool:
    """True when *addr* bears the destroyed-case signature: an address on a
    case-sensitive alphabet that contains letters and none of them uppercase.

    When *chain* is given, only chains in :data:`CASE_SENSITIVE_CHAINS` are
    judged (an ``eth``/``base``-tagged row can never be case-destroyed). When
    *chain* is None the judgement is by address form alone.
    """
    a = addr.strip()
    if _EVM_RE.fullmatch(a) or _is_bech32_form(a):
        return False
    if chain is not None and chain.strip().lower() not in CASE_SENSITIVE_CHAINS:
        return False
    return a.islower() and any(c.isalpha() for c in a)
