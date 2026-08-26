"""Tron base58check ↔ 0x-hex address converter.

Tron transaction graphs in this toolkit store addresses as `0x<40-hex>` with
the 0x41 prefix byte stripped. Public Tron sources (OFAC SDN, Tronscan,
walletexplorer) publish addresses in base58check T-format. This module is
the one-way bridge between address spaces.

Reference (Tron docs): mainnet prefix byte is 0x41, followed by a 20-byte
address (same scheme as EVM), 4-byte double-SHA256 checksum, then
base58-encoded to 34-char string starting with 'T'.

Round-trip example (USDT TRC20 contract):
    TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t
    → 0xa614f803b6fd780986a42c78ec9c7f77e6ded13c
"""

from __future__ import annotations

import hashlib

_B58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_ALPHABET_SET = set(_B58_ALPHABET)
_TRON_MAINNET_PREFIX = 0x41


class InvalidTronAddress(ValueError):  # noqa: N818 — public API; renaming would break callers
    """Raised when a string is not a valid Tron base58check mainnet address."""


def _b58decode(s: str) -> bytes:
    if not s:
        raise InvalidTronAddress("empty string")
    b = s.encode("ascii", errors="strict")
    for c in b:
        if c not in _B58_ALPHABET_SET:
            raise InvalidTronAddress(f"non-base58 character: {chr(c)!r}")
    n = 0
    for c in b:
        n = n * 58 + _B58_ALPHABET.index(c)
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    n_pad = len(s) - len(s.lstrip("1"))
    return b"\x00" * n_pad + raw


def _b58encode(payload: bytes) -> str:
    n_pad = len(payload) - len(payload.lstrip(b"\x00"))
    n = int.from_bytes(payload, "big")
    out = bytearray()
    while n > 0:
        n, r = divmod(n, 58)
        out.append(_B58_ALPHABET[r])
    out.extend(b"1" * n_pad)
    return out[::-1].decode("ascii")


def base58check_to_hex(t_addr: str) -> str:
    """Convert Tron base58check T-address to lowercase 0x-prefixed 40-hex.

    Drops the 0x41 Tron mainnet prefix byte — output is the 20-byte EVM-
    style address the Tron graph uses.

    Raises:
        InvalidTronAddress: wrong length, bad checksum, or non-0x41 prefix.
    """
    if not isinstance(t_addr, str):
        raise InvalidTronAddress(f"expected str, got {type(t_addr).__name__}")
    if len(t_addr) != 34:
        raise InvalidTronAddress(f"wrong length {len(t_addr)}, expected 34")
    raw = _b58decode(t_addr)
    if len(raw) != 25:
        raise InvalidTronAddress(
            f"decoded length {len(raw)}, expected 25 (21 payload + 4 checksum)"
        )
    payload, checksum = raw[:21], raw[21:]
    expected = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    if checksum != expected:
        raise InvalidTronAddress("checksum mismatch")
    if payload[0] != _TRON_MAINNET_PREFIX:
        raise InvalidTronAddress(
            f"prefix byte {payload[0]:#x}, expected {_TRON_MAINNET_PREFIX:#x} (mainnet)"
        )
    return "0x" + payload[1:].hex()


def hex_to_base58check(hex_addr: str) -> str:
    """Convert 0x-prefixed 40-hex Tron address back to base58check T-format.

    Inverse of `base58check_to_hex`. Used for cross-checking and for
    emitting human-readable forms in logs / reports.

    Raises:
        InvalidTronAddress: wrong length or non-hex characters.
    """
    if not isinstance(hex_addr, str):
        raise InvalidTronAddress(f"expected str, got {type(hex_addr).__name__}")
    if not hex_addr.startswith("0x"):
        raise InvalidTronAddress("missing 0x prefix")
    body = hex_addr[2:]
    if len(body) != 40:
        raise InvalidTronAddress(f"hex body length {len(body)}, expected 40")
    try:
        addr_bytes = bytes.fromhex(body)
    except ValueError as exc:
        raise InvalidTronAddress(f"non-hex: {exc}") from None
    payload = bytes([_TRON_MAINNET_PREFIX]) + addr_bytes
    checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    return _b58encode(payload + checksum)
