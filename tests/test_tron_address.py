"""Tests for openheads.tron_address — both directions of the converter.

`base58check_to_hex` was previously covered only indirectly (through
test_build_label_set); `hex_to_base58check` had zero references anywhere in
the tree, so an encoder defect (checksum, zero-byte padding, alphabet) would
have shipped unverified.

Every address here is either a public contract address used as a documented
reference vector, or a synthetic encoding of fixed payload bytes — no
sanction programme or attribution is attached to anything.
"""

from __future__ import annotations

import pytest

from openheads.tron_address import (
    InvalidTronAddress,
    base58check_to_hex,
    hex_to_base58check,
)

# Documented reference vector (module docstring): the public USDT TRC20
# contract address, published by Tron itself.
USDT_T = "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"
USDT_HEX = "0xa614f803b6fd780986a42c78ec9c7f77e6ded13c"

# Synthetic: the all-zero payload. Its T-form starts with T9 and exercises
# the leading-zero-byte handling of both encode and decode.
ZERO_T = "T9yD14Nj9j7xAB4dbGeiX9h8unkKHxuWwb"
ZERO_HEX = "0x" + "00" * 20


class TestHexToBase58check:
    def test_reference_vector(self) -> None:
        assert hex_to_base58check(USDT_HEX) == USDT_T

    def test_zero_payload_vector(self) -> None:
        assert hex_to_base58check(ZERO_HEX) == ZERO_T

    def test_round_trip_both_directions(self) -> None:
        for t_addr, hex_addr in ((USDT_T, USDT_HEX), (ZERO_T, ZERO_HEX)):
            assert base58check_to_hex(hex_to_base58check(hex_addr)) == hex_addr
            assert hex_to_base58check(base58check_to_hex(t_addr)) == t_addr

    def test_output_shape(self) -> None:
        out = hex_to_base58check(USDT_HEX)
        assert len(out) == 34
        assert out.startswith("T")

    def test_rejects_missing_prefix(self) -> None:
        with pytest.raises(InvalidTronAddress):
            hex_to_base58check(USDT_HEX[2:])

    def test_rejects_wrong_length(self) -> None:
        with pytest.raises(InvalidTronAddress):
            hex_to_base58check(USDT_HEX + "00")

    def test_rejects_non_hex(self) -> None:
        with pytest.raises(InvalidTronAddress):
            hex_to_base58check("0x" + "zz" * 20)

    def test_rejects_non_str(self) -> None:
        with pytest.raises(InvalidTronAddress):
            hex_to_base58check(0xA614F803)  # type: ignore[arg-type]


class TestBase58checkToHex:
    def test_reference_vector(self) -> None:
        assert base58check_to_hex(USDT_T) == USDT_HEX

    def test_rejects_corrupted_checksum(self) -> None:
        bad = USDT_T[:-1] + ("a" if USDT_T[-1] != "a" else "b")
        with pytest.raises(InvalidTronAddress):
            base58check_to_hex(bad)

    def test_rejects_non_base58_character(self) -> None:
        with pytest.raises(InvalidTronAddress):
            base58check_to_hex("T" + "0" * 33)  # '0' is not in the alphabet
