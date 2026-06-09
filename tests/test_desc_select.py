"""Unit tests for descriptor-column selection (`_desc_select`).

The helper itself is pure-Python (only ``hashlib``); the test runs in
milliseconds once imports resolve. Covers the include/exclude/explicit
precedence and the cache-tag stability that key the Phase-2 descriptor-type
study (documents/projects/gwu.md).

Run: pytest -q tests/test_desc_select.py
"""
import pytest

from graphgps.loader.dataset._desc_select import (
    select_descriptor_columns,
    selection_tag,
)

COLS = ["mw", "logp_gwu", "tpsa", "homo_gwu", "nrings", "lumo_gwu"]


def test_all_when_no_spec():
    assert select_descriptor_columns(COLS) == COLS


def test_include_gwu():
    assert select_descriptor_columns(COLS, include=["_gwu"]) == [
        "logp_gwu", "homo_gwu", "lumo_gwu"]


def test_exclude_gwu():
    assert select_descriptor_columns(COLS, exclude=["_gwu"]) == [
        "mw", "tpsa", "nrings"]


def test_explicit_columns_kept_in_original_order():
    # request out of order -> returned in all_columns order
    assert select_descriptor_columns(COLS, columns=["tpsa", "mw"]) == [
        "mw", "tpsa"]


def test_columns_take_precedence_then_exclude_applied():
    out = select_descriptor_columns(
        COLS, include=["_gwu"], columns=["mw", "homo_gwu"], exclude=["homo"])
    assert out == ["mw"]


def test_unknown_explicit_column_raises():
    with pytest.raises(ValueError):
        select_descriptor_columns(COLS, columns=["does_not_exist"])


def test_empty_selection_raises():
    with pytest.raises(ValueError):
        select_descriptor_columns(COLS, include=["zzz_no_match"])


def test_tag_stable_and_order_sensitive():
    a = selection_tag(["mw", "tpsa"])
    assert a == selection_tag(["mw", "tpsa"])      # stable
    assert a != selection_tag(["tpsa", "mw"])      # order-sensitive
    assert len(a) == 8
