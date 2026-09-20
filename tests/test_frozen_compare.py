"""
Tests tying frozen instances, equality, hashing and container behavior to a
single set of rules.
"""

import pytest

import attr

from attr import diff, field, frozen, mutable
from attr.exceptions import (
    FrozenInstanceError,
    NotAnAttrsClassError,
    NotInitializedError,
)


@frozen
class Point:
    x: int
    y: int


@frozen
class Skip:
    a: int
    b: int = field(default=0, eq=False)


@frozen
class Triple:
    x: int
    y: int
    z: int


class TestFrozenWrites:
    def test_set_names_field_and_keeps_value(self):
        p = Point(1, 2)

        with pytest.raises(FrozenInstanceError) as exc_info:
            p.x = 9

        assert exc_info.value.name == "x"
        assert "x" in str(exc_info.value)
        assert p.x == 1
        assert p.y == 2

    def test_del_names_field_and_keeps_value(self):
        p = Point(1, 2)

        with pytest.raises(FrozenInstanceError) as exc_info:
            del p.x

        assert exc_info.value.name == "x"
        assert p.x == 1

    def test_failed_mutation_keeps_set_member_usable(self):
        p = Point(1, 2)
        members = {p}

        with pytest.raises(FrozenInstanceError):
            p.x = 9

        assert Point(1, 2) in members


class TestMutable:
    def test_change_reflected_in_equality(self):
        @mutable
        class M:
            v: int

        m = M(1)
        m.v = 2

        assert m == M(2)

    def test_unhashable_rejected_as_mapping_key(self):
        @mutable
        class M:
            v: int

        with pytest.raises(TypeError):
            hash(M(1))

        with pytest.raises(TypeError):
            {M(1): "value"}


class TestValueEquality:
    def test_equality_by_value_not_identity(self):
        assert Point(1, 2) == Point(1, 2)
        assert Point(1, 2) is not Point(1, 2)
        assert Point(1, 2) != Point(2, 1)

    def test_identity_mode_preserved_when_requested(self):
        @attr.s(eq=False, auto_attribs=True)
        class OldStyle:
            v: int

        a = OldStyle(1)
        b = OldStyle(1)

        assert a != b
        assert a == a

    def test_skipped_field_cannot_change_verdict(self):
        assert Skip(1, 2) == Skip(1, 99)
        assert Skip(1, 2) != Skip(2, 2)

    def test_different_field_count_is_unequal(self):
        assert Point(1, 2) != Triple(1, 2, 3)
        assert Point(1, 2).__eq__(Triple(1, 2, 3)) is NotImplemented

    def test_half_built_instance_is_not_compared(self):
        half = object.__new__(Point)
        full = Point(1, 2)

        with pytest.raises(NotInitializedError):
            half == full

        with pytest.raises(NotInitializedError):
            full == half

        with pytest.raises(NotInitializedError):
            half == object.__new__(Point)


class TestHashing:
    def test_equal_objects_have_equal_hashes(self):
        assert hash(Point(1, 2)) == hash(Point(1, 2))

    def test_skipped_field_does_not_change_hash(self):
        assert hash(Skip(1, 2)) == hash(Skip(1, 99))

    def test_half_built_instance_is_not_hashed(self):
        half = object.__new__(Point)

        with pytest.raises(Exception):
            hash(half)

    def test_hash_collision_still_holds_both_values(self):
        @frozen
        class Collision:
            v: int

            def __hash__(self):
                return 1234

        a = Collision(1)
        b = Collision(2)

        assert hash(a) == hash(b)
        assert a != b
        assert len({a, b}) == 2


class TestContainers:
    def test_equal_values_occupy_one_slot(self):
        assert len({Point(1, 2), Point(1, 2)}) == 1
        mapping = {Point(1, 2): "here"}
        assert mapping[Point(1, 2)] == "here"

    def test_set_returns_inserted_object(self):
        p = Point(7, 8)
        (retrieved,) = {p}

        assert retrieved is p
        assert retrieved.x == 7
        assert retrieved.y == 8


class TestDiff:
    def test_empty_when_equal_ignoring_skipped_fields(self):
        assert diff(Skip(1, 2), Skip(1, 99)) == []

    def test_reasons_omit_skipped_fields(self):
        differences = diff(Skip(1, 2), Skip(7, 99))

        assert differences == [("a", 1, 7)]
        assert all(name != "b" for name, _, _ in differences)

    def test_reports_all_compared_differences(self):
        differences = diff(Point(1, 2), Point(3, 4))

        assert differences == [("x", 1, 3), ("y", 2, 4)]

    def test_rejects_non_attrs_objects(self):
        with pytest.raises(NotAnAttrsClassError):
            diff(Point(1, 2), object())

    def test_rejects_different_classes(self):
        with pytest.raises(TypeError):
            diff(Point(1, 2), Triple(1, 2, 3))
