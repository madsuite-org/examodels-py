"""The Python-side handles: blocks, index sets, tables, and what they refuse.

These classes hold no model structure -- the backend does -- but they do hold
the index bookkeeping and the input validation, and every refusal here names
its fix. The tests pin both the accepted shapes and the messages.
"""
import dataclasses

import numpy as np
import pytest

import examodels as exa
from examodels.node import Constraint, Product, _field_names, _table, is_table


@pytest.fixture
def block():
    core = exa.Core()
    return core, core.add_var(4)


# ----------------------------------------------------------------- nodes ------
def test_node_reprs_show_the_backend_type(block):
    core, x = block
    node = x[0] + 1
    assert repr(node).startswith("<Node ")
    assert "Node" in node.julia_type


def test_private_names_never_reach_the_backend(block):
    # A plain name is a field lookup into the index set (the table tests build
    # those); an underscored one must fail here rather than confuse the backend.
    _core, x = block
    with pytest.raises(AttributeError):
        x[0]._not_a_field


# ---------------------------------------------------------------- blocks ------
def test_block_axes_and_shape_agree(block):
    _core, x = block
    assert x.axes == (range(4),)
    assert x.shape == (4,)
    assert len(x) == 4


# --------------------------------------------------------------- product ------
def test_product_takes_only_unit_step_ranges():
    with pytest.raises(TypeError, match="unit-step range"):
        Product()
    with pytest.raises(TypeError, match="unit-step range"):
        Product(range(0, 10, 2))
    with pytest.raises(TypeError, match="unit-step range"):
        Product([0, 1, 2])


def test_product_len_iter_and_repr():
    p = exa.product(range(2), range(3))
    assert len(p) == 6
    assert repr(p) == "<product 2 x 3>"
    cursor = iter(p)
    assert iter(cursor) is cursor          # a generator expression re-iterates it
    with pytest.raises(StopIteration):     # and it never yields: tracing, not looping
        next(cursor)


# ----------------------------------------------------------- constraints ------
def test_a_placeholder_sized_constraint_refuses_len():
    core, n = exa.recipe()
    x = core.add_var(n)
    con = core.add_con(lambda i: x[i], over=exa.srange(0, n))
    with pytest.raises(TypeError, match="model.ncon"):
        len(con)
    assert repr(con) == "<constraint block, sized when the model is built>"


def test_a_sized_constraint_reports_its_rows(block):
    core, x = block
    con = core.add_con(lambda i: x[i] - x[i + 1], over=range(3))
    assert len(con) == 3
    assert repr(con) == "<constraint block of 3>"
    assert repr(Constraint(None, None)) == "<constraint block, sized when the model is built>"


# ----------------------------------------------------------------- tables -----
def test_field_names_come_from_any_reasonable_record():
    @dataclasses.dataclass
    class DC:
        a: int
        b: float

    class Slotted:
        __slots__ = ("a", "b")

    class OneSlot:
        __slots__ = "a"

    class Attrs:
        __attrs_attrs__ = (type("A", (), {"name": "a"}), type("B", (), {"name": "b"}))

    assert _field_names(DC(1, 2.0)) == ["a", "b"]
    assert _field_names(Slotted()) == ["a", "b"]
    assert _field_names(OneSlot()) == ["a"]
    assert _field_names(Attrs()) == ["a", "b"]
    assert _field_names(object()) is None


def test_an_empty_table_is_refused():
    with pytest.raises(ValueError, match="at least one row"):
        _table([])


def test_dict_rows_are_refused_by_name():
    with pytest.raises(TypeError, match="namedtuple"):
        _table([{"a": 1}])


def test_nameless_rows_are_refused_by_type():
    with pytest.raises(TypeError, match="named fields.*got.*int"):
        _table([42])


def test_is_table_rejects_the_obvious_non_tables():
    assert is_table(range(5)) is False
    assert is_table("abc") is False
    assert is_table(np.zeros(3)) is False          # unstructured array
    assert is_table(np.zeros(2, dtype=[("a", "i8")])) is True


# ------------------------------------------------------------- generators -----
def test_a_plain_generator_function_cannot_be_traced(block):
    core, x = block

    def rows():
        yield x[0]

    with pytest.raises(TypeError, match="generator expression"):
        core.add_obj(rows())


def test_an_unrecoverable_index_set_names_the_alternative(block):
    core, x = block

    class Opaque:
        """Iterable whose iterator hides what it iterates."""

        def __iter__(self):
            return self

        def __next__(self):
            return 0

        def __reduce__(self):
            raise TypeError("no")

    with pytest.raises(TypeError, match="pass a function and `over=`"):
        core.add_obj(x[i] ** 2 for i in Opaque())


def test_a_plain_list_is_a_fine_index_set(block):
    core, x = block
    con = core.add_con(lambda i: x[i] - 1.0, over=[0, 2])
    model = core.build()
    assert model.ncon == 2
    assert len(con) == 2
