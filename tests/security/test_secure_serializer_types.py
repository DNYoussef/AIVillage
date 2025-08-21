import pytest

from packages.core.legacy.security.secure_serializer import SecureSerializer


@pytest.mark.parametrize(
    "type_name,value",
    [
        ("str", "hello"),
        ("int", 1),
        ("float", 1.5),
        ("bool", True),
    ],
)
def test_restore_data_allowed_types(type_name, value):
    serializer = SecureSerializer()
    data = {"__type__": type_name, "__value__": value}
    assert serializer._restore_data(data) == value


def test_restore_data_disallowed_type():
    serializer = SecureSerializer()
    data = {"__type__": "complex", "__value__": "1+2j"}
    with pytest.raises(ValueError):
        serializer._restore_data(data)
