"""Vendored LazyProperty descriptor — avoids external lib dependency."""
import typing

_T = typing.TypeVar('_T')
_R = typing.TypeVar('_R')


class LazyProperty(typing.Generic[_T, _R]):
    """Lazy-computed cached property with non-None assertion."""

    def __init__(self, func: typing.Callable[[_T], _R]) -> None:
        self.func: typing.Callable[[_T], _R] = func
        self.attr_name: str = f"_{func.__name__}"

    def __get__(self, instance: typing.Optional[_T], owner: typing.Optional[type] = None) -> _R:
        if instance is None:
            return self  # type: ignore[return-value]

        attr_value = getattr(instance, self.attr_name)
        if attr_value is None:
            attr_value = self.func(instance)
            setattr(instance, self.attr_name, attr_value)
        assert attr_value is not None, f"`{self.attr_name}` has not been initialized despite the LazyProperty."
        return attr_value
