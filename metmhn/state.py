# SPDX-License-Identifier: MIT-0

from collections.abc import Collection, Hashable, Iterable, Iterator, MutableSet, MutableSequence, Sequence
from functools import singledispatchmethod
from copy import copy


class _State:
    '''Dummy class for overloading State.'''
    pass


class MetState(_State, Hashable, MutableSet):
    '''Subclass of Set representing a state in a MHN.'''

    __slots__ = '__data', '__size', '__PT'  # , '__labels'

    def __init__(self, data: Iterable[int] | int, /, size: int, PT: Sequence[bool]  # , labels: Sequence[str] | None = None
                 ):
        '''Create a new state from an Iterable or an integer.'''
        self.__size = size
        # Only create a new tuple if mutable; otherwise, use as-is.
        # self.__labels = tuple(labels) if isinstance(
        #     labels, MutableSequence) else labels

        if isinstance(data, Iterable):
            self.__data = 0
            for i in data:
                self.add(i)
        elif isinstance(data, int):
            if data < 0:
                raise ValueError('The given integer must be non-negative')
            self.__data = data
        else:
            raise TypeError('unsupported argument type: \'{}\')',
                            type(data).__name__)

    @property
    def data(self) -> int:
        return self.__data

    @property
    def events(self) -> Iterator:
        temp = self.__data
        for i in range(self.size):
            if temp & 1:
                yield i
            temp >>= 1

    # @property
    # def labels(self) -> Sequence[str] | None:
    #     return self.__labels

    @property
    def size(self) -> int:
        return self.__size

    # def copy(self) -> _State:
    #     return copy(self)

    @classmethod
    def from_seq(cls, seq: Collection[bool], /, labels: Sequence[str] | None = None) -> _State:
        '''Create a new state from a collection of booleans.'''
        return cls((i for i, j in enumerate(seq) if j), size=len(seq), labels=labels)

    def to_seq(self) -> tuple[bool]:
        return tuple(i in self for i in range(self.size))

    # # repr()
    # def __repr__(self) -> str:
    #     if self.labels:
    #         return '{}({}, size={}, labels={})'.format(
    #             type(self).__name__,
    #             set(self),
    #             self.size,
    #             self.labels[:self.size],
    #         )
    #     else:
    #         return '{}({}, size={})'.format(
    #             type(self).__name__,
    #             set(self),
    #             self.size,
    #         )

    # # bool()
    # def __bool__(self) -> bool:
    #     return bool(self.data)

    # # int()
    # def __int__(self) -> int:
    #     return self.data

    # # unary ~
    # def __invert__(self) -> _State:
    #     return type(self)((1<<self.size) + ~self.data, size=self.size, labels=self.labels)

    # # MutableSet mixin
    # def _from_iterable(self, iterable: Iterable[int]) -> _State:
    #     return type(self)(iterable, size=self.size, labels=self.labels)

    # MutableSet, in
    @singledispatchmethod
    def __contains__(self, item):
        raise TypeError('unsupported operand type: \'{}\')',
                        type(item).__name__)

    @__contains__.register
    def _(self, item: int) -> bool:
        return self.data & 1 << item

    # @__contains__.register
    # def _(self, item: str) -> bool:
    #     if self.labels is None:
    #         raise TypeError('unsupported argument type: \'{}\' for \'{}\' without labels', type(item).__name__, type(self).__name__)
    #     return self.labels.index(item) in self

    # MutableSet, iter()
    def __iter__(self) -> Iterator[int]:
        for i in range(self.data.bit_length()):
            if self.data >> i & 1:
                yield self.size-1-i

    # MutableSet, len()
    def __len__(self) -> int:
        return self.data.bit_count()

    # MutableSet
    @singledispatchmethod
    def add(self, item) -> None:
        raise TypeError('unsupported argument type: \'{}\')',
                        type(item).__name__)

    @add.register
    def _(self, item: int) -> None:
        self.__data |= 1 << item

    # @add.register
    # def _(self, item: str) -> None:
    #     if self.labels is None:
    #         raise TypeError('unsupported argument type: \'{}\' for \'{}\' without labels', type(item).__name__, type(self).__name__)
    #     self.add(self.labels.index(item))

    # MutableSet
    @singledispatchmethod
    def discard(self, item) -> None:
        raise TypeError('unsupported argument type: \'{}\')',
                        type(item).__name__)

    @discard.register
    def _(self, item: int) -> None:
        self.__data &= ~(1 << self.size-1-item)

    @discard.register
    def _(self, item: str) -> None:
        if self.labels is None:
            raise TypeError('unsupported argument type: \'{}\' for \'{}\' without labels', type(
                item).__name__, type(self).__name__)
        self.discard(self.labels.index(item))

    # # MutableSet mixin, ==
    # @singledispatchmethod
    # def __eq__(self, other) -> bool:
    #     return super().__eq__(other)

    # @__eq__.register
    # def _(self, other) -> bool:
    #     return self.data == other.data

    # # MutableSet mixin, <=
    # @singledispatchmethod
    # def __le__(self, other) -> bool:
    #     return super().__le__(other)

    # @__le__.register
    # def _(self, other: _State) -> bool:
    #     return not self.data & ~other.data

    # # MutableSet mixin, >=
    # @singledispatchmethod
    # def __ge__(self, other) -> bool:
    #     return super().__ge__(other)

    # @__ge__.register
    # def _(self, other: _State) -> bool:
    #     return not other.data & ~self.data

    # # MutableSet mixin, &
    # @singledispatchmethod
    # def __and__(self, other) -> _State:
    #     return super().__and__(other)

    # @__and__.register
    # def _(self, other: _State) -> _State:
    #     return type(self)(self.data & other.data, size=self.size, labels=self.labels)

    # # MutableSet mixin, |
    # @singledispatchmethod
    # def __or__(self, other) -> _State:
    #     return super().__or__(other)

    # @__or__.register
    # def _(self, other: _State) -> _State:
    #     return type(self)(self.data | other.data, size=self.size, labels=self.labels)

    # # MutableSet mixin, ^
    # @singledispatchmethod
    # def __xor__(self, other) -> _State:
    #     return super().__xor__(other)

    # @__xor__.register
    # def _(self, other: _State) -> _State:
    #     return type(self)(self.data ^ other.data, size=self.size, labels=self.labels)

    # # MutableSet mixin
    # def clear(self) -> None:
    #     self.__data = 0

    # Hashable, hash()
    def __hash__(self) -> int:
        return self.data | 1 << self.size
