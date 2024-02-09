# SPDX-License-Identifier: MIT-0

from collections.abc import Collection, Hashable, Iterable, Iterator, MutableSet, MutableSequence, Sequence
from functools import singledispatchmethod
from copy import copy
from itertools import chain, repeat, cycle


class _State:
    '''Dummy class for overloading State.'''
    pass


class State(_State, Hashable, MutableSet):
    '''Subclass of Set representing a state in a MHN.'''

    __slots__ = '__data', '__size'

    def __init__(
            self,
            data: Iterable[int] | int,
            /,
            size: int,
    ):
        '''Create a new state from an Iterable or an integer.'''
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

    # MutableSet, in
    @singledispatchmethod
    def __contains__(self, item):
        raise TypeError('unsupported operand type: \'{}\')',
                        type(item).__name__)

    @__contains__.register
    def _(self, item: int) -> bool:
        return self.data & 1 << item

    # MutableSet, iter()
    def __iter__(self) -> Iterator[int]:
        for i in range(self.data.bit_length()):
            if self.data >> i & 1:
                yield i

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

    # MutableSet
    @singledispatchmethod
    def discard(self, item) -> None:
        raise TypeError('unsupported argument type: \'{}\')',
                        type(item).__name__)

    @discard.register
    def _(self, item: int) -> None:
        self.__data &= ~(1 << item)

    # Hashable, hash()
    def __hash__(self) -> int:
        return self.data

    @classmethod
    def from_seq(cls, seq: Collection[bool], /) -> _State:
        '''Create a new state from a collection of booleans.'''
        return cls((i for i, j in enumerate(seq) if j), size=len(seq))


class RestrState(_State, Hashable, MutableSet):
    '''Subclass of Set representing a state in a MHN.'''

    __slots__ = '__data', '__size', '__restrict'

    def __init__(
            self,
            data: Iterable[int] | int,
            /,
            restrict: State,
    ):

        self.__restrict = restrict
        '''Create a new state from an Iterable or an integer.'''
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
    def events(self) -> Iterator[int]:
        for i, e in enumerate(self.__restrict):
            if self.__data >> i & 1:
                yield e

    # MutableSet, in

    @singledispatchmethod
    def __contains__(self, item):
        raise TypeError('unsupported operand type: \'{}\')',
                        type(item).__name__)

    @__contains__.register
    def _(self, item: int) -> bool:
        return self.data & 1 << item

    # MutableSet, iter()
    def __iter__(self) -> Iterator[int]:
        for i in range(self.data.bit_length()):
            if self.data >> i & 1:
                yield i

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

    # MutableSet
    @singledispatchmethod
    def discard(self, item) -> None:
        raise TypeError('unsupported argument type: \'{}\')',
                        type(item).__name__)

    @discard.register
    def _(self, item: int) -> None:
        self.__data &= ~(1 << item)

    # Hashable, hash()
    def __hash__(self) -> int:
        return self.data

    # @classmethod
    # def from_seq(cls, seq: Collection[bool], /) -> _State:
    #     '''Create a new state from a collection of booleans.'''
    #     return cls((i for i, j in enumerate(seq) if j), size=len(seq))


class MetState(_State, Hashable, MutableSet):
    '''Subclass of Set representing a state in a MHN.'''

    __slots__ = '__data', '__size', '__PT', '__events'

    def __init__(
            self,
            data: Iterable[int] | int,
            /,
            size: int,
    ):
        '''Create a new state from an Iterable or an integer.'''
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

        self.__PT = list()
        self.__events = list()
        for i, pt, event in zip(range(size), cycle([True, False]), chain.from_iterable([repeat(i, 2) for i in range(size)])):
            if (self.data >> i) & 1:
                if i == size - 1:
                    self.__PT.append(False)
                else:
                    self.__PT.append(pt)
                self.__events.append(event)
        self.__PT = tuple(self.__PT)
        self.__events = tuple(self.__events)

    @property
    def data(self) -> int:
        return self.__data

    @property
    def PT(self) -> State:
        return self.__PT

    # MutableSet, in
    @singledispatchmethod
    def __contains__(self, item):
        raise TypeError('unsupported operand type: \'{}\')',
                        type(item).__name__)

    @__contains__.register
    def _(self, item: int) -> bool:
        return self.data & 1 << item

    # MutableSet, iter()
    def __iter__(self) -> Iterator[int]:
        for i in range(self.data.bit_length()):
            if self.data >> i & 1:
                yield i

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

    # MutableSet
    @singledispatchmethod
    def discard(self, item) -> None:
        raise TypeError('unsupported argument type: \'{}\')',
                        type(item).__name__)

    @discard.register
    def _(self, item: int) -> None:
        self.__data &= ~(1 << item)

    # Hashable, hash()
    def __hash__(self) -> int:
        return self.data | 1 << self.size

    @classmethod
    def from_seq(cls, seq: Collection[bool], /, labels: Sequence[str] | None = None) -> _State:
        '''Create a new state from a collection of booleans.'''
        return cls((i for i, j in enumerate(seq) if j), size=len(seq))


class RestrMetState(_State, Hashable, MutableSet):
    '''Subclass of Set representing a state in a MHN.'''

    __slots__ = '__data', '__size'

    def __init__(
        self,
        data: Iterable[int] | int,
        /,
        restrict: MetState,
        # , labels: Sequence[str] | None = None
    ):
        self.__restrict = restrict

        '''Create a new state from an Iterable or an integer.'''
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

    # @property
    # def events(self) -> Iterator:
    #     temp = self.__data
    #     for i in range(self.size):
    #         if temp & 1:
    #             yield i
    #         temp >>= 1

    # @property
    # def labels(self) -> Sequence[str] | None:
    #     return self.__labels

    # @property
    # def size(self) -> int:
    #     return self.__size

    @property
    def PT(self) -> Iterator[int]:
        for i, pt in self.__restrict.PT:
            if pt:
                yield i
        return self.__PT

    @property
    def MT(self) -> int:
        return self.__PT

    @property
    def events(self) -> int:
        return self.__events

    # def copy(self) -> _State:
    #     return copy(self)

    # @classmethod
    # def from_seq(cls, seq: Collection[bool], /, labels: Sequence[str] | None = None) -> _State:
    #     '''Create a new state from a collection of booleans.'''
    #     return cls((i for i, j in enumerate(seq) if j), size=len(seq), labels=labels)

    # def to_seq(self) -> tuple[bool]:
    #     return tuple(i in self for i in range(self.size))

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

    # MutableSet, iter()
    def __iter__(self) -> Iterator[int]:
        for i in range(self.data.bit_length()):
            if self.data >> i & 1:
                yield i

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

    # MutableSet
    @singledispatchmethod
    def discard(self, item) -> None:
        raise TypeError('unsupported argument type: \'{}\')',
                        type(item).__name__)

    @discard.register
    def _(self, item: int) -> None:
        self.__data &= ~(1 << item)

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


if __name__ == "__main__":
    pt = [True, True, False, False]
    events = [0, 1, 1, 2]
    s = MetState.from_seq([True, True, False, True, True])
    print(
        s.data,
        s.PT,
        s.events
    )
