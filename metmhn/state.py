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

        self.__size = size
        self.__n = size // 2

    @property
    def data(self) -> int:
        return self.__data

    @property
    def size(self) -> int:
        return self.__size

    @property
    def n(self) -> int:
        return self.__n

    @property
    def PT(self) -> Iterator[int]:
        _data = self.data
        for i in range(self.n):
            if _data & 1:
                yield i
            _data >>= 2

    @property
    def MT(self) -> Iterator[int]:
        _data = self.data >> 1
        for i in range(self.n):
            if _data & 1:
                yield i
            _data >>= 2

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

    __slots__ = '__data', '__restrict'

    def __init__(
        self,
        data: Iterable[int] | int,
        /,
        restrict: MetState,
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
        self.__restrict = restrict

    @property
    def data(self) -> int:
        return self.__data

    @property
    def restrict(self) -> int:
        return self.__restrict

    @property
    def PT(self) -> Iterator[int]:
        _restrict = self.__restrict.data
        _data = self.data
        for i in range(self.__restrict.n):
            if _restrict & 1:
                _data >>= 1
                if _data & 1:
                    yield i
            _restrict >>= 1
            if _restrict & 1:
                _data >>= 1
            _restrict >>= 1

    @property
    def MT(self) -> Iterator[int]:
        _restrict = self.__restrict.data
        _data = self.data
        for i in range(self.__restrict.n):
            if _restrict & 1:
                _data >>= 1
            _restrict >>= 1
            if _restrict & 1:
                if _data & 1:
                    yield i
                _data >>= 1
            _restrict >>= 1

    @property
    def events(self) -> Iterator[int]:
        _restrict = self.__restrict.data
        _data = self.data
        for i in range(self.__restrict.n):
            if _restrict & 1:
                if _data & 1:
                    yield i
                _data >>= 1
            _restrict >>= 1
            if _restrict & 1:
                if _data & 1:
                    yield i
                _data >>= 1
            _restrict >>= 1
        _restrict >>= 1
        if _restrict & 1 and _data & 1:
            yield self.__restrict.n

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


if __name__ == "__main__":
    pt = [True, True, False, False]
    events = [0, 1, 1, 2]
    s = MetState.from_seq([True, True, False, True, True])
    print(
        s.data,
        s.PT,
        s.events
    )
