# SPDX-License-Identifier: MIT-0

from collections.abc import Collection, Hashable, Iterable, Iterator, MutableSet, MutableSequence, Sequence
from typing import Tuple
from functools import singledispatchmethod
from copy import copy
from itertools import chain, repeat, cycle
import numpy as np


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
                self.add(int(i))
        elif isinstance(data, int):
            if data < 0:
                raise ValueError('The given integer must be non-negative')
            self.__data = data
        else:
            raise TypeError('unsupported argument type: \'{}\')',
                            type(data).__name__)
        self.__size = size

    @property
    def data(self) -> int:
        return self.__data

    @property
    def size(self) -> int:
        return self.__size

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

    def to_seq(self) -> np.array:
        seq = np.zeros(self.size, dtype=bool)
        seq[list(self)] = 1
        return seq


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
    def restrict(self) -> int:
        return self.__restrict

    @property
    def events(self) -> tuple[int]:
        result = ()
        _restrict = self.restrict.data
        _data = self.data
        for i in range(self.__restrict.size):
            if _restrict & 1:
                if _data & 1:
                    result += (i,)
                _data >>= 1
            _restrict >>= 1
        return result

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

    __slots__ = '__data', '__size', '__n'

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
                self.add(int(i))
        elif isinstance(data, (int, np.int32)):
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
    def events(self) -> tuple[int]:
        result = ()
        for i in range(self.size):
            if self.data >> i & 1:
                result += (i,)
        return result

    @property
    def PT_events(self) -> Tuple[int]:
        return tuple(i for i in range(self.n) if (self.data >> 2*i) & 1)

    @property
    def MT_events(self) -> Tuple[int]:
        return tuple(i for i in range(self.n) if (self.data >> (2*i + 1)) & 1)

    @property
    def Seeding(self) -> Tuple[int]:
        if self.data >> (self.size - 1) & 1:
            return (self.n,)
        else:
            return ()

    @property
    def PT(self) -> State:
        return State(
            (i for i in range(self.n) if (self.data >> 2 * i) & 1),
            size=self.n)

    @property
    def PT_S(self) -> State:
        state = State(
            (i for i in range(self.n) if (self.data >> 2 * i) & 1),
            size=self.n + 1)
        if self.Seeding:
            state.add(self.n)
        return state

    @property
    def MT(self) -> State:
        return State(
            tuple(i for i in range(self.n) if (self.data >> (2 * i + 1))
                  & 1) + (self.n,) if self.Seeding else (),
            size=self.n + 1)

    @property
    def reachable(self) -> bool:
        if self.size - 1 in self:
            return True
        return self.PT_events == self.MT_events

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

    def to_seq(self) -> np.array:
        seq = np.zeros(self.size, dtype=bool)
        seq[list(self)] = 1
        return seq


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
    def PT_events(self) -> Tuple[int]:
        _restrict = self.restrict.data
        _data = self.data
        result = ()
        for i in range(self.restrict.n):
            if _restrict & 1:
                if _data & 1:
                    result += (i,)
                _data >>= 1
            _restrict >>= 1
            if _restrict & 1:
                _data >>= 1
            _restrict >>= 1
        return result

    @property
    def PT_S_events(self) -> Tuple[int]:
        raise NotImplementedError

    @property
    def MT_events(self) -> Tuple[int]:
        _restrict = self.restrict.data
        _data = self.data
        result = ()
        for i in range(self.restrict.n):
            if _restrict & 1:
                _data >>= 1
            _restrict >>= 1
            if _restrict & 1:
                if _data & 1:
                    result += (i,)
                _data >>= 1
            _restrict >>= 1
        return result

    @property
    def Seeding(self) -> tuple[int]:
        if self.data >> (len(self.restrict) - 1) & 1:
            return (self.restrict.n,)
        else:
            return ()

    @property
    def PT(self) -> RestrState:
        return RestrState(
            sum(1 << i for i, e in enumerate(
                self.restrict.PT_events) if e in self.PT_events),
            restrict=State(self.restrict.PT_events, size=self.restrict.n),
        )

    @property
    def PT_S(self) -> RestrState:
        return RestrState(
            sum(1 << i for i, e in enumerate(self.restrict.PT_events +
                self.restrict.Seeding) if e in self.PT_events + self.Seeding),
            restrict=State(self.restrict.PT_events +
                           self.restrict.Seeding, size=self.restrict.n + 1),
        )

    @property
    def MT(self) -> RestrState:
        return RestrState(
            sum(1 << i for i, e in enumerate(self.restrict.MT_events +
                self.restrict.Seeding) if e in self.MT_events + self.Seeding),
            restrict=State(self.restrict.MT_events +
                           self.restrict.Seeding, size=self.restrict.n + 1),
        )

    @property
    def reachable(self) -> bool:
        # whether seeding has happened
        if self.restrict.size - 1 in self.restrict and \
                len(self.restrict) - 1 in self:
            return True
        return self.PT_events == self.MT_events

    @property
    def events(self) -> Tuple[int]:
        _restrict = self.__restrict.data
        _data = self.data
        result = ()
        for i in range(self.__restrict.n):
            if _restrict & 1:
                if _data & 1:
                    result += (i,)
                _data >>= 1
            _restrict >>= 1
            if _restrict & 1:
                if _data & 1:
                    result += (i,)
                _data >>= 1
            _restrict >>= 1
        if _restrict & 1 and _data & 1:
            result += (self.__restrict.n,)
        return result

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

        # MutableSet mixin, ^
    @singledispatchmethod
    def __xor__(self, other) -> _State:
        return super().__xor__(other)

    @__xor__.register
    def _(self, other: _State) -> _State:
        return type(self)(self.data ^ other.data, restrict=self.restrict)
