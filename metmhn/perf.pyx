cdef int my_factorial(int n):
    cdef int result = 1
    cdef int i
    for i in range(1, n + 1):
        result *= i
    return result


cdef int[20] factorial = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200, 1307674368000, 20922789888000, 355687428096000, 6402373705728000, 121645100408832000]

cdef int order_to_int(int[:] order):
    cdef int my_int = 0
    cdef int n, index_o

    if len(order) == 0:
        return 0

    cdef list indices = list(order)
    indices.sort()
    n = len(order)
    
    cdef int o
    for o in order:
        index_o = indices.index(o)
        my_int = my_int * n + index_o
        indices.pop(index_o)
        n -= 1

    return my_int

cdef int[:] int_to_order(int my_int, int[:] numbers):
    cdef int f
    numbers.sort()
    cdef list order = list()

    for i in range(len(numbers) - 1, -1, -1):
        f = factorial[i]
        order.append(numbers.pop(my_int // f))
        my_int %= f

    return tuple(order)

cdef int append_to_int_order(int my_int, int[:] numbers, int new_event):
    cdef int i, f, j, e, new_int
    numbers = numbers.copy()
    numbers.sort()
    new_int = 0

    for i in range(len(numbers) - 1, -1, -1):
        f = factorial[i]
        j = my_int // f
        e = numbers.pop(j)

        if e > new_event:
            j += 1

        new_int += (j * f * (i + 1))
        my_int %= f

    return new_int
