import cython
import time
import math

print("Hello World")

result = []


def make_calculation(numbers):
    for number in numbers:
        result.append(math.sqrt(number ** 5))


#   10,477881669998169  sec without cython
#   7,725867033004761   sec with cython



number_list = list(range(10000000))
start = time.time()

make_calculation(number_list)

end = time.time()
print(end - start)
