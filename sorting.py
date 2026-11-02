def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break

def selectionSort(array, size):
    for ind in range(size - 1):
        min_index = ind
        for j in range(ind + 1, size):
            if array[j] < array[min_index]:
                min_index = j
        array[ind], array[min_index] = array[min_index], array[ind]


def counting_sort(arr):
   # Find the maximum value in the array
   max_val = max(arr)
   size = len(arr)
   # Initialize count and output arrays
   count = [0] * (max_val + 1)
   output = [0] * size
   # Count occurrences of each element
   for num in arr:
       count[num] += 1
   # Compute cumulative counts
   for i in range(1, len(count)):
       count[i] += count[i - 1]
   # Build the output array
   for i in range(size - 1, -1, -1):
       output[count[arr[i]] - 1] = arr[i]
       count[arr[i]] -= 1
   # Copy sorted elements back to original array
   for i in range(size):
       arr[i] = output[i]

def radixSort(arr):
    max_element = max(arr) # Find the largest number
    place = 1

# Apply counting sort for each digit place
    while max_element // place > 0:
        countingSort(arr, place)
        place *= 10

