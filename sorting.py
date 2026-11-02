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
        counting_sort(arr, place)
        place *= 10

def merge_sort(arr):
   # Base case: A list with 1 or no elements is already sorted
   if len(arr) <= 1:
       return arr
   # Divide the array into two halves
   mid = len(arr) // 2
   left_half = arr[:mid]
   right_half = arr[mid:]
   # Recursively sort both halves
   left_half = merge_sort(left_half)
   right_half = merge_sort(right_half)
   # Merge the sorted halves
   return merge(left_half, right_half)
def merge(left, right):
   result = [] # List to store the merged result
   i = j = 0 # Pointers for left and right halves
   # Compare elements from both halves and add the smaller one to the result
   while i < len(left) and j < len(right):
       if left[i] <= right[j]:
           result.append(left[i])
           i += 1
       else:
           result.append(right[j])
           j += 1
   # Add any remaining elements from the left half
   while i < len(left):
       result.append(left[i])
       i += 1
   # Add any remaining elements from the right half
   while j < len(right):
       result.append(right[j])
       j += 1
   return result


def shell_sort(arr):
   n = len(arr)
   gap = n // 2 # Initial gap size
   while gap > 0:
       for i in range(gap, n):
           temp = arr[i]
           j = i
           # Perform a gapped insertion sort
           while j >= gap and arr[j - gap] > temp:
               arr[j] = arr[j - gap]
               j -= gap
           arr[j] = temp
       gap //= 2 # Reduce the gap


def heapify(arr, n, i):
   # Find the largest among root, left child, and right child
   largest = i
   left = 2 * i + 1
   right = 2 * i + 2
   if left < n and arr[left] > arr[largest]:
       largest = left
   if right < n and arr[right] > arr[largest]:
       largest = right
   # Swap and continue heapifying if root is not the largest
   if largest != i:
       arr[i], arr[largest] = arr[largest], arr[i]
       heapify(arr, n, largest)
def heap_sort(arr):
   n = len(arr)
   # Build a max heap
   for i in range(n // 2 - 1, -1, -1):
       heapify(arr, n, i)
   # Extract elements from the heap one by one
   for i in range(n - 1, 0, -1):
       arr[i], arr[0] = arr[0], arr[i] # Swap
       heapify(arr, i, 0)


def pigeonhole_sort(arr):
   min_val = min(arr)
   max_val = max(arr)
   size = max_val - min_val + 1
   # Create pigeonholes
   holes = [[] for _ in range(size)]
   # Place elements into pigeonholes
   for num in arr:
       holes[num - min_val].append(num)
   # Collect elements back into array
   index = 0
   for hole in holes:
       for num in hole:
           arr[index] = num
           index += 1
   return arr

def bucket_sort(arr):
   n = len(arr)
   buckets = [[] for _ in range(n)]
   # Scatter: distribute elements into buckets
   for num in arr:
       index = int(n * num)
       buckets[index].append(num)
   # Sort each bucket
   for i in range(n):
       buckets[i].sort()
   # Gather: concatenate buckets
   sorted_arr = []
   for bucket in buckets:
       sorted_arr.extend(bucket)
   return sorted_arr

def insertion_sort(arr):
   for i in range(1, len(arr)):
       key = arr[i] # Current element to insert
       j = i - 1
       # Shift elements greater than key to one position ahead
       while j >= 0 and key < arr[j]:
           arr[j + 1] = arr[j]
           j -= 1
       arr[j + 1] = key # Place key at correct position
