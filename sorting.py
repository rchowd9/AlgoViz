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
# Example usage
array = [38, 27, 43, 3, 9, 82, 10]
sorted_array = merge_sort(array)
print("Original Array:", array)
print("Sorted Array:", sorted_array)

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
# Usage
data = [36, 34, 43, 11, 15, 20, 28, 45]
print("Before:", data)
shell_sort(data)
print("After: ", data)
