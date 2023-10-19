import numpy as np

def chamfer_distance(array1, array2):
    """
    Calculate Chamfer Distance between two point clouds
    :param array1: NxD numpy array, where N is the number of points and D is the dimension
    :param array2: MxD numpy array, where M is the number of points and D is the dimension
    :return: Chamfer Distance
    """
    distance = 0
    
    for point1 in array1:
        distance += np.min(np.sum((array2 - point1) ** 2, axis=1))
        
    for point2 in array2:
        distance += np.min(np.sum((array1 - point2) ** 2, axis=1))
        
    distance = distance / (len(array1) + len(array2))
    
    return distance

# Example usage
array1 = np.array([[1, 1], [2, 2], [3, 3]])  # 3x2 array
array2 = np.array([[1, 1.5], [2, 2.5], [3, 3.5]])  # 3x2 array

cd = chamfer_distance(array1, array2)
print(f"Chamfer Distance: {cd}")