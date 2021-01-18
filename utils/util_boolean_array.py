import numpy as np

def deserialize_boolean_array(serialized_array,shape) :
    """
    Inverse of serialize_boolean_array.
    """
    num_elements = np.prod(shape)
    packed_bits = np.frombuffer(serialized_array, dtype='uint8')
    result = np.unpackbits(packed_bits)[:num_elements]
    return result

def serialize_boolean_array(array) :
    """
    Takes a numpy.array with boolean values and converts it to a space-efficient
    binary representation.
    """
    array = array.astype(np.bool)
    array = np.packbits(array).astype(np.uint8).tolist()
    return array