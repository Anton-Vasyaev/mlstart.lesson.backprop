import struct

def byte_array_to_int32(array_of_bytes, byte_order='<'):
    return int(struct.unpack(f'{byte_order}i', array_of_bytes)[0])

def byte_array_to_float(array_of_bytes, byte_order='<'):
    return float(struct.unpack(f'{byte_order}f', array_of_bytes)[0])


def float_to_byte_array(f_value):
    return struct.pack('<f', f_value)