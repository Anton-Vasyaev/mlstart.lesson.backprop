# 3rdparty
import numpy as np
import numpy as np
import cv2 as cv


def segment_func(x, max_value, segments):
    segment_range = max_value / segments

    mod_val = x % segment_range
    
    return x - mod_val 
vsegment_func = np.vectorize(segment_func)


def reverse_mat(mat):
    max_val = mat.max()
    mat = -mat
    mat += max_val
    return mat


def visualise_func(map_array, graident_way, options):
    img_size = options['img_size']
    min_r = options['min_r']
    max_r = options['max_r']
    segments = options['segments']
    range_val = max_r - min_r

    min_value = map_array.min()
    result_mat = map_array - min_value
    max_value = result_mat.max()
    result_mat /= max_value

    result_mat = vsegment_func(result_mat, result_mat.max(), segments)

    img_show = np.zeros(result_mat.shape + (3,))
    img_show[:,:,0] = result_mat

    for point in graident_way:
        x = point[0]
        y = point[1]

        x = (x - min_r) / range_val
        x *= img_size
        y = (y - min_r) / range_val
        y *= img_size
        cv.circle(img_show, (int(x), int(y)), 1, (0, 0, 255), 10)


    cv.imshow('result', img_show)


def get_map_func(func, options):
    min_r = options['min_r']
    max_r = options['max_r']
    img_size = options['img_size']

    result_mat = np.fromfunction(
        lambda i, j : func(j,i, (min_r, max_r), img_size), 
        (img_size, img_size), 
        dtype=np.float32
    )
    return result_mat

def our_func(x, y, range, numpy_size):
    x = x / numpy_size
    y = y / numpy_size
    min_r = range[0]
    max_r = range[1]
    r_range = (max_r - min_r)
    
    x = x * r_range + min_r
    y = y * r_range + min_r

    return calculate_f(x, y)


def calculate_f(x, y):
    return 0.5*(x ** 3) + 10*(x ** 2) + 10*(y ** 2) + 2*y + y ** 3


def calculate_gradient_f(x, y):
    grad_x = 1.5 * (x ** 2) + 20 * x
    grad_y = 3 * (y ** 2) + 20 * y + 2

    return grad_x, grad_y


if __name__ == '__main__':
    options = {
        'img_size' : 900,
        'segments' : 30,
        'min_r' : -8.0,
        'max_r' : 5.0,
        'rate' : 0.1
    }

    rate = options['rate']

    gradient_way = [(-7.0, -6.3)]

    result_mat = get_map_func(our_func, options)


    while True:
        x, y = gradient_way[-1]
        delta_x, delta_y = grad_calc = calculate_gradient_f(
            x, 
            y
        )

        x += -delta_x * rate
        y += -delta_y * rate
        gradient_way.append((x, y))

        visualise_func(result_mat, gradient_way, options)
        key = cv.waitKey(500)
        if key == 27:
            break