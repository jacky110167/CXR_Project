import numpy as np

def conv_output_size(input_size, filter_size, stride = 1, pad = 0):
    return int((input_size - filter_size + 2 *pad) / stride + 1)


def im2col(input_data, fh, fw, stride=1, pad=0):
    n, c, h, w = input_data.shape
    
    out_h = conv_output_size(h, fh, stride, pad)
    out_w = conv_output_size(w, fw, stride, pad)

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col_x = np.zeros((n, c, fh, fw, out_h, out_w))

    for y in range(fh):
        y_max = y + stride * out_h
        for x in range(fw):
            x_max = x + stride * out_w
            col_x[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col_x = col_x.transpose(0, 4, 5, 1, 2, 3).reshape(n*out_h*out_w, -1)
    
    return col_x


def col2im(col_x, input_data, fh, fw, stride=1, pad=0):
    n, c, h, w = input_data.shape
    
    out_h = conv_output_size(h, fh, stride, pad)
    out_w = conv_output_size(w, fw, stride, pad)
    
    col_x = col_x.reshape(n, out_h, out_w, c, fh, fw).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((n, c, h + 2 * pad + stride - 1, w + 2*pad + stride - 1))
    
    for y in range(fh):
        y_max = y + stride * out_h
        
        for x in range(fw):
            x_max = x + stride * out_w
            
            img[:, :, y:y_max:stride, x:x_max:stride] += col_x[:, :, y, x, :, :]

    return img[:, :, pad:h + pad, pad:w + pad]