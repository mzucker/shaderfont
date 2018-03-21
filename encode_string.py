import sys
import numpy as np

######################################################################

def glsl_i(t):
    if t == 0:
        return '0'
    d = str(t) 
    h = hex(t).rstrip('L') 
    if len(d) < len(h):
        return d
    else:
        return h

######################################################################
    
def glsl_ivec(tile):
    return 'ivec{}({})'.format(
        len(tile),
        ', '.join( [ glsl_i(t) for t in tile ] ) )

######################################################################

def main():

    string_in = ' '.join(sys.argv[1:])

    # so you have 7 bits per character
    # uvec4 has 4*32 = 128 bits
    # float has 32 bits
    # 4*7 characters = 28 bits
    # 2 bits left for count within float
    # then we have 4*4 = 16 characters per uvec

    l = len(string_in)
    lmod16 = l % 16

    if lmod16:
        lextra = 16 - lmod16
    else:
        lextra = 0
        
    string_in += '\0' * lextra

    assert len(string_in) % 16 == 0

    data = np.zeros(len(string_in) / 4, dtype=np.uint32)

    for i in range(len(data)):
        i0 = i*4
        for j in range(4):
            cidx = ord(string_in[i0+j])
            if cidx not in range(127):
                cidx = 0
            data[i] |= (cidx << (j*7))

    string_out = ''
    for i in range(len(data)):
        for j in range(4):
            x = (data[i] >> (j*7)) & 127
            string_out += chr(x)

    assert string_out == string_in

    data = data.reshape(-1, 4)
    print ',\n'.join( [glsl_ivec(di) for di in data] )
    print len(data)

    
    
######################################################################

if __name__ == '__main__':
    main()

