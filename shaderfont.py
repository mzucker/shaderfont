# shaderfont.py 
# Tiny proportional vector font for pixel shaders.
#
# License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported
# https://creativecommons.org/licenses/by-nc-sa/3.0/

from __future__ import print_function

import re
import sys
from collections import namedtuple

import numpy as np
from PIL import Image

######################################################################

'''
opcodes:

000 M moveto  x y
001 T clipping x y
010 C circle anchor x y
011 A angular start length
100 E ellipse x y 
101 D half-ellipse x y 
110 U half-ellipse x y
111 L lineto x y

3 bit opcode
5 bit immediate data (2x)

13 bits per full instruction

26 bits is 2 instructions
leaves 4 bits easily accessible per 30-bit word

which is handy because we need to store:

0 clip (2bits) + symmetry (2 bits)
1 width (4bits)
2 height (4bits)
3 y descent (4 bits)

'''

######################################################################

OPCODES = 'MTCAEDUL'

INTEGER_EXPR = r'(-?[0-9]+)'
OPCODE_EXPR = r'([' + OPCODES + '])'
SEP_EXPR = r'[, ] *'
OPTIONAL_SEP_EXPR = r'[, ]? *'

INSTRUCTION_EXPR = OPCODE_EXPR + OPTIONAL_SEP_EXPR + INTEGER_EXPR + SEP_EXPR + INTEGER_EXPR
PROGRAM_EXPR = r'^(' + INSTRUCTION_EXPR + OPTIONAL_SEP_EXPR + r')*$'

IMMEDIATE_DATA_RANGE = range(-16, 16)
ANGLE_RUN_VALUES = set(range(-16, 17)) - set([0])

MAX_INSTRUCTION_COUNT = 8

WIDTH_HEIGHT_RANGE = range(16)
Y0_RANGE = range(-8, 8)

NOSYM = 0
SYM_X = 1
SYM_Y = 2
SKEWY = 3

# we invert the expected bit values to make CLIPXY be value 0
NOCLIP = 0
CLIP_X = 1
CLIP_Y = 2
CLIPXY = 3


PX_PER_UNIT = 8


THICKNESS = 0.75
GLYPH_SEP = THICKNESS

ISOLINES = 0
MAX_MITER_ZONE = 3.
SHADE_EXTENTS = True

FONT = [

    # 32-47
    (' ',  6,  1,  0, CLIPXY, NOSYM, ''),
    ('!',  2, 10,  0, CLIPXY, SYM_X, 'M1,9 L1,4 T1,5 L-1,12 M1,1 L1,1'),
    ('"',  5,  3,  7, CLIPXY, NOSYM, 'M1,8 L1,9 T1,9 L0,8 M4,8 L4,9 T4,9 L3,8'),
    ('#',  7, 10,  0, CLIPXY, SYM_Y, 'M1,3 L6,3 M2,0 L2,5 M5,0 L5,5'),
    ('$',  6, 12, -1, CLIPXY, SKEWY, 'A-13,5 E5,5 D5,5 D1,9 M3,0 L3,1'),
    ('%', 10, 10,  0, CLIP_X, NOSYM, 'M2,1 L8,9 M6,1 E9,4 M1,6 E4,9'),
    ('&',  7, 10,  0, CLIPXY, NOSYM, 'E5,5 M5,6 L5,3 U9,1 M5,9 L3,9 D1,5 L6,5'),
    ("'",  2,  3,  7, CLIPXY, NOSYM, 'M1,8 L1,9 T1,9 L0,8'),
    ('(',  3, 10,  0, CLIPXY, NOSYM, 'E5,9'),
    (')',  3, 10,  0, CLIPXY, NOSYM, 'C-2,1 E2,9'),
    ('*',  6,  6,  4, CLIP_X, SYM_X, 'M1,6 L3,7 L1,8 M3,5 L3,9'),
    ('+',  6,  6,  2, CLIPXY, NOSYM, 'M3,3 L3,7 M1,5 L5,5'),
    (',',  2,  3, -1, CLIPXY, NOSYM, 'M1,0 L1,1 T1,1 L0,0'),
    ('-',  6,  2,  4, CLIPXY, NOSYM, 'M1,5 L5,5'),
    ('.',  2,  2,  0, CLIPXY, NOSYM, 'L1,1'),
    ('/',  5, 10,  0, CLIP_Y, NOSYM, 'M1,0 L4,10'),

    # 48-63
    ('0',  7, 10,  0, CLIPXY, NOSYM, 'M1,1 L6,9 M1,1 E6,9 T1,1 E6,9'),
    ('1',  4, 10,  0, CLIPXY, NOSYM, 'L3,1 M2,1 L2,9 L0,8'),
    ('2',  6, 10,  0, CLIPXY, NOSYM, 'M5,1 L1,1 C1,5 A-3,3 E5,9 U1,9'),
    ('3',  6, 10,  0, CLIPXY, SYM_Y, 'A-14,6 E5,5 D5,5'),
    ('4',  6, 10,  0, CLIPXY, NOSYM, 'M4,1 L4,9 L1,4 L5,4'),
    ('5',  6, 10,  0, CLIPXY, NOSYM, 'A-15,7 E5,6 D5,6 L1,6 L2,9 L5,9 L6,12'),
    ('6',  6, 10,  0, CLIPXY, NOSYM, 'M6,9 A6,10 E1,1 L1,3 U5,1 L5,4 U1,6'),
    ('7',  6, 10,  0, CLIPXY, NOSYM, 'M1,9 L5,9 L1,-1'),
    ('8',  6, 10,  0, CLIPXY, SYM_Y, 'E5,5'),
    ('9',  6, 10,  0, CLIPXY, NOSYM, 'C5,1 A-11,11 E0,9 L5,7 U1,9 L1,6 U5,4'),
    (':',  2,  7,  0, CLIPXY, NOSYM, 'L1,1 M1,6 L1,6'),
    (';',  2,  8, -1, CLIPXY, NOSYM, 'M1,0 L1,1 M1,6 L1,6 T1,1 L0,0'),
    ('<',  6,  8,  1, CLIP_X, SYM_Y, 'M6,2 L1,5'),
    ('=',  6,  5,  2, CLIPXY, NOSYM, 'M1,3 L5,3 M1,6 L5,6'),
    ('>',  6,  8,  1, CLIP_X, SYM_Y, 'M0,2 L5,5'),
    ('?',  6, 10,  0, CLIP_Y, NOSYM, 'M1,5 A14,-6 E5,9 D5,5 L3,4 M3,1 L3,1'),
    ('@',  8, 10,  0, CLIPXY, NOSYM, 'A-7,-9 E7,9 U7,9 M4,3 E7,6 T1,1 E7,9'),

    # 64-79
    ('A',  8, 10,  0, CLIPXY, SYM_X, 'M0,-3 L2,3 L4,3 M2,3 L4,9'),
    ('B',  6, 10,  0, CLIPXY, SYM_Y, 'L3,1 D5,5 L1,5 L1,1'),
    ('C',  7, 10,  0, CLIPXY, SYM_Y, 'A11,16 E7,9'),
    ('D',  7, 10,  0, CLIPXY, NOSYM, 'L3,1 D6,9 L1,9 L1,1'),
    ('E',  6, 10,  0, CLIPXY, SYM_Y, 'M5,1 L1,1 L1,5 L4,5'),
    ('F',  6, 10,  0, CLIPXY, NOSYM, 'L1,9 L5,9 M1,5 L4,5'),
    ('G',  9, 10,  0, CLIPXY, NOSYM, 'A4,12 E8,9 U8,1 L5,5'),
    ('H',  7, 10,  0, CLIPXY, SYM_Y, 'L1,5 L6,5 L6,1'),
    ('I',  2, 10,  0, CLIPXY, NOSYM, 'L1,9'),
    ('J',  6, 10,  0, CLIPXY, NOSYM, 'M5,9 L5,3 U1,1'), 
    ('K',  7, 10,  0, CLIP_Y, SYM_Y, 'L1,5 L2,5 L8,-3'),
    ('L',  6, 10,  0, CLIPXY, NOSYM, 'L1,9 M1,1 L5,1'),
    ('M',  9, 10,  0, CLIPXY, SYM_X, 'L1,9 L7,1'),
    ('N',  7, 10,  0, CLIPXY, NOSYM, 'L1,9 L6,1 L6,9'),
    ('O',  8, 10,  0, CLIPXY, NOSYM, 'E7,9'),

    # 80-95
    ('P',  6, 10,  0, CLIPXY, NOSYM, 'L1,9 L3,9 D5,5 L1,5'),
    ('Q',  8, 10,  0, NOCLIP, NOSYM, 'E7,9 M5,3 L7,1'),
    ('R',  6, 10,  0, CLIPXY, NOSYM, 'L1,9 L3,9 D5,5 L1,5 M2,5 L3,5 L6,-2'), 
    ('S',  6, 10,  0, CLIPXY, SKEWY, 'A-13,5 E5,5 D5,5 D1,9'),
    ('T',  6, 10,  0, CLIPXY, NOSYM, 'M1,9 L5,9 M3,1 L3,9'),
    ('U',  6, 10,  0, CLIPXY, SYM_X, 'M5,3 U1,1 L1,9'),
    ('V',  8, 10,  0, CLIP_Y, SYM_X, 'M1,10 L4,1'),
    ('W', 11, 10,  0, CLIP_Y, SYM_X, 'M1,11 L3,1 L7,9'),
    ('X',  8, 10,  0, CLIP_Y, NOSYM, 'M0,-2 L8,12 M0,12 L8,-2'),
    ('Y',  8, 10,  0, CLIP_Y, SYM_X, 'M0,12 L5,3 M4,0 L4,5'),
    ('Z',  6, 10,  0, CLIPXY, SKEWY, 'M5,1 L1,1 L5,9'),
    ('[',  3, 10,  0, CLIPXY, SYM_Y, 'M2,1 L1,1 L1,9'),
    ('\\', 5, 10,  0, CLIP_Y, NOSYM, 'M1,10 L4,0'),
    (']',  3, 10,  0, CLIPXY, SYM_Y, 'L2,1 L2,9'),
    ('^',  6,  7,  3, CLIP_X, SYM_X, 'M-1,4 L5,11'),
    ('_',  6,  2, -1, CLIPXY, NOSYM, 'M1,0 L5,0'),

    # 96-111
    ('`',  2,  3,  7, CLIPXY, NOSYM, 'M1,8 L1,9 T1,9 L0,10'),
    ('a',  6,  8,  0, CLIPXY, NOSYM, 'E5,4 M5,1 L5,5 C5,4 A0,11 E1,7'),
    ('b',  6, 10,  0, CLIPXY, NOSYM, 'L1,9 M1,1 E5,7'),
    ('c',  5,  8,  0, CLIP_Y, SYM_Y, 'A11,16 E5,7'),
    ('d',  6, 10,  0, CLIPXY, NOSYM, 'E5,7 M5,1 L5,9'),    
    ('e',  6,  8,  0, CLIPXY, NOSYM, 'A-4,-12 E5,7 U5,7 L1,4 T1,1 E5,7'),
    ('f',  6, 10,  0, CLIPXY, NOSYM, 'M2,1 L2,7 C2,5 A-16,-13 E5,9 M1,5 L4,5'),
    ('g',  6, 12, -4, CLIPXY, NOSYM, 'E5,7 M5,7 L5,0 A0,-13 C5,3 E1,-3'),
    ('h',  6, 10,  0, CLIPXY, NOSYM, 'L1,9 M1,1 L1,4 U5,7 L5,1'),
    ('i',  2, 10,  0, CLIPXY, NOSYM, 'L1,6 M1,9 L1,9'),
    ('j',  6, 14, -4, CLIPXY, NOSYM, 'M5,9 L5,9 M5,6 L5,0 A0,-13 C5,3 E1,-3'),
    ('k',  6, 10,  0, CLIPXY, NOSYM, 'L1,9 M4,6 L1,4 L5,0'),
    ('l',  2, 10,  0, CLIPXY, NOSYM, 'L1,9'),
    ('m', 10,  8,  0, CLIPXY, NOSYM, 'L1,7 M1,1 L1,4 U5,7 L5,1 M5,4 U9,7 L9,1'),
    ('n',  6,  8,  0, CLIPXY, NOSYM, 'L1,7 M1,1 L1,4 U5,7 L5,1'),
    ('o',  6,  8,  0, CLIPXY, NOSYM, 'E5,7'),

    # 112-126
    ('p',  6, 11, -3, CLIPXY, NOSYM, 'E5,7 M1,-2 L1,7'),
    ('q',  6, 11, -3, CLIPXY, NOSYM, 'E5,7 M5,-2 L5,7'),
    ('r',  6,  8,  0, CLIPXY, NOSYM, 'L1,4 A-16,-13 E5,7 M1,1 L1,7'),
    ('s',  6,  8,  0, CLIPXY, NOSYM, 'A-13,5 E5,4 D5,4 D1,7 C1,7 A8,-5 E5,4'),
    ('t',  6, 10,  0, CLIPXY, NOSYM, 'M2,9 L2,3 C2,1 A-16,13 E5,5 M1,7 L4,7'),
    ('u',  6,  8,  0, CLIPXY, NOSYM, 'M1,7 L1,4 U5,1 M5,1 L5,7'),
    ('v',  6,  8,  0, CLIPXY, SYM_X, 'M1,8 L3,1'),
    ('w', 10,  8,  0, CLIPXY, SYM_X, 'M1,8 L3,1 L5,6 L6,6'),
    ('x',  6,  8,  0, CLIPXY, NOSYM, 'M1,0 L5,8 M1,8 L5,0'),
    ('y',  6, 12, -4, CLIPXY, NOSYM, 'M1,7 L1,4 U5,1 M5,7 L5,0 A0,-13 C5,3 E1,-3'),
    ('z',  6,  8,  0, CLIPXY, SKEWY, 'M5,1 L1,1 L5,7'),
    ('{',  4, 10,  0, CLIP_X, SYM_Y, 'C0,3 A8,-8 E2,5 A-16,8 C2,1 E4,3'),
    ('|',  2, 10,  0, CLIPXY, NOSYM, 'L1,9'),
    ('}',  4, 10,  0, CLIP_X, SYM_Y, 'C4,3 A8,8 E2,5 A0,-8 C2,1 E0,3'), 
    ('~',  7,  3,  4, CLIPXY, NOSYM, 'C1,4 A-16,-9, E4,6 C6,5 A-9,9 E3,7'),
    
]

GlyphInfo = namedtuple('GlyphInfo', 'char, width, height, y0, clip, sym, program')

FONT = [ GlyphInfo(*glyph) for glyph in FONT ]

######################################################################

def smoothstep(lo, hi, X):

    rval = (X - lo) / (hi - lo)
    rval = np.clip(rval, 0, 1)
    rval = 3*rval**2 - 2*rval**3

    return rval

######################################################################

def max_combine(X, Y=None):
    if Y is None:
        return X.copy()
    else:
        return np.maximum(X, Y)

######################################################################
    
def min_combine(X, Y=None):
    if Y is None:
        return X.copy()
    else:
        return np.minimum(X, Y)

######################################################################
    
def perp(v):
    return np.array( [ -v[1], v[0] ] ).copy()

######################################################################

def from_angle(t):
    return np.array( [ np.cos(t), np.sin(t) ] )

######################################################################

def normalize(x):
    return x / np.linalg.norm(x)

######################################################################

def half_plane_dist(p0, p1, p):
    d10 = p1 - p0
    n = perp(d10)
    n /= np.linalg.norm(n)
    return np.dot(p, n) - np.dot(p0, n)
    
######################################################################

def box_dist(ctr, rad, p):
    return np.max( np.abs(p - ctr) - rad, axis=1 )

######################################################################

def line_dist(p0, p1, p):

    d10 = p1 - p0

    l = np.linalg.norm(d10)

    if l < 1e-5:
        t = [0, 1]
    else:
        t = d10 / l
    
    n = np.array([-t[1], t[0]])

    pc = 0.5*(p1 + p0)

    p = p - pc

    R = np.array([t, n])

    p = np.dot(p, R.T)

    return box_dist([0,0], [0.5*l, 0], p)

######################################################################

def mydot(x,y):
    return np.sum(x*y, axis=1)

######################################################################

# TODO: investigate blocky artifacts with short-aspect-ratio ellipses
def ellipse_dist(ctr, ab, p, alim, filled):

    p = (p - ctr)

    ab = np.abs(ab)
    a = ab[0]
    b = ab[1]

    x = p[:,0]
    y = p[:,1]

    x2 = x**2
    y2 = y**2

    a2 = a**2
    b2 = b**2

    d = np.sqrt(x2/a2 + y2/b2) - 1

    f = np.maximum(a2*y2 + b2*x2, 1e-5)

    dfdx2 = b2*x2 / (a2*f)
    dfdy2 = a2*y2 / (b2*f)
    
    denom = np.sqrt(dfdx2 + dfdy2)

    d /= np.maximum(denom, 1e-5) 

    if not filled:
        d = abs(d)

    if alim[1]:

        ax = alim[0]
            
        start_angle = ax*np.pi/16 # 32 sectors
            
        block_size = alim[1]
        end_angle = start_angle + alim[1]*np.pi/16 # 32 sectors
        
        cdir = np.sign(end_angle - start_angle)

        p0 = from_angle(start_angle)*ab
        p1 = from_angle(end_angle)*ab

        ba = ab[::-1]

        clip0 = normalize( from_angle(start_angle)*ba )
        clip1 = normalize( from_angle(end_angle)*ba )

        path_tangent0 = perp(clip0)*cdir
        path_tangent1 = perp(clip1)*cdir

        # on positive side of this, closer to tangent 1, otherwise
        # closer to tangent 0
        n_split = perp(normalize( path_tangent0 - path_tangent1 )) * cdir

        d_split = np.dot(p, n_split) < 0

        d0 = np.linalg.norm(p - p0, axis=1)
        d1 = np.linalg.norm(p - p1, axis=1)

        # choose closest endpoint
        p_end = np.where(d_split[:,None], p0[None,:], p1[None,:])
        n = np.where(d_split[:,None], -path_tangent0[None,:], path_tangent1[None,:])
        t = np.where(d_split[:,None], clip0[None,:], clip1[None,:])

        d_end = mydot(p - p_end, n)
        
        if not filled:
            d_end = np.maximum( np.abs(d_end), np.abs( mydot(p - p_end, t) ) )

        d = np.where(mydot(p - p_end, n) < 0, d, d_end)

    else:

        p0 = np.array([a, 0])
        p1 = p0
        path_tangent0 = np.array([0, 1])
        path_tangent1 = path_tangent0

    return d, ctr+p0, path_tangent0, ctr+p1, path_tangent1

######################################################################

def clip_angles(p0, alim, p):

    p = p - p0
    
    start_angle = alim[0]*np.pi/8
    block_size = alim[1]
    end_angle = start_angle + alim[1]*np.pi/16
    
    if start_angle > end_angle:
        start_angle, end_angle = end_angle, start_angle

    n0 = np.array([ -np.sin(start_angle), np.cos(start_angle) ])
    n1 = np.array([ -np.sin(end_angle), np.cos(end_angle) ])
        
    return -np.maximum(np.dot(p, -n0), np.dot(p, n1))

######################################################################

def align(u, v):
    sign = -1. if np.dot(u,v) < 0 else 1.
    return u * sign

######################################################################

def bisect(ta, tc):
    if (np.dot(ta, tc) < -0.9999):
        return perp(ta)
    else:
        return align(normalize(ta + tc), ta)
            
######################################################################

def miter(da, dc, p, ta, tc):

    tmid = bisect(ta, tc)
    nsplit = align(perp(tmid), ta)

    na = align(perp(ta), tc)
    nc = align(perp(tc), ta)

    pna = np.dot(p, na)
    pnc = np.dot(p, nc)

    pta = np.dot(p, ta)
    ptc = np.dot(p, tc)

    ps = np.dot(p, nsplit)

    amask = np.abs(pna) < MAX_MITER_ZONE
    da = np.where((pta > 0) & amask, pna, da)    
    da = np.where((np.minimum(ps, pta) > 0) & amask, 1e5, da)

    cmask = np.abs(pnc) < MAX_MITER_ZONE
    dc = np.where((ptc > 0) & cmask, pnc, dc)    
    dc = np.where((np.minimum(-ps, ptc) > 0) & cmask, 1e5, dc)

    return da, dc

######################################################################

def tokenize(glyph):

    if not re.match(PROGRAM_EXPR, glyph.program):
        raise RuntimeError('bad program for {}: {}'.format(
            glyph.char, glyph.program))

    instructions = re.findall(INSTRUCTION_EXPR, glyph.program)

    assert len(instructions) <= MAX_INSTRUCTION_COUNT

    instructions = [ (opcode, int(x), int(y)) for (opcode, x, y) in instructions ]
    
    for (opcode, x, y) in instructions:
        
        assert x in IMMEDIATE_DATA_RANGE

        if opcode == 'A':
            assert y in ANGLE_RUN_VALUES
        else:
            assert y in IMMEDIATE_DATA_RANGE

    return instructions

######################################################################

def symmetrize(p, width, height, y0, sym):

    if sym == SYM_X:
        p[:,0] = 0.5*width - np.abs(p[:,0] - 0.5*width)
    elif sym == SYM_Y or sym == SKEWY:
        mask = p[:,1]-y0 > 0.5*height
        p[:,1] = y0 + 0.5*height - np.abs(p[:,1] - y0 - 0.5*height)
        if sym == SKEWY:
            p[mask,0] = width - p[mask,0]

def rasterize(glyph, scl, p, dst):


    assert glyph.width in WIDTH_HEIGHT_RANGE
    assert glyph.height in WIDTH_HEIGHT_RANGE
    assert glyph.y0 in Y0_RANGE

    instructions = tokenize(glyph)

    print('*** rasterizing {} ***'.format(glyph.char))
    print()

    symmetrize(p, glyph.width, glyph.height, glyph.y0, glyph.sym)

    dist_field = None

    prev_stroke = None
    
    prev_t1 = np.array([0., 0.])
    prev_opcode = None

    p0 = np.array([1., 1.,])
    ellipse_corner = p0

    clip_mode = False

    alim = np.array([0., 0.])

    for opcode, x, y in instructions:

        print('instruction is', opcode, x, y)

        p1 = np.array([x, y], dtype=float)

        if opcode == 'C':
            
            ellipse_corner = p1
            continue
        
        elif opcode == 'A':
            
            alim = np.array([x, y])
            continue
        
        elif opcode in 'MT':
            
            if prev_stroke is not None:
                print('  stroking {}'.format(prev_opcode))
                dist_field = min_combine(prev_stroke, dist_field)

            prev_stroke = None
            prev_t1 = np.array([0., 0.])
            alim = np.array([0., 0.])
            ellipse_corner = p1
            p0 = p1
            clip_mode = (opcode == 'T')
            continue

        assert opcode in 'EDUL' # Go Delaware?

        connect_ellipse = False

        if opcode in 'EDU':

            m10 = 0.5 * (p1 + p0)
            d10 = (p1 - p0)
            delta = np.sign(d10)

            if opcode == 'D':
                ctr = [p0[0], m10[1]]
                rad = d10 * [1, 0.5]
                alim = [-delta[1]*8, 16*delta[0]*delta[1]]
            elif opcode == 'U':
                ctr = [m10[0], p0[1]]
                rad = d10 * [0.5, 1]
                alim = [8+delta[0]*8, -16*delta[0]*delta[1]]
            else:
                ctr = 0.5 * (p1 + ellipse_corner)
                rad = 0.5 * (p1 - ellipse_corner)

            estroke, p1, et0, ep1, et1 = ellipse_dist(
                ctr, rad, p, alim, clip_mode)

            connect_ellipse = (not clip_mode and 
                               prev_stroke is not None and
                               np.linalg.norm(prev_t1) and
                               np.linalg.norm(p1 - p0) > 1e-3)

        if opcode == 'L' or connect_ellipse:
           
            if clip_mode:
                cur_stroke = half_plane_dist(p0, p1, p)
            else:
                cur_stroke = line_dist(p0, p1, p)

            cur_t0 = p1-p0
            n = np.linalg.norm(cur_t0)
            if n:
                cur_t0 /= n

            cur_t1 = cur_t0

            print('  making {}line from {} to {} with tangent {}'.format(
                ('connecting ' if connect_ellipse else ''),
                p0, p1, cur_t0))

        if opcode in 'UDE':

            if connect_ellipse:

                assert np.linalg.norm(prev_t1)
                
                print('  calling miter between {} and '
                      'connecting line, cur_t0 is {}'.format(
                          prev_opcode, cur_t0))

                prev_stroke, cur_stroke = miter(prev_stroke, cur_stroke,
                                                p-p0, prev_t1, -cur_t0)

                print('  stroking {}'.format(prev_opcode))
                
                dist_field = min_combine(prev_stroke, dist_field)

                
                prev_stroke = cur_stroke
                prev_t1 = cur_t1
                prev_opcode = 'connecting line'

            cur_stroke = estroke
            p0 = p1
            p1 = ep1
            cur_t0 = et0
            cur_t1 = et1

            print(('  making ellipse arc from {} to {} '
                  'starting in dir {} ending in dir {}').format(
                      p0, p1, cur_t0, cur_t1))

        if clip_mode:

            dist_field = max_combine(cur_stroke, dist_field)
            prev_stroke = None
            prev_t1 = np.zeros(2)

        elif cur_stroke is not None:

            if np.linalg.norm(prev_t1) and np.linalg.norm(cur_t0):
                
                assert prev_stroke is not None

                print('  calling miter between {} and {}, '
                      'cur_t0 is {}'.format(
                          prev_opcode, opcode, cur_t0))
                prev_stroke, cur_stroke = miter(prev_stroke, cur_stroke,
                                                p-p0, prev_t1, -cur_t0)

                print('  stroking {}'.format(prev_opcode))
                dist_field = min_combine(prev_stroke, dist_field)

            prev_stroke = cur_stroke
            prev_t1 = cur_t1

        p0 = p1
        prev_opcode = opcode
        
    if prev_stroke is not None:
        print('  stroking {}'.format(prev_opcode))
        dist_field = min_combine(prev_stroke, dist_field)

    print()

    clipy = np.abs(p[:,1]-0.5*glyph.height-glyph.y0)-0.5*glyph.height+1
    clipx = np.abs(p[:,0]-0.5*glyph.width)-0.5*glyph.width+1

    #if g.clip == CLIP_Y or g.clip == CLIPXY:
    if glyph.clip & CLIP_Y:
        if dist_field is not None:
            dist_field = max_combine(dist_field, clipy)
    else:
        clipy -= 0.4
        
    if glyph.clip & CLIP_X:
        if dist_field is not None:
            dist_field = max_combine(dist_field, clipx)
    else:
        clipx -= 0.4

    if dist_field is not None:
        dist_field = dist_field.reshape(dst.shape)
        dst[:] = np.minimum(dst, smoothstep(0, scl, dist_field-THICKNESS))

        for t in np.linspace(0, THICKNESS, ISOLINES, False):
            dst[:] = np.maximum(dst, smoothstep(scl, 0, abs(dist_field-t)-0.25*scl))

    if SHADE_EXTENTS:
        clip_rect = (np.maximum(clipy, clipx)-THICKNESS).reshape(dst.shape)
        dst[:] = np.minimum(dst, smoothstep(0, scl, clip_rect)*0.15+0.85)

######################################################################

def make_fontmap_image():

    np.set_printoptions(precision=3)

    img_width = 148 * PX_PER_UNIT
    img_height = 80 * PX_PER_UNIT

    scl = 1.0 / PX_PER_UNIT

    image = np.ones((img_height, img_width))
    dst_x = 0
    dst_y = 16*PX_PER_UNIT

    rng = np.linspace(-4, 12, 16*PX_PER_UNIT, False) + 0.5*scl
    
    X, Y = np.meshgrid(rng, rng[::-1])
    porig = np.hstack( ( X.reshape(-1, 1), Y.reshape(-1, 1) ) )

    ascii_codes = set(ord(f[0]) for f in FONT)
    assert ascii_codes == set(range(32,127))

    for glyph in FONT:

        subimage = image[dst_y - 16*PX_PER_UNIT:dst_y, dst_x:dst_x+16*PX_PER_UNIT]

        rasterize(glyph, scl, porig.copy(), subimage)

        char, width = glyph[:2]

        dst_x += int(round((width + GLYPH_SEP) * PX_PER_UNIT))

        if dst_x + 16*PX_PER_UNIT > image.shape[1]:
            dst_x = 0
            dst_y += 16*PX_PER_UNIT
            if dst_y > image.shape[0]:
                print('quitting early after', char)

    pil_image = Image.fromarray((image*255).astype(np.uint8), 'L')
    pil_image.save('font.png')
    print('wrote font.png')


######################################################################

def main():

    make_fontmap_image()

    
######################################################################

if __name__ == '__main__':
    main()

