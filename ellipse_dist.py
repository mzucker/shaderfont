# Code adopted from https://www.shadertoy.com/view/4sS3zz
#
# The MIT License
# Copyright (C) 2013 Inigo Quilez
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions: The above copyright notice and
# this permission notice shall be included in all copies or
# substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS
# IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np

def ellipse_dist(p, ab):

    ab = np.abs(ab)
    
    ps = np.sign(p)
    p = np.abs(p)

    swap = p[:,0] > p[:,1]
    p = np.where(swap[:,None], p[:,::-1], p)
    ab = np.where(swap[:,None], ab[None,::-1], ab[None,:])

    l = ab[:,1]**2 - ab[:,0]**2

    m = ab[:,0] * p[:,0] / l
    n = ab[:,1] * p[:,1] / l
    m2 = m*m
    n2 = n*n

    c = (m2 + n2 - 1.) / 3.
    c3 = c*c*c

    q = c3 + m2*n2*2.
    d = c3 + m2*n2
    g = m + m*n2

    dneg = d < 0

    co = np.empty_like(g)

    h = np.arccos(q[dneg] / c3[dneg]) / 3.
    s = np.cos(h)
    t = np.sin(h) * np.sqrt(3)
    rx = np.sqrt( -c[dneg]*(s + t + 2.) + m2[dneg] )
    ry = np.sqrt( -c[dneg]*(s - t + 2.) + m2[dneg] )
    co[dneg] = ( ry + np.sign(l[dneg])*rx + np.abs(g[dneg])/(rx*ry) - m[dneg] ) / 2.

    dpos = ~dneg

    h = 2.*m[dpos]*n[dpos] * np.sqrt(d[dpos])
    s = np.sign(q[dpos]+h) * np.abs(q[dpos]+h)**(1.0/3.0)
    u = np.sign(q[dpos]-h) * np.abs(q[dpos]-h)**(1.0/3.0)
    rx = -s - u - c[dpos]*4. + 2.*m2[dpos]
    ry = (s - u)*np.sqrt(3.)
    rm = np.sqrt( rx*rx + ry*ry )
    co[dpos] = (ry/np.sqrt(rm-rx) + 2.*g[dpos]/rm - m[dpos])/2.

    si = np.sqrt( 1 - co*co )

    r = ab * np.hstack( [ co.reshape(-1, 1), si.reshape(-1, 1) ] )
    
    dist = np.linalg.norm(r-p, axis=1) * np.sign(p[:,1] - r[:,1])

    r = np.where(swap[:,None], r[:, ::-1], r)

    return np.hstack( ( r*ps, dist[:,None] ) )
