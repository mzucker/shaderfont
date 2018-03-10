# ellipse.py 
# Test code for ellipse_dist module 


from __future__ import print_function

import sys
import numpy as np
from ellipse_dist import ellipse_dist

from PIL import Image



def smoothstep(lo, hi, X):

    rval = (X - lo) / (hi - lo)
    rval = np.clip(rval, 0, 1)
    rval = 3*rval**2 - 2*rval**3

    return rval

def main():

    rng = np.linspace(-10, 10, 512, False)
    X, Y = np.meshgrid(rng, rng)

    p = np.hstack( ( X.reshape(-1, 1), Y.reshape(-1,1) ) )
    ab = np.array([9., 4.])

    xyd = ellipse_dist(p, ab)
    d = xyd[:,2]

    scl = rng[1] - rng[0]

    f = 12.

    np.random.seed(123456)
    pts = np.random.random((25,2)) * 20 - 10

    epts = ellipse_dist(pts, ab)[:,:2]

    image = np.ones_like(d)
    image *= 0.8 + 0.2*smoothstep(0.5*scl, -0.5*scl, np.cos(f*d)/f)
    image *= smoothstep(0, scl, abs(d)-0.25*scl)

    for pt, ept in zip(pts, epts):
        dp = np.linalg.norm(p - pt, axis=1)
        de = np.linalg.norm(p - ept, axis=1)
        d = np.minimum(dp, de) - 4.*scl

        u = np.clip(np.dot(p - pt, ept - pt) / np.dot(ept - pt, ept - pt),
                    0., 1.)

        pc = pt + u[:,None] * (ept - pt)
        d = np.minimum(d, np.linalg.norm(p - pc, axis=1)-0.25*scl)
        
        image *= smoothstep(0, scl, d)
    
    image = image.reshape(X.shape)

    pil_image = Image.fromarray((image*255).astype(np.uint8), 'L')
    pil_image.save('ellipse.png')

    print('wrote ellipse.png')
    
    
if __name__ == '__main__':
    main()
