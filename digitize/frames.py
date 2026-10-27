import numpy as np
from PIL import Image

def longest_run(b):
    # b: 2D bool, returns longest True run per column
    H,W=b.shape
    best=np.zeros(W,int); cur=np.zeros(W,int)
    for i in range(H):
        cur=np.where(b[i],cur+1,0); best=np.maximum(best,cur)
    return best

def groups(idx):
    out=[];s=None;p=None
    for v in idx:
        if s is None: s=v
        elif v-p>2: out.append((s,p)); s=v
        p=v
    if s is not None: out.append((s,p))
    return out

def find_frames(path, minw_frac=0.6, minh=250):
    im=np.array(Image.open(path).convert('L')).astype(int)
    dark=im<110
    cols=np.where(longest_run(dark)>=minh)[0]
    rows=np.where(longest_run(dark.T)>=minw_frac*im.shape[1]*0.8)[0]
    return groups(rows),groups(cols),im.shape
if __name__=="__main__":
    import sys
    for f in sys.argv[1:]:
        print(f,find_frames(f))
