"""Template-matching digitiser for marker symbols in the raster figures of Liu & Shan (2019)."""
import cv2, numpy as np
from PIL import Image

def load(path):
    return np.array(Image.open(path).convert("RGB")).astype(int)

def mask_color(im, kind):
    R,G,B=im[...,0],im[...,1],im[...,2]
    if kind=="red":   m=(R>150)&(G<110)&(B<110)
    elif kind=="blue":m=(B>140)&(R<110)&(G<110)
    elif kind=="gray":m=(abs(R-G)<10)&(abs(G-B)<10)&(R>120)&(R<215)
    elif kind=="black":m=(R<90)&(G<90)&(B<90)
    return m.astype(np.uint8)*255

def template(shape, s, t=3):
    n=int(s*1.4)|1; T=np.zeros((n,n),np.uint8); c=n//2; r=s/2
    if shape=="circle": cv2.circle(T,(c,c),int(r),255,t)
    elif shape=="square": cv2.rectangle(T,(int(c-r),int(c-r)),(int(c+r),int(c+r)),255,t)
    elif shape=="triangle":
        pts=np.array([[c,c-r],[c+r*1.0,c+r*0.8],[c-r*1.0,c+r*0.8]],np.int32); cv2.polylines(T,[pts],True,255,t)
    elif shape=="diamond":
        pts=np.array([[c,c-r],[c+r,c],[c,c+r],[c-r,c]],np.int32); cv2.polylines(T,[pts],True,255,t)
    return T

def find(mask, shape, sizes, thr=0.5, t=3, nms=None):
    best=[]
    m=cv2.dilate(mask,np.ones((3,3),np.uint8))
    for s in sizes:
        T=template(shape,s,t)
        res=cv2.matchTemplate(m.astype(np.float32),cv2.dilate(T,np.ones((3,3),np.uint8)).astype(np.float32),cv2.TM_CCORR_NORMED)
        # weight by coverage of template by mask to avoid matches on solid blobs
        cov=cv2.matchTemplate(m.astype(np.float32)/255,(T>0).astype(np.float32),cv2.TM_CCORR)/ (T>0).sum()
        score=res*(cov>0.8)
        ys,xs=np.where(score>thr)
        for y,x in zip(ys,xs):
            best.append((score[y,x],x+T.shape[1]//2,y+T.shape[0]//2,s))
    best.sort(reverse=True)
    nms=nms or min(sizes)*0.6
    keep=[]
    for sc,x,y,s in best:
        if all((x-kx)**2+(y-ky)**2>nms**2 for _,kx,ky,_ in keep): keep.append((sc,x,y,s))
    return keep

def to_data(pts, frame, xr, yr):
    x0,x1,y0,y1=frame   # pixel: left,right,top,bottom
    X=[xr[0]+(x-x0)/(x1-x0)*(xr[1]-xr[0]) for _,x,y,_ in pts]
    Y=[yr[0]+(y1-y)/(y1-y0)*(yr[1]-yr[0]) for _,x,y,_ in pts]
    return np.array(X),np.array(Y)

def overlay(im, found, out, crop=None):
    img=cv2.cvtColor(im.astype(np.uint8),cv2.COLOR_RGB2BGR)
    for col,pts in found:
        for sc,x,y,s in pts: cv2.circle(img,(int(x),int(y)),4,col,-1); cv2.circle(img,(int(x),int(y)),int(s/2)+2,col,1)
    if crop: img=img[crop[2]:crop[3],crop[0]:crop[1]]
    cv2.imwrite(out,img)
