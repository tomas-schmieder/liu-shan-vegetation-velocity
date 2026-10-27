import sys, json; sys.path.insert(0,'.')
from dig import *

def clean(m, frame):
    x0,x1,y0,y1=[int(round(v)) for v in frame]
    out=np.zeros_like(m); out[y0+3:y1-2, x0+3:x1-2]=m[y0+3:y1-2, x0+3:x1-2]; return out

def gray_mask(im):
    R,G,B=im[...,0],im[...,1],im[...,2]
    m=((abs(R-G)<8)&(abs(G-B)<8)&(R>120)&(R<232)).astype(np.uint8)*255
    dark=cv2.dilate(((R<100)&(G<100)&(B<100)).astype(np.uint8)*255,np.ones((5,5),np.uint8))
    m[dark>0]=0; return m

def det(im, frame, kind, shape, sizes, thr, t=3, NMS=0.55):
    if kind=="gray": m=gray_mask(im)
    else: m=mask_color(im,kind)
    m=clean(m,frame)
    mm=cv2.dilate(m,np.ones((3,3),np.uint8)).astype(np.float32)/255
    best=[]
    for s in sizes:
        T=template(shape,s,t); tt=(cv2.dilate(T,np.ones((3,3),np.uint8))>0).astype(np.float32)
        cov=cv2.matchTemplate(mm,tt,cv2.TM_CCORR)/tt.sum()
        ys,xs=np.where(cov>thr)
        for y,x in zip(ys,xs): best.append((cov[y,x],x+T.shape[1]//2,y+T.shape[0]//2,s))
    best.sort(reverse=True); keep=[]
    for sc,x,y,s in best:
        if all((x-kx)**2+(y-ky)**2>(NMS*min(sizes))**2 for _,kx,ky,_ in keep): keep.append((sc,x,y,s))
    return keep

JOBS={}
LB={1:2.5,2:7.5,3:12.5,4:100/30,5:250/30,6:400/30,7:100/30,8:200/30,9:10.0}
# figure 5 (this study)
f5=[("fig-000.png",(146,1422,22,393.5)),("fig-000.png",(146,1422,576,947)),("fig-000.png",(146,1422,1129.5,1500.5))]
for i,(fn,fr) in enumerate(f5):
    for kind,shape,case,sz,thr in (("blue","triangle",3*i+1,[28,30,32],0.6),("red","circle",3*i+2,[26,28,30,32,34],0.4),("gray","square",3*i+3,[25,27,29,31,33],0.45)):
        JOBS[f"Case{case}"]=(fn,fr,(-10,15),(0,1.2),kind,shape,sz,thr)
if __name__=="__main__":
    res={}; ov={}
    for name,(fn,fr,xr,yr,kind,shape,sz,thr) in JOBS.items():
        im=load(fn); p=det(im,fr,kind,shape,sz,thr,NMS=0.8 if kind=="gray" else 0.55)
        n=int(name[4:]); Lb=LB[n]
        Xp,Yp=to_data(p,fr,xr,yr)
        p=[q for q,x,y in zip(p,Xp,Yp) if x<=Lb+0.3 and not (x>-0.6 and y>1.04)]
        X,Y=to_data(p,fr,xr,yr); o=np.argsort(X)
        res[name]=dict(x_over_b=X[o].round(4).tolist(),Ud_over_U0=Y[o].round(4).tolist())
        print(name,len(p)); ov.setdefault(fn,[]).append(((0,0,255) if kind=="red" else (255,0,0) if kind=="blue" else (0,160,0),p))
    json.dump(res,open('digitized_fig5.json','w'))
    for fn,l in ov.items(): overlay(load(fn),l,'ov_fig5.png')
