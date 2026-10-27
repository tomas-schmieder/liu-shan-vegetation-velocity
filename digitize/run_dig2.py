import sys, json; sys.path.insert(0,'.')
from run_dig import *
# (name, file, frame, xr, yr, kind, shape, sizes, thr)
F6=[(165,1421,21.5,386.5),(165,1421,567.5,932.5),(165,1421,1113,1478)]
F7=[(156.5,1420.5,22.5,390),(156.5,1420.5,572.5,939.5),(156.5,1420.5,1121.5,1488.5)]
J={}
for n,fr in zip(("RN1","RN2","RN3"),F6): J[n]=("fig-001.png",fr,(-6,10),(-0.2,1.2),"red","triangle",[22,24,26],0.5)
for u,fr in zip(("a","b","c"),F7):
    J["ZN_%s_1.6"%u]=("fig-002.png",fr,(-5,20),(-0.2,1.2),"red","circle",[16,19,22,25],0.45)
    J["ZN_%s_8.4"%u]=("fig-002.png",fr,(-5,20),(-0.2,1.2),"blue","square",[16,19,22,25],0.45)
J["Chen"]=("fig-003.png",(129.5,1431.5,13.5,393),(-3,3),(0,1.2),"red","diamond",[22,26,30,34],0.45)
if __name__=="__main__":
    res={}; ov={}
    for name,(fn,fr,xr,yr,kind,shape,sz,thr) in J.items():
        im=load(fn); p=det(im,fr,kind,shape,sz,thr,NMS=0.5 if shape=='triangle' else 0.7)
        X,Y=to_data(p,fr,xr,yr); o=np.argsort(X)
        res[name]=dict(x_over_b=X[o].round(4).tolist(),Ud_over_U0=Y[o].round(4).tolist()); print(name,len(p))
        ov.setdefault(fn,[]).append(((0,0,255) if kind=="red" else (255,0,0),p))
    json.dump(res,open('digitized_fig678.json','w'))
    for fn,l in ov.items(): overlay(load(fn),l,'ov_'+fn)
