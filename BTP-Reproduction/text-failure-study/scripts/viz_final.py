import json, numpy as np, easyocr, os, statistics as st
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

data=json.load(open("results/viz3/capture3.json"))
os.makedirs("results/viz3/overlays", exist_ok=True)
reader=easyocr.Reader(['en'], gpu=False)
rows=[]
for e in data:
    t,h,w=e["grid"][0]; H,W=h//2,w//2; n=e["img_num"]
    kept=set(e["final_kept"])                       # 12.5% retained
    removed=np.array([[0 if (r*W+c) in kept else 1 for c in range(W)] for r in range(H)])
    img=Image.open(e["image_path"]).convert("RGB"); iw,ih=img.size
    # overlay
    m=np.array(Image.fromarray((removed*255).astype('uint8')).resize((iw,ih),Image.NEAREST))/255.0
    fig,ax=plt.subplots(1,2,figsize=(12,5))
    ax[0].imshow(img); ax[0].set_title(f"{e['tag']}\noriginal"); ax[0].axis('off')
    ax[1].imshow(img); red=np.zeros((ih,iw,4)); red[...,0]=1; red[...,3]=m*0.6; ax[1].imshow(red)
    ax[1].set_title(f"REMOVED (red) — {100*removed.sum()/n:.0f}% pruned (final 12.5% kept)"); ax[1].axis('off')
    plt.tight_layout(); plt.savefig(f"results/viz3/overlays/{e['tag']}.png",dpi=90,bbox_inches='tight'); plt.close()
    # OCR text-region overlap
    dets=reader.readtext(np.array(img)); textmask=np.zeros((H,W))
    for box,txt,conf in dets:
        if conf<0.3: continue
        xs=[p[0] for p in box]; ys=[p[1] for p in box]
        for r in range(int(min(ys)/ih*H),min(H,int(max(ys)/ih*H)+1)):
            for c in range(int(min(xs)/iw*W),min(W,int(max(xs)/iw*W)+1)): textmask[r,c]=1
    nt=textmask.sum()
    if nt==0: rows.append((e["tag"],e["category"],0,None,None)); continue
    rt=(removed*textmask).sum()/nt*100
    rn=(removed*(1-textmask)).sum()/max(1,H*W-nt)*100
    rows.append((e["tag"],e["category"],int(nt),round(rt,1),round(rn,1)))
    print(f"{e['tag']:26s} {e['category']:11s} textpatch={int(nt):4d} rem_text={rt:5.1f}% rem_nontext={rn:5.1f}%")
valid=[r for r in rows if r[3] is not None]
print("\n=== SUMMARY (FINAL 12.5% operating point) ===")
print(f"avg % TEXT removed:     {st.mean([r[3] for r in valid]):.1f}%")
print(f"avg % NON-TEXT removed: {st.mean([r[4] for r in valid]):.1f}%")
json.dump([{"tag":r[0],"cat":r[1],"text_patches":r[2],"rem_text_pct":r[3],"rem_nontext_pct":r[4]} for r in rows],
          open("results/viz3/ocr_analysis3.json","w"),indent=1)
print("saved overlays + ocr_analysis3.json")
