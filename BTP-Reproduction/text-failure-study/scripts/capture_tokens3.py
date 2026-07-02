import os
os.environ['BTP_VIZ']='1'; os.environ.setdefault('BTP_RETAIN','0.125')
import torch, json, io, numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as M
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from PIL import Image

MID="Qwen/Qwen2.5-VL-7B-Instruct"
proc=AutoProcessor.from_pretrained(MID)
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MID, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map="cuda").eval()

def run_one(img, q):
    M._BTP_STAGES.clear(); M._BTP_META[0]=None
    msg=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":q}]}]
    text=proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    imgs,vids=process_vision_info(msg)
    inp=proc(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        model.generate(**inp, max_new_tokens=6, do_sample=False)
    if M._BTP_META[0] is None or len(M._BTP_STAGES)<3: return None
    s=[np.array(x) for x in M._BTP_STAGES[:3]]
    final = s[0][s[1]][s[2]]          # compose to original-grid indices (12.5%)
    meta=dict(M._BTP_META[0])
    meta["stage1_kept"]=s[0].tolist()
    meta["final_kept"]=[int(x) for x in final.tolist()]
    return meta

PLAN=[
 ("lmms-lab/textvqa","validation","image","What text is shown in the image?","text_heavy",5),
 ("lmms-lab/ChartQA","test","image","What does this chart show?","chart",4),
 ("lmms-lab/ai2d","test","image","Describe this diagram.","diagram",3),
]
out=[]; os.makedirs("results/viz3/images", exist_ok=True)
for ds_name,split,ikey,q,cat,n in PLAN:
    try: ds=load_dataset(ds_name, split=split, streaming=True)
    except Exception as e: print("SKIP",ds_name,e); continue
    got=0
    for ex in ds:
        if got>=n: break
        img=ex.get(ikey)
        if img is None: continue
        if not isinstance(img, Image.Image):
            try: img=Image.open(io.BytesIO(img['bytes'])) if isinstance(img,dict) else img
            except: continue
        img=img.convert("RGB")
        try: rec=run_one(img,q)
        except Exception as e: print("fail",ds_name,got,e); continue
        if not rec: print("no rec",ds_name,got); continue
        tag=f"{cat}_{ds_name.split('/')[-1]}_{got}"
        ip=f"results/viz3/images/{tag}.png"; img.save(ip)
        rec["tag"]=tag; rec["category"]=cat; rec["image_path"]=ip
        out.append(rec); got+=1
        print(f"OK {tag}: img_num={rec['img_num']} stage1={len(rec['stage1_kept'])} final={len(rec['final_kept'])}")
json.dump(out, open("results/viz3/capture3.json","w"), indent=1)
print(f"\nSaved {len(out)} entries. final should be ~1/8 of img_num.")
