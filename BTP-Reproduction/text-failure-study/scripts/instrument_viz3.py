import sys
qmod = sys.argv[1]
src = open(qmod + ".BTP_PARAM").read()

# A) globals
anchor = "_BTP_F = float(_os.environ.get('BTP_RETAIN','0.125')) ** (1.0/3.0)\n"
add = anchor + "_BTP_STAGES = []\n_BTP_META = [None]\n"
if "_BTP_STAGES" not in src:
    src = src.replace(anchor, add, 1)

# B) capture grid/img_num/start_idx after img_num set
b_old = "                self.model.img_num = image_embeds.shape[0]\n"
b_new = b_old + ("                try:\n"
                 "                    globals()['_BTP_META'][0] = {'grid': image_grid_thw.detach().cpu().tolist(), 'img_num': int(image_embeds.shape[0]), 'img_start_idx': int(self.model.img_start_idx)}\n"
                 "                except Exception:\n"
                 "                    pass\n")
assert b_old in src
if "'_BTP_META'][0] = {'grid'" not in src:
    src = src.replace(b_old, b_new, 1)

# C) log each stage's sorted indices (all 3 active stages, same pattern)
c_old = "                            indices, _ = torch.sort(indices)\n"
c_new = ("                            indices, _ = torch.sort(indices)\n"
         "                            try:\n"
         "                                if _os.environ.get('BTP_VIZ','0')=='1':\n"
         "                                    globals()['_BTP_STAGES'].append([int(x) for x in indices.tolist()])\n"
         "                            except Exception:\n"
         "                                pass\n")
cnt = src.count(c_old)
src = src.replace(c_old, c_new)  # all occurrences
open(qmod + ".BTP_VIZ3", "w").write(src)
print(f"Wrote .BTP_VIZ3. stage-log inserted at {cnt} sort lines (expect 3), meta+globals ok:",
      "_BTP_STAGES" in src, "'_BTP_META'][0] = {'grid'" in src)
