const pptx = require("pptxgenjs");
const fs = require("fs");
const p = new pptx();
p.layout = "LAYOUT_WIDE";                       // 13.3 x 7.5
p.author = "P. S. Kedar";
p.title  = "A Task-Specific Visual Token Access Threshold in Vision-Language Models";

const FIG   = "/sessions/modest-epic-cerf/mnt/divya_maam/reports/figs/";
const NAVY  = "1E2761", ICE = "CADCFC", WHITE = "FFFFFF";
const CRIMS = "A3312F", TEAL = "1D6B6B", INK = "1A1A1A", MUTE = "5A5A5A";
const HEAD = "Cambria", BODY = "Calibri";

const img = f => ({ data: "image/png;base64," + fs.readFileSync(FIG + f).toString("base64") });

function dark(s) { s.background = { color: NAVY }; }

function title(s, t, sub) {
  s.addText(t, { x: 0.7, y: 2.3, w: 11.9, h: 1.5, isTextBox: true,
                 fontFace: HEAD, fontSize: 40, bold: true, color: WHITE });
  if (sub) s.addText(sub, { x: 0.7, y: 3.8, w: 11.9, h: 0.9, isTextBox: true,
                            fontFace: BODY, fontSize: 18, color: ICE });
}

function head(s, t, kicker) {
  if (kicker) s.addText(kicker.toUpperCase(), { x: 0.7, y: 0.42, w: 11.9, h: 0.3,
    isTextBox: true, fontFace: BODY, fontSize: 11, bold: true, color: CRIMS, charSpacing: 2 });
  s.addText(t, { x: 0.7, y: kicker ? 0.75 : 0.6, w: 11.9, h: 0.8, isTextBox: true,
                 fontFace: HEAD, fontSize: 30, bold: true, color: NAVY });
}

function bullets(s, items, o = {}) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
    { x: o.x || 0.75, y: o.y || 1.75, w: o.w || 6.0, h: o.h || 4.4, isTextBox: true,
      fontFace: BODY, fontSize: o.size || 15, color: INK, paraSpaceAfter: 10, lineSpacing: 22 });
}

function stat(s, x, y, w, big, label, colour) {
  s.addShape(p.ShapeType.roundRect, { x, y, w, h: 1.65, rectRadius: 0.08,
    fill: { color: "F4F6FA" }, line: { color: "D8DEE9", width: 1 } });
  s.addText(big, { x, y: y + 0.18, w, h: 0.8, isTextBox: true, align: "center",
    fontFace: HEAD, fontSize: 34, bold: true, color: colour || NAVY });
  s.addText(label, { x: x + 0.15, y: y + 1.0, w: w - 0.3, h: 0.55, isTextBox: true, align: "center",
    fontFace: BODY, fontSize: 11, color: MUTE });
}

// ---------------------------------------------------------------- 1 title
let s = p.addSlide(); dark(s);
title(s, "A task-specific visual token access threshold",
         "What visual token pruning removes, and when it is safe to remove it");
s.addText("P. S. Kedar   |   Supervised by Prof. Divya Saxena   |   Mentored by Aditya Sharma, PhD Scholar",
  { x: 0.7, y: 5.4, w: 11.9, h: 0.4, isTextBox: true, fontFace: BODY, fontSize: 13, color: ICE });
s.addText("IIT Jodhpur   |   Phase 3   |   September 2026",
  { x: 0.7, y: 5.8, w: 11.9, h: 0.4, isTextBox: true, fontFace: BODY, fontSize: 12, color: "9FB3D9" });
s.addNotes("Phase 3 of the BTP reproduction. Subject paper is Balanced Token Pruning, NeurIPS 2025. Everything here is measured; the report has the full record.");

// ---------------------------------------------------------------- 2 problem
s = p.addSlide();
head(s, "Qwen stopped reading", "the observation");
bullets(s, [
  "BTP reproduces cleanly on LLaVA-1.5-7B, LLaVA-1.5-13B and Qwen2.5-VL-7B",
  "But on tasks that need the model to read text inside an image, Qwen collapses",
  "AI2D has text and rarely needs it read word by word. It barely moves, so it is our control",
  "The paper reports no reading task, so nothing in its results would show this"
], { w: 5.6 });
s.addTable(
  [[{ text: "Benchmark", options: { bold: true, color: WHITE, fill: NAVY } },
    { text: "Baseline", options: { bold: true, color: WHITE, fill: NAVY } },
    { text: "BTP", options: { bold: true, color: WHITE, fill: NAVY } }],
   ["TextVQA", "86.2", { text: "23.3", options: { color: CRIMS, bold: true } }],
   ["DocVQA", "94.7", { text: "19.2", options: { color: CRIMS, bold: true } }],
   ["ChartQA", "76.8", { text: "29.0", options: { color: CRIMS, bold: true } }],
   ["AI2D (control)", "86.4", "79.6"]],
  { x: 6.8, y: 2.0, w: 5.8, colW: [2.6, 1.6, 1.6], fontFace: BODY, fontSize: 13,
    border: { pt: 1, color: "000000" }, align: "center", valign: "middle", rowH: 0.42 });
s.addText("Qwen2.5-VL-7B, limit 500", { x: 6.8, y: 4.45, w: 5.8, h: 0.3, isTextBox: true,
  fontFace: BODY, fontSize: 10, italic: true, color: MUTE, align: "center" });
s.addNotes("Phase 1 reproduced the paper. Phase 2 added text-heavy benchmarks the paper does not report, and found the collapse. AI2D is the control throughout.");

// ---------------------------------------------------------------- 3 eliminations
s = p.addSlide();
head(s, "Four explanations, all eliminated", "what it was not");
s.addTable(
  [[{ text: "Hypothesis", options: { bold: true, color: WHITE, fill: NAVY } },
    { text: "Test", options: { bold: true, color: WHITE, fill: NAVY } },
    { text: "Result", options: { bold: true, color: WHITE, fill: NAVY } }],
   ["Selector picks badly", "Diversity, attention, random at one budget", "17.3 to 19.4, inside the error bars"],
   ["Text patches look alike", "Feature similarity at layer 4", "Text 0.379, background 0.438"],
   ["Too much is removed", "Retention swept 100 to 12.5 per cent", "90 per cent already gives the full drop"],
   ["Wrong tokens kept", "Ground-truth OCR oracle", "0.197 oracle, 0.197 control"]],
  { x: 0.7, y: 1.7, w: 11.9, colW: [2.9, 4.4, 4.6], fontFace: BODY, fontSize: 13,
    border: { pt: 1, color: "000000" }, valign: "middle", rowH: 0.52 });
s.addShape(p.ShapeType.roundRect, { x: 0.7, y: 4.55, w: 11.9, h: 1.5, rectRadius: 0.08,
  fill: { color: "FCE9E7" }, line: { color: CRIMS, width: 1 } });
s.addText("Four unrelated interventions, one number. The control that broke it open: at 100 per cent retention, where nothing is pruned at all, the score is still 0.178. The loss was happening after the pruning stages.",
  { x: 1.0, y: 4.75, w: 11.3, h: 1.1, isTextBox: true, fontFace: BODY, fontSize: 15, color: INK });
s.addNotes("Each experiment changed something upstream and measured an answer produced downstream. All four were measuring the same floor. The 100 per cent retention control is what redirected the investigation.");

// ---------------------------------------------------------------- 4 the code
s = p.addSlide();
head(s, "One branch, no image index", "the cause");
s.addShape(p.ShapeType.roundRect, { x: 0.7, y: 1.7, w: 7.3, h: 2.5, rectRadius: 0.06,
  fill: { color: "FCE9E7" }, line: { color: CRIMS, width: 1 } });
s.addText([
  { text: "elif layer_index == self.end_layer:   # 23\n", options: { breakLine: true } },
  { text: "    remain_sys_index  = ...\n", options: { breakLine: true } },
  { text: "    remain_text_index = ...\n", options: { breakLine: true } },
  { text: "    # no remain_img_index below\n", options: { breakLine: true, color: CRIMS } },
  { text: "    all_remained_index = torch.cat(\n", options: { breakLine: true } },
  { text: "        (remain_sys_index, remain_text_index))", options: {} }
], { x: 0.95, y: 1.9, w: 6.9, h: 2.1, isTextBox: true, fontFace: "Courier New", fontSize: 12, color: INK });
bullets(s, [
  "Every image token surviving three pruning stages is removed here, unconditionally",
  "Appendix 7.3 specifies this for LLaVA only. For Qwen it asks for 12.5 per cent retention",
  "Code at commit 9682db0, file modeling_qwen2_5_vl.py, branch at line 1385"
], { x: 8.3, y: 1.75, w: 4.3, size: 14, h: 2.6 });
s.addTable(
  [[{ text: "Arm", options: { bold: true, color: WHITE, fill: NAVY } },
    { text: "TextVQA", options: { bold: true, color: WHITE, fill: NAVY } },
    { text: "AI2D", options: { bold: true, color: WHITE, fill: NAVY } }],
   ["No pruning", "0.8565", "0.8750"],
   ["BTP as released", { text: "0.1725", options: { color: CRIMS, bold: true } }, "0.8150"],
   ["Deletion disabled", { text: "0.7860", options: { color: TEAL, bold: true } }, "0.8100"]],
  { x: 0.7, y: 4.5, w: 7.3, colW: [3.1, 2.1, 2.1], fontFace: BODY, fontSize: 13,
    border: { pt: 1, color: "000000" }, align: "center", valign: "middle", rowH: 0.42 });
s.addText("One condition changed, the 12.5 per cent pruning left intact. Job 416735, 200 samples.",
  { x: 8.3, y: 4.6, w: 4.3, h: 1.2, isTextBox: true, fontFace: BODY, fontSize: 13, color: MUTE });
s.addNotes("Code inspection is not evidence of cause. The intervention is: one condition disabled, everything else identical. AI2D unchanged shows this is not a lever that lifts every score.");

// ---------------------------------------------------------------- 5 threshold
s = p.addSlide();
head(s, "Is layer 23 special, or just early?", "the threshold");
s.addImage(Object.assign({ x: 0.75, y: 1.6, w: 8.4, h: 3.44 }, img("fig7_depth_sweep.png")));
bullets(s, [
  "We moved the deletion across the decoder, graded pruning off",
  "Both models sit on a floor, then climb sharply",
  "Threshold: the half-of-baseline crossing, interpolated",
  "Measured on TextVQA, so it is task-specific"
], { x: 9.4, y: 1.7, w: 3.3, size: 13, h: 3.3 });
s.addTable(
  [[{ text: "Model", options: { bold: true, color: WHITE, fill: NAVY } },
    { text: "Layers", options: { bold: true, color: WHITE, fill: NAVY } },
    { text: "Threshold", options: { bold: true, color: WHITE, fill: NAVY } },
    { text: "Depth", options: { bold: true, color: WHITE, fill: NAVY } }],
   ["InternVL2-2B", "24", "18.0", { text: "74.8 %", options: { bold: true, color: TEAL } }],
   ["Qwen2.5-VL-7B", "28", "22.5", { text: "80.3 %", options: { bold: true, color: NAVY } }]],
  { x: 0.75, y: 5.25, w: 8.4, colW: [3.0, 1.6, 1.9, 1.9], fontFace: BODY, fontSize: 13,
    border: { pt: 1, color: "000000" }, align: "center", valign: "middle", rowH: 0.42 });
s.addNotes("Same shape in both models. With two models this is a pattern worth testing further, not a general property. The depths differ, so the value cannot be read off the layer count.");

// ---------------------------------------------------------------- 6 one layer
s = p.addSlide();
head(s, "The released constant lands one layer short", "what it costs");
s.addImage(Object.assign({ x: 1.3, y: 1.75, w: 5.6, h: 3.65 }, img("fig8_one_layer.png")));
stat(s, 7.5, 1.9, 2.4, "22", "layers BTP allows Qwen", CRIMS);
stat(s, 10.1, 1.9, 2.4, "22.5", "measured threshold", NAVY);
stat(s, 7.5, 3.75, 5.0, "47.7 points", "TextVQA retention, one layer either side of the threshold", TEAL);
s.addText("BTP does calibrate. Section 4.3 selects the three pruning layers from 64 samples. The depth at which complete deletion becomes safe is a different quantity, and does not appear to be calibrated.",
  { x: 1.3, y: 5.55, w: 11.2, h: 0.9, isTextBox: true, fontFace: BODY, fontSize: 13,
    italic: true, color: MUTE });
s.addNotes("Important nuance for questions: BTP is not uncalibrated. It calibrates where to prune. What is not calibrated is where complete removal stops being harmful.");

// ---------------------------------------------------------------- 7 blind spot
s = p.addSlide(); dark(s);
s.addText("THE FINDING THAT MATTERS", { x: 0.75, y: 0.6, w: 11.8, h: 0.35, isTextBox: true,
  fontFace: BODY, fontSize: 12, bold: true, color: ICE, charSpacing: 2 });
s.addText("None of this is visible on the paper's benchmarks", { x: 0.75, y: 1.0, w: 11.8, h: 0.8,
  isTextBox: true, fontFace: HEAD, fontSize: 30, bold: true, color: WHITE });
s.addTable(
  [[{ text: "", options: { fill: NAVY } },
    { text: "Paper's suite", options: { bold: true, color: NAVY, fill: ICE } },
    { text: "Reading tasks", options: { bold: true, color: NAVY, fill: ICE } }],
   [{ text: "BTP as released", options: { color: WHITE, fill: "2A3A7A" } },
    { text: "96.1 %", options: { color: WHITE, fill: "2A3A7A", bold: true } },
    { text: "28.4 %", options: { color: "FF9B9B", fill: "2A3A7A", bold: true } }],
   [{ text: "BTP as specified", options: { color: WHITE, fill: "2A3A7A" } },
    { text: "96.9 %", options: { color: WHITE, fill: "2A3A7A", bold: true } },
    { text: "77.9 %", options: { color: "9FE8C8", fill: "2A3A7A", bold: true } }],
   [{ text: "InternVL, no discrepancy", options: { color: WHITE, fill: "2A3A7A" } },
    { text: "94.9 %", options: { color: WHITE, fill: "2A3A7A", bold: true } },
    { text: "56.5 %", options: { color: "FFC48C", fill: "2A3A7A", bold: true } }]],
  { x: 0.75, y: 2.1, w: 7.4, colW: [3.0, 2.2, 2.2], fontFace: BODY, fontSize: 14,
    border: { pt: 1, color: "4A5A9A" }, align: "center", valign: "middle", rowH: 0.5 });
s.addText("Percentage of each model's own unpruned baseline. InternVL row is graded pruning only, with no deletion at all.",
  { x: 0.75, y: 4.35, w: 7.4, h: 0.6, isTextBox: true, fontFace: BODY, fontSize: 11,
    italic: true, color: "9FB3D9" });
s.addText([
  { text: "Same code, two verdicts.\n", options: { bold: true, breakLine: true } },
  { text: "The paper claims 96 to 98 per cent. The released code, discrepancy included, scores 96.1 on its own suite.\n\n", options: { breakLine: true } },
  { text: "And on InternVL2-2B, where we found no such discrepancy, correct pruning still loses half the reading ability while the same suite reports 95 per cent.", options: {} }
], { x: 8.5, y: 2.1, w: 4.1, h: 3.2, isTextBox: true, fontFace: BODY, fontSize: 14, color: WHITE });
s.addText("The blind spot is in the evaluation, not only in one implementation.",
  { x: 0.75, y: 5.15, w: 11.8, h: 0.8, isTextBox: true, fontFace: HEAD, fontSize: 21,
    italic: true, color: ICE });
s.addNotes("This is the slide to spend time on. The evaluation finding does not depend on the implementation finding being true. Even a correct pruning implementation is invisible to this suite.");

// ---------------------------------------------------------------- 8 two costs
s = p.addSlide();
head(s, "Two costs, and they are not the same", "separating the effects");
s.addTable(
  [[{ text: "Benchmark", options: { bold: true, color: WHITE, fill: NAVY } },
    { text: "Cost of pruning", options: { bold: true, color: WHITE, fill: NAVY } },
    { text: "Cost of deletion", options: { bold: true, color: WHITE, fill: NAVY } },
    { text: "Dominant", options: { bold: true, color: WHITE, fill: NAVY } }],
   ["TextVQA", "5.7", "57.2", "deletion"],
   ["DocVQA", "17.9", "57.6", "deletion"],
   [{ text: "ChartQA", options: { bold: true } },
    { text: "31.4", options: { bold: true, color: CRIMS } },
    { text: "16.4", options: { bold: true } },
    { text: "pruning", options: { bold: true, color: CRIMS } }],
   ["AI2D (control)", "7.2", "0.4", "pruning"],
   ["GQA", "2.1", "2.9", "both small"],
   ["POPE", "1.3", "0.1", "both small"],
   ["MMBench-EN", "4.7", "0.3", "both small"]],
  { x: 0.7, y: 1.7, w: 7.6, colW: [2.4, 1.9, 1.9, 1.4], fontFace: BODY, fontSize: 12,
    border: { pt: 1, color: "000000" }, align: "center", valign: "middle", rowH: 0.4 });
s.addShape(p.ShapeType.roundRect, { x: 8.6, y: 1.7, w: 4.0, h: 3.2, rectRadius: 0.08,
  fill: { color: "FCE9E7" }, line: { color: CRIMS, width: 1 } });
s.addText([
  { text: "ChartQA reverses\n\n", options: { bold: true, breakLine: true, fontSize: 17 } },
  { text: "Pruning costs twice what the deletion does there, so the discrepancy is not the dominant problem on every reading task.\n\n", options: { breakLine: true } },
  { text: "An earlier version of our analysis treated deletion as the main cause throughout. These figures do not support that.", options: {} }
], { x: 8.85, y: 1.95, w: 3.5, h: 2.7, isTextBox: true, fontFace: BODY, fontSize: 13, color: INK });
s.addText("Baseline minus corrected is the pruning cost. Corrected minus released is the deletion cost. Qwen2.5-VL-7B.",
  { x: 0.7, y: 5.25, w: 11.9, h: 0.4, isTextBox: true, fontFace: BODY, fontSize: 12,
    italic: true, color: MUTE });
s.addNotes("Keeping these apart is what stops the overclaim. If asked which matters more, the answer is: it depends on the benchmark.");

// ---------------------------------------------------------------- 9 three levels
s = p.addSlide();
head(s, "Three findings, at three levels", "conclusion");
const rows = [
  ["Implementation", "The released code deletes all remaining image tokens just below the threshold we measured for Qwen, while the paper's appendix asks for retention.", "Commit 9682db0. Says nothing about author intent, or about Qwen as a model."],
  ["Model", "Visual token access has a measurable, abrupt depth threshold. 80.3 per cent in Qwen, 74.8 in InternVL.", "Two models, measured on TextVQA. Task-specific."],
  ["Evaluation", "The benchmark suite did not reveal the degradation in reading, with the discrepancy present or absent.", "The one finding that does not depend on the other two."]
];
let y = 1.7;
rows.forEach(r => {
  s.addShape(p.ShapeType.roundRect, { x: 0.7, y, w: 11.9, h: 1.35, rectRadius: 0.06,
    fill: { color: "F4F6FA" }, line: { color: "D8DEE9", width: 1 } });
  s.addText(r[0], { x: 0.95, y: y + 0.12, w: 2.3, h: 0.5, isTextBox: true,
    fontFace: HEAD, fontSize: 17, bold: true, color: NAVY });
  s.addText(r[1], { x: 3.3, y: y + 0.12, w: 5.6, h: 1.1, isTextBox: true,
    fontFace: BODY, fontSize: 13, color: INK });
  s.addText(r[2], { x: 9.1, y: y + 0.12, w: 3.3, h: 1.1, isTextBox: true,
    fontFace: BODY, fontSize: 12, italic: true, color: MUTE });
  y += 1.5;
});
s.addNotes("They stand or fall separately and should not be quoted as one result.");

// ---------------------------------------------------------------- 10 limits and next
s = p.addSlide();
head(s, "What we would not claim, and what is next", "limits");
s.addText("Limitations", { x: 0.75, y: 1.6, w: 5.6, h: 0.4, isTextBox: true,
  fontFace: HEAD, fontSize: 18, bold: true, color: CRIMS });
bullets(s, [
  "Two architectures. The matching shape is a pattern, not a law",
  "Thresholds measured on TextVQA only. ChartQA may differ",
  "The InternVL pruning is ours, diversity selector only, layers transplanted not calibrated",
  "The OCR oracle ran at one budget",
  "500 sample subsets, matched across arms"
], { x: 0.75, y: 2.05, w: 5.7, size: 13, h: 3.0 });
s.addText("Claims withdrawn", { x: 6.9, y: 1.6, w: 5.7, h: 0.4, isTextBox: true,
  fontFace: HEAD, fontSize: 18, bold: true, color: "8A5A00" });
bullets(s, [
  "Text has no redundancy. Our own measurement says the opposite, 0.379 against 0.438",
  "Text coverage predicts correctness. Vanished at a larger sample",
  "The effect is Qwen-specific. The sweep showed both models have a threshold",
  "Qwen's threshold is at 83.9 per cent. A counting error; it is 80.3"
], { x: 6.9, y: 2.05, w: 5.7, size: 13, h: 3.0 });
s.addShape(p.ShapeType.roundRect, { x: 0.75, y: 5.25, w: 11.85, h: 1.1, rectRadius: 0.06,
  fill: { color: "E9F2EC" }, line: { color: TEAL, width: 1 } });
s.addText("Next: thresholds on DocVQA and ChartQA, ChartQA first. Then a third architecture, then the manuscript.",
  { x: 1.05, y: 5.5, w: 11.2, h: 0.6, isTextBox: true, fontFace: BODY, fontSize: 15, color: INK });
s.addNotes("Reporting the withdrawn claims is deliberate. Four review rounds caught things the experiments did not.");

// ---------------------------------------------------------------- 11 close
s = p.addSlide(); dark(s);
s.addText("Measure the depth. Test the reading.", { x: 0.75, y: 2.2, w: 11.8, h: 1.0,
  isTextBox: true, fontFace: HEAD, fontSize: 34, bold: true, color: WHITE });
s.addText([
  { text: "A pruning method has to choose a depth at which visual tokens stop being needed. That depth is measurable, it differs by architecture, and the sweep is cheap: about seven evaluations, no training, no calibration set.\n\n", options: { breakLine: true } },
  { text: "And include one reading benchmark. Not because reading matters more than reasoning, but because it degrades first when visual evidence is removed.", options: {} }
], { x: 0.75, y: 3.4, w: 8.6, h: 2.0, isTextBox: true, fontFace: BODY, fontSize: 16, color: ICE });
s.addText("Report, results log and code:\ngithub.com/milab-iitj-dev/btp-reproduction",
  { x: 0.75, y: 5.9, w: 11.8, h: 0.7, isTextBox: true, fontFace: BODY, fontSize: 13, color: "9FB3D9" });
s.addNotes("Close on the recommendation, not the bug. The tool is written and validated on one model.");

p.writeFile({ fileName: "/sessions/modest-epic-cerf/work/deck/BTP_Phase3_Deck.pptx" })
 .then(f => console.log("WROTE", f));
