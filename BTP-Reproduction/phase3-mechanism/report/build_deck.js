const pptx = require("pptxgenjs");
const fs = require("fs");
const p = new pptx();
p.layout = "LAYOUT_WIDE";                 // 13.3 x 7.5
p.author = "P. S. Kedar";
p.title  = "A Task-Specific Visual Token Access Threshold in Vision-Language Models";

const FIG = "/sessions/modest-epic-cerf/mnt/divya_maam/reports/figs/";
const NAVY="1E2761", ICE="CADCFC", WHITE="FFFFFF", CRIMS="A3312F", TEAL="1D6B6B",
      INK="1A1A1A", MUTE="5A5A5A", TINT="F4F6FA", LINE="D8DEE9", ROSE="FCE9E7", MINT="E9F2EC";
const HEAD="Cambria", BODY="Calibri";
const img = f => ({ data: "image/png;base64," + fs.readFileSync(FIG + f).toString("base64") });

// ---------------------------------------------------------------- helpers
function T(s, text, o) { s.addText(text, Object.assign({ isTextBox: true, fontFace: BODY, color: INK, margin: 0 }, o)); }

function head(s, t, kicker) {
  if (kicker) T(s, kicker.toUpperCase(), { x: 0.7, y: 0.4, w: 11.9, h: 0.3, fontSize: 11, bold: true, color: CRIMS, charSpacing: 2 });
  T(s, t, { x: 0.7, y: 0.72, w: 11.9, h: 0.75, fontFace: HEAD, fontSize: 28, bold: true, color: NAVY });
}

function label(s, text, x, y, colour) {
  T(s, text.toUpperCase(), { x, y, w: 5, h: 0.28, fontSize: 11, bold: true, color: colour, charSpacing: 2 });
}

function card(s, x, y, w, h, fill, lineColour) {
  s.addShape(p.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: lineColour || LINE, width: 1 } });
}

// tested / got / means layout. result() draws inside the left box.
function experiment(s, tested, result, means) {
  label(s, "What we tested", 0.7, 1.6, NAVY);
  T(s, tested, { x: 0.7, y: 1.92, w: 11.9, h: 0.75, fontSize: 16 });
  label(s, "What we got", 0.7, 2.85, TEAL);
  result(s, { x: 0.7, y: 3.2, w: 7.5, h: 3.6 });
  card(s, 8.55, 2.85, 4.05, 3.95, ROSE, CRIMS);
  label(s, "What it tells us", 8.8, 3.05, CRIMS);
  T(s, means, { x: 8.8, y: 3.45, w: 3.6, h: 3.2, fontSize: 16, valign: "top" });
}

// a row of token squares; kept(i) decides filled or hollow
function tokens(s, x, y, n, kept, sz) {
  sz = sz || 0.32;
  for (let i = 0; i < n; i++) {
    const on = kept(i);
    s.addShape(p.ShapeType.rect, { x: x + i * (sz + 0.08), y, w: sz, h: sz,
      fill: { color: on ? NAVY : WHITE }, line: { color: on ? NAVY : "B8BFCC", width: 1, dashType: on ? "solid" : "dash" } });
  }
}

function arrow(s, x, y, w) {
  s.addShape(p.ShapeType.rightArrow, { x, y, w, h: 0.4, fill: { color: "9AA5BF" }, line: { color: "9AA5BF" } });
}

function table(s, rows, o) {
  const hdr = rows[0].map(c => ({ text: c, options: { bold: true, color: WHITE, fill: NAVY } }));
  s.addTable([hdr, ...rows.slice(1)], Object.assign({ fontFace: BODY, fontSize: 14, border: { pt: 1, color: "000000" },
    align: "center", valign: "middle", rowH: 0.45 }, o));
}

// ================================================================ 1 title
let s = p.addSlide(); s.background = { color: NAVY };
T(s, "When does a vision-language model\nstop needing the image?", { x: 0.7, y: 1.9, w: 11.9, h: 1.8, fontFace: HEAD, fontSize: 40, bold: true, color: WHITE });
T(s, "Phase 3: why Balanced Token Pruning breaks text reading on Qwen2.5-VL", { x: 0.7, y: 3.8, w: 11.9, h: 0.6, fontSize: 20, color: ICE });
T(s, "P. S. Kedar   |   Supervised by Prof. Divya Saxena   |   Mentored by Aditya Sharma, PhD Scholar", { x: 0.7, y: 5.5, w: 11.9, h: 0.4, fontSize: 14, color: ICE });
T(s, "IIT Jodhpur", { x: 0.7, y: 5.9, w: 11.9, h: 0.4, fontSize: 13, color: "9FB3D9" });
s.addNotes("Subject paper: Balanced Token Pruning, NeurIPS 2025. Phase 1 reproduced it; this phase explains a failure we found.");

// ================================================================ 2 terms
s = p.addSlide();
head(s, "Three terms we use throughout", "before we start");
const terms = [
  ["Baseline", i => true, "The original model, no pruning. Every visual token is kept.", "It is the reference: every other result is compared against it."],
  ["Pruning", i => i % 2 === 0, "Remove some visual tokens, keep the rest.", "Fewer tokens to process, so the model runs faster."],
  ["Deletion", i => false, "Remove all remaining visual tokens at one layer.", "From that layer onward, the model cannot see the image at all."]
];
terms.forEach((t, k) => {
  const x = 0.7 + k * 4.1;
  card(s, x, 1.7, 3.85, 4.6, k === 2 ? ROSE : TINT, k === 2 ? CRIMS : LINE);
  T(s, t[0], { x: x + 0.3, y: 1.95, w: 3.3, h: 0.5, fontFace: HEAD, fontSize: 22, bold: true, color: k === 2 ? CRIMS : NAVY });
  tokens(s, x + 0.3, 2.65, 8, t[1], 0.33);
  T(s, t[2], { x: x + 0.3, y: 3.3, w: 3.3, h: 1.0, fontSize: 16, bold: true });
  T(s, t[3], { x: x + 0.3, y: 4.35, w: 3.3, h: 1.4, fontSize: 15, color: MUTE });
});
T(s, "Pruning and deletion are not the same thing. That difference is the whole story of this phase.", { x: 0.7, y: 6.5, w: 11.9, h: 0.45, fontSize: 15, italic: true, color: NAVY });
s.addNotes("Filled squares are visual tokens the model can still see. Pruning thins them out; deletion removes whatever is left.");

// ================================================================ 3 observation
s = p.addSlide();
head(s, "Qwen stays fine on general tasks, but stops reading", "1. what we observed");
s.addChart(p.charts.BAR, [
  { name: "Baseline", labels: ["POPE", "MMBench", "GQA", "TextVQA", "DocVQA", "ChartQA"], values: [87.6, 83.7, 60.9, 86.2, 94.7, 76.8] },
  { name: "With BTP", labels: ["POPE", "MMBench", "GQA", "TextVQA", "DocVQA", "ChartQA"], values: [86.2, 79.3, 55.9, 23.3, 19.2, 29.0] }
], { x: 0.7, y: 1.65, w: 8.2, h: 4.9, barDir: "col", barGrouping: "clustered",
  chartColors: [NAVY, CRIMS], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11,
  dataLabelFormatCode: "0.0", showLegend: true, legendPos: "t", legendFontSize: 13,
  valAxisMinVal: 0, valAxisMaxVal: 100, valAxisLabelFontSize: 11, catAxisLabelFontSize: 13,
  valGridLine: { color: "E5E5E5", size: 0.5 }, catGridLine: { style: "none" } });
card(s, 9.2, 1.9, 3.4, 4.4, TINT);
T(s, "General tasks", { x: 9.45, y: 2.1, w: 3.0, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY });
T(s, "Lose only a few points.", { x: 9.45, y: 2.5, w: 3.0, h: 0.6, fontSize: 15 });
T(s, "Reading tasks", { x: 9.45, y: 3.35, w: 3.0, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: CRIMS });
T(s, "Collapse. TextVQA falls from 86.2% to 23.3%.", { x: 9.45, y: 3.75, w: 3.0, h: 0.9, fontSize: 15 });
T(s, "The paper reports only general tasks, so it would not show this.", { x: 9.45, y: 4.85, w: 3.0, h: 1.2, fontSize: 14, italic: true, color: MUTE });
s.addNotes("Qwen2.5-VL-7B. Reading tasks are the ones where the answer is text written inside the image.");

// ================================================================ 4 hypotheses
s = p.addSlide();
head(s, "Four possible explanations", "2. what could be causing it?");
const hyp = [
  ["Wrong tokens chosen", "BTP keeps the wrong visual tokens and throws away the text."],
  ["Text looks redundant", "Text patches look alike, so BTP treats them as duplicates."],
  ["Too much removed", "Simply too many tokens are thrown away for reading to work."],
  ["Text not protected", "If we made sure every text token survived, reading would come back."]
];
hyp.forEach((h, k) => {
  const x = 0.7 + (k % 2) * 6.05, y = 1.75 + Math.floor(k / 2) * 2.35;
  card(s, x, y, 5.85, 2.1, TINT);
  s.addShape(p.ShapeType.ellipse, { x: x + 0.3, y: y + 0.35, w: 0.7, h: 0.7, fill: { color: NAVY }, line: { color: NAVY } });
  T(s, String(k + 1), { x: x + 0.3, y: y + 0.35, w: 0.7, h: 0.7, fontFace: HEAD, fontSize: 22, bold: true, color: WHITE, align: "center", valign: "middle" });
  T(s, h[0], { x: x + 1.25, y: y + 0.3, w: 4.4, h: 0.5, fontFace: HEAD, fontSize: 19, bold: true, color: NAVY });
  T(s, h[1], { x: x + 1.25, y: y + 0.85, w: 4.4, h: 1.0, fontSize: 15 });
});
s.addNotes("Each idea is testable. We designed one experiment per idea.");

// ================================================================ 5 all ruled out
s = p.addSlide();
head(s, "All four were ruled out", "3. what those experiments told us");
table(s, [
  ["Explanation", "What we tested", "What we got", "What it tells us"],
  ["Wrong tokens chosen", "Replaced BTP's token choice with other rules, including random", "All scored 17 to 19% on TextVQA", "Which tokens are kept does not matter here"],
  ["Text looks redundant", "Compared how similar text patches are to background patches", "Text patches were less alike, not more (0.38 vs 0.44)", "Our starting idea was wrong"],
  ["Too much removed", "Gradually reduced the tokens kept, from 100% down to 12.5%", "At 90% kept, TextVQA had already dropped to the same low level as at 12.5%", "The collapse is not just about removing too many"],
  ["Text not protected", "Forced every text token to survive, using the true text locations", "No improvement: 19.7% with, 19.7% without", "Even a perfect choice of tokens does not fix it"]
], { x: 0.7, y: 1.7, w: 11.9, colW: [2.3, 3.4, 3.3, 2.9], fontSize: 13, rowH: 0.9, align: "left" });
T(s, "Four unrelated changes, and the score barely moved. That pointed somewhere else.", { x: 0.7, y: 6.45, w: 11.9, h: 0.45, fontSize: 15, italic: true, color: NAVY });
s.addNotes("All four changed how tokens are chosen or how many are kept. None of them helped, which suggested the loss was happening somewhere else.");

// ================================================================ 6 the redirect
s = p.addSlide();
head(s, "So where is the loss happening?", "4. the experiment that redirected us");
experiment(s,
  "We kept every visual token through all of BTP's pruning stages, so nothing was pruned at all. If pruning were the cause, performance should come back to the baseline.",
  (s, b) => {
    table(s, [["Condition", "TextVQA"], ["Baseline", "87.6%"], ["BTP, but keeping 100% of tokens", { text: "17.8%", options: { bold: true, color: CRIMS } }]],
      { x: b.x, y: b.y + 0.2, w: b.w, colW: [5.0, 2.5], fontSize: 16, rowH: 0.6 });
    T(s, "Nothing pruned, and it still collapsed.", { x: b.x, y: b.y + 2.4, w: b.w, h: 0.5, fontSize: 18, bold: true, color: CRIMS });
  },
  "The damage is not coming from the pruning stages.\n\nSomething after them is removing the image. That is where we looked next.");
s.addNotes("This was meant as a sanity check. It turned out to be the key result, because it ruled out pruning as the main cause.");

// ================================================================ 7 what we found
s = p.addSlide();
head(s, "The code removes every remaining visual token at one layer", "5. what we found");
const stages = [
  ["Image enters", i => true, NAVY],
  ["After BTP's pruning stages", i => i === 0, NAVY],
  ["At one fixed later layer", i => false, CRIMS]
];
stages.forEach((st, k) => {
  const x = 0.7 + k * 4.2;
  card(s, x, 1.8, 3.5, 2.3, k === 2 ? ROSE : TINT, k === 2 ? CRIMS : LINE);
  tokens(s, x + 0.3, 2.2, 8, st[1], 0.3);
  T(s, st[0], { x: x + 0.3, y: 2.75, w: 3.0, h: 0.45, fontFace: HEAD, fontSize: 17, bold: true, color: st[2] });
  T(s, ["All visual tokens", "Only 12.5% remain", "All of them removed"][k], { x: x + 0.3, y: 3.25, w: 3.0, h: 0.6, fontSize: 15 });
  if (k < 2) arrow(s, x + 3.55, 2.75, 0.6);
});
card(s, 0.7, 4.5, 11.9, 1.9, TINT);
T(s, "Pruning keeps 12.5% of the visual tokens. Then, at a later layer, the released code deletes all of them.", { x: 1.0, y: 4.7, w: 11.3, h: 0.6, fontSize: 17, bold: true });
T(s, "For Qwen, the paper describes keeping 12.5% at this last stage. The released implementation removes everything instead. This is a difference between the description and the code we tested; it says nothing about why it is there.", { x: 1.0, y: 5.3, w: 11.3, h: 1.0, fontSize: 15, color: MUTE });
s.addNotes("Plain version: some tokens are removed by pruning, then all the rest are removed. The model finishes answering without any view of the image.");

// ================================================================ 8 intervention
s = p.addSlide();
head(s, "Does that deletion actually cause the collapse?", "6. the intervention");
experiment(s,
  "We kept BTP's 12.5% pruning exactly as it is, and switched off only the final step that deletes the remaining tokens.",
  (s, b) => {
    s.addChart(p.charts.BAR, [{ name: "TextVQA", labels: ["No pruning", "BTP as released", "Deletion switched off"], values: [85.65, 17.25, 78.60] }],
      { x: b.x, y: b.y, w: b.w, h: b.h, barDir: "col", chartColors: [NAVY], showValue: true, dataLabelPosition: "outEnd",
        dataLabelFormatCode: "0.00\"%\"", dataLabelFontSize: 14, showLegend: false, valAxisMinVal: 0, valAxisMaxVal: 100,
        valAxisLabelFontSize: 11, catAxisLabelFontSize: 14, valGridLine: { color: "E5E5E5", size: 0.5 }, catGridLine: { style: "none" } });
  },
  "Switching off that one step brings TextVQA from 17% back to 79%.\n\nThe collapse on Qwen is mainly caused by removing all remaining tokens, not by the earlier pruning.");
s.addNotes("Only one thing changed between the middle and right bars. Pruning is still there in both.");

// ================================================================ 9 is the layer special
s = p.addSlide();
head(s, "Is that layer special, or just too early?", "7. moving the deletion point");
label(s, "What we tested", 0.7, 1.6, NAVY);
T(s, "We moved the point of complete deletion through the model, from early layers to late ones, and measured TextVQA each time. On two models.", { x: 0.7, y: 1.92, w: 11.9, h: 0.7, fontSize: 16 });
label(s, "What we got", 0.7, 2.75, TEAL);
s.addImage(Object.assign({ x: 0.7, y: 3.05, w: 7.6, h: 3.1 }, img("fig7_depth_sweep.png")));
card(s, 8.55, 2.75, 4.05, 3.95, ROSE, CRIMS);
label(s, "What it tells us", 8.8, 2.95, CRIMS);
T(s, "Delete too early and reading collapses. Delete late enough and it barely hurts.\n\nThe change is sudden, not gradual. There is a threshold.\n\nBoth models show the same pattern, at different depths.", { x: 8.8, y: 3.35, w: 3.6, h: 3.3, fontSize: 15, valign: "top" });
s.addNotes("Left panel: by number of layers. Right panel: the same, as a share of the model. Qwen and InternVL are very different models, yet the curve has the same shape.");

// ================================================================ 10 threshold and depth
s = p.addSlide();
head(s, "Threshold and depth, in plain terms", "8. what these numbers mean");
card(s, 0.7, 1.7, 5.85, 2.3, TINT);
T(s, "Threshold", { x: 1.0, y: 1.9, w: 5.3, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: NAVY });
T(s, "How many layers the model needs to see the image for, to keep at least half of its original TextVQA score.", { x: 1.0, y: 2.4, w: 5.3, h: 1.4, fontSize: 15 });
card(s, 6.75, 1.7, 5.85, 2.3, TINT);
T(s, "Depth %", { x: 7.05, y: 1.9, w: 5.3, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: NAVY });
T(s, "The same threshold as a share of the model. How far through the model the image needs to stay visible.", { x: 7.05, y: 2.4, w: 5.3, h: 1.4, fontSize: 15 });

function big(x, model, calc, pct, colour) {
  card(s, x, 4.25, 5.85, 1.75, WHITE, colour);
  T(s, model, { x: x + 0.3, y: 4.4, w: 5.3, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: colour });
  T(s, calc, { x: x + 0.3, y: 4.85, w: 3.4, h: 0.9, fontSize: 15 });
  T(s, pct, { x: x + 3.5, y: 4.55, w: 2.1, h: 1.1, fontFace: HEAD, fontSize: 34, bold: true, color: colour, align: "right" });
}
big(0.7, "Qwen2.5-VL", "22.5 of 28 layers\n22.5 / 28 × 100", "80.3%", NAVY);
big(6.75, "InternVL2", "18.0 of 24 layers\n(17.95 before rounding)", "74.8%", TEAL);
T(s, "Measured on TextVQA, so these are TextVQA-specific. BTP's Qwen setting keeps the image for 22 layers, just short of 22.5.", { x: 0.7, y: 6.3, w: 11.9, h: 0.6, fontSize: 15, italic: true, color: CRIMS });
s.addNotes("Say it as: Qwen needs to see the image through roughly 80% of its layers. BTP's setting stops half a layer short, which is why it lands on the wrong side of the cliff.");

// ================================================================ 11 two costs defined
s = p.addSlide();
head(s, "Two separate costs, worked through on TextVQA", "9. what actually causes the loss?");
T(s, "Corrected BTP = BTP with only the final deletion switched off. It still prunes to 12.5%.", { x: 0.7, y: 1.6, w: 11.9, h: 0.45, fontSize: 15, italic: true, color: MUTE });
// staircase of three bars
const base = 5.95, scale = 0.036;
const bars = [["Baseline", 86.2, NAVY], ["Corrected BTP", 80.5, TEAL], ["BTP as released", 23.3, CRIMS]];
bars.forEach((b, k) => {
  const h = b[1] * scale, x = 0.9 + k * 2.3;
  s.addShape(p.ShapeType.rect, { x, y: base - h, w: 1.5, h, fill: { color: b[2] }, line: { color: b[2] } });
  T(s, b[1].toFixed(1) + "%", { x, y: base - h - 0.4, w: 1.5, h: 0.35, fontSize: 15, bold: true, align: "center" });
  T(s, b[0], { x: x - 0.2, y: base + 0.1, w: 1.9, h: 0.5, fontSize: 13, align: "center" });
});
card(s, 7.8, 2.2, 4.8, 1.75, TINT);
T(s, "Pruning cost", { x: 8.05, y: 2.35, w: 4.3, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: TEAL });
T(s, "baseline − corrected BTP\n86.2 − 80.5 = 5.7 points", { x: 8.05, y: 2.8, w: 4.3, h: 1.0, fontSize: 15 });
card(s, 7.8, 4.15, 4.8, 1.75, ROSE, CRIMS);
T(s, "Deletion cost", { x: 8.05, y: 4.3, w: 4.3, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: CRIMS });
T(s, "corrected BTP − released BTP\n80.5 − 23.3 = 57.2 points", { x: 8.05, y: 4.75, w: 4.3, h: 1.0, fontSize: 15 });
T(s, "On TextVQA, almost all of the loss comes from deletion.", { x: 0.7, y: 6.55, w: 11.9, h: 0.45, fontSize: 16, bold: true, color: NAVY });
s.addNotes("Pruning cost: what we lose by removing some tokens. Deletion cost: the extra we lose by then removing all of them.");

// ================================================================ 12 across tasks
s = p.addSlide();
head(s, "Which cost dominates depends on the task", "10. does this hold across tasks?");
table(s, [
  ["Task", "Pruning cost", "Deletion cost", "Dominant"],
  ["TextVQA", "5.7", "57.2", "deletion"],
  ["DocVQA", "17.9", "57.6", "deletion"],
  [{ text: "ChartQA", options: { bold: true, fill: ROSE } }, { text: "31.4", options: { bold: true, color: CRIMS, fill: ROSE } },
   { text: "16.4", options: { fill: ROSE } }, { text: "pruning", options: { bold: true, color: CRIMS, fill: ROSE } }],
  ["AI2D", "7.2", "0.4", "pruning"],
  ["GQA", "2.1", "2.9", "both small"],
  ["POPE", "1.3", "0.1", "both small"],
  ["MMBench", "4.7", "0.3", "both small"]
], { x: 0.7, y: 1.7, w: 7.5, colW: [2.1, 1.8, 1.8, 1.8], rowH: 0.52 });
card(s, 8.55, 1.7, 4.05, 4.2, ROSE, CRIMS);
label(s, "What it tells us", 8.8, 1.9, CRIMS);
T(s, "Dominant = whichever caused the bigger share of the drop.\n\nOn TextVQA and DocVQA, deletion does most of the damage.\n\nOn ChartQA, pruning itself costs twice as much as deletion.", { x: 8.8, y: 2.3, w: 3.6, h: 3.5, fontSize: 15, valign: "top" });
T(s, "So the finding is not simply \"deletion is bad\". The main source of loss depends on the task.", { x: 0.7, y: 6.3, w: 11.9, h: 0.5, fontSize: 16, bold: true, color: NAVY });
s.addNotes("Points lost on each benchmark, Qwen2.5-VL. Charts need fine visual detail, so removing tokens hurts even before deletion.");

// ================================================================ 13 why paper missed it
s = p.addSlide(); s.background = { color: NAVY };
T(s, "11. WHY THE PAPER DID NOT SHOW THIS", { x: 0.7, y: 0.4, w: 11.9, h: 0.3, fontSize: 11, bold: true, color: ICE, charSpacing: 2 });
T(s, "Its benchmarks cannot see the reading loss", { x: 0.7, y: 0.72, w: 11.9, h: 0.75, fontFace: HEAD, fontSize: 28, bold: true, color: WHITE });
s.addTable([
  [{ text: "Share of baseline kept", options: { bold: true, color: NAVY, fill: ICE } },
   { text: "Paper's benchmarks", options: { bold: true, color: NAVY, fill: ICE } },
   { text: "Reading tasks", options: { bold: true, color: NAVY, fill: ICE } }],
  [{ text: "Qwen, BTP as released", options: { color: WHITE, fill: "2A3A7A" } }, { text: "96.1%", options: { color: WHITE, fill: "2A3A7A", bold: true } }, { text: "28.4%", options: { color: "FF9B9B", fill: "2A3A7A", bold: true } }],
  [{ text: "Qwen, deletion switched off", options: { color: WHITE, fill: "2A3A7A" } }, { text: "96.9%", options: { color: WHITE, fill: "2A3A7A", bold: true } }, { text: "77.9%", options: { color: "9FE8C8", fill: "2A3A7A", bold: true } }],
  [{ text: "InternVL, pruning only", options: { color: WHITE, fill: "2A3A7A" } }, { text: "94.9%", options: { color: WHITE, fill: "2A3A7A", bold: true } }, { text: "56.5%", options: { color: "FFC48C", fill: "2A3A7A", bold: true } }]
], { x: 0.7, y: 1.8, w: 7.6, colW: [3.2, 2.2, 2.2], fontFace: BODY, fontSize: 15, border: { pt: 1, color: "4A5A9A" }, align: "center", valign: "middle", rowH: 0.6 });
T(s, "On the paper's benchmarks, every row looks healthy.\n\nOn reading, the same configurations lose between a quarter and three quarters of their ability.\n\nInternVL has no deletion step at all, and the gap is still there.", { x: 8.65, y: 1.8, w: 4.0, h: 3.6, fontSize: 15, color: WHITE, valign: "top" });
T(s, "A benchmark set with no reading task cannot tell a model that reads from one that does not.", { x: 0.7, y: 4.9, w: 11.9, h: 0.8, fontFace: HEAD, fontSize: 21, italic: true, color: ICE });
s.addNotes("This is the finding that does not depend on the Qwen code at all. The InternVL row has no deletion step.");

// ================================================================ 14 takeaway
s = p.addSlide();
head(s, "Three separate findings", "12. final takeaway");
[
  ["Implementation", "In the Qwen configuration we tested, removing all remaining visual tokens causes most of the reading collapse.", NAVY],
  ["Model", "Each model needs to see the image up to a measurable depth: about 80% of Qwen, 75% of InternVL, on TextVQA.", TEAL],
  ["Evaluation", "Standard benchmarks can hide large losses on reading tasks, with or without the deletion step.", CRIMS]
].forEach((r, k) => {
  const y = 1.75 + k * 1.6;
  card(s, 0.7, y, 11.9, 1.4, TINT);
  s.addShape(p.ShapeType.ellipse, { x: 1.0, y: y + 0.35, w: 0.7, h: 0.7, fill: { color: r[2] }, line: { color: r[2] } });
  T(s, String(k + 1), { x: 1.0, y: y + 0.35, w: 0.7, h: 0.7, fontFace: HEAD, fontSize: 22, bold: true, color: WHITE, align: "center", valign: "middle" });
  T(s, r[0], { x: 2.0, y: y + 0.2, w: 2.6, h: 1.0, fontFace: HEAD, fontSize: 20, bold: true, color: r[2], valign: "middle" });
  T(s, r[1], { x: 4.6, y: y + 0.2, w: 7.7, h: 1.0, fontSize: 16, valign: "middle" });
});
T(s, "These stand separately. The third holds even if the first did not.", { x: 0.7, y: 6.6, w: 11.9, h: 0.45, fontSize: 15, italic: true, color: MUTE });
s.addNotes("Keep them apart when discussing. The evaluation finding is the one most useful beyond this paper.");

// ================================================================ 15 limits and next
s = p.addSlide();
head(s, "Limits, and what comes next", "scope");
card(s, 0.7, 1.7, 5.85, 2.85, TINT);
T(s, "What we would not claim", { x: 1.0, y: 1.9, w: 5.3, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: CRIMS });
s.addText([
  "Two models only. The shared pattern is worth testing further, not a rule",
  "Thresholds come from TextVQA. Other tasks may differ",
  "The InternVL pruning is our own, simplified version",
  "Results use 500-question subsets per task"
].map((t, i, a) => ({ text: t, options: { bullet: true, breakLine: i < a.length - 1 } })),
  { x: 1.0, y: 2.45, w: 5.3, h: 3.3, isTextBox: true, fontFace: BODY, fontSize: 15, color: INK, paraSpaceAfter: 10, valign: "top" });
card(s, 6.75, 1.7, 5.85, 2.85, MINT, TEAL);
T(s, "Next", { x: 7.05, y: 1.9, w: 5.3, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: TEAL });
s.addText([
  "Measure the threshold on ChartQA and DocVQA, ChartQA first",
  "Repeat on a third model",
  "Prepare the manuscript"
].map((t, i, a) => ({ text: t, options: { bullet: true, breakLine: i < a.length - 1 } })),
  { x: 7.05, y: 2.45, w: 5.3, h: 3.3, isTextBox: true, fontFace: BODY, fontSize: 15, color: INK, paraSpaceAfter: 10, valign: "top" });
s.addNotes("ChartQA first because it is the task where pruning, not deletion, is the bigger cost.");

p.writeFile({ fileName: "/sessions/modest-epic-cerf/work/deck/BTP_Phase3_Deck.pptx" }).then(f => console.log("WROTE", f));
