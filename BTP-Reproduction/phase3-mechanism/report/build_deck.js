const pptx = require("pptxgenjs");
const fs = require("fs");
const p = new pptx();
p.layout = "LAYOUT_WIDE";
p.author = "P. S. Kedar";
p.title = "Visual Token Pruning and Text-Reading Performance in Vision-Language Models";

const FIG = "" + require("path").join(__dirname, "..", "figures") + "/";
// one primary, one accent, neutrals
const NAVY = "1E2761", ACC = "A3312F", INK = "1A1A1A", MUTE = "5A5A5A",
      FILL = "F4F5F8", LINE = "C9CED8", ACCFILL = "FBEDEC", WHITE = "FFFFFF";
const HEAD = "Cambria", BODY = "Calibri";
const img = f => ({ data: "image/png;base64," + fs.readFileSync(FIG + f).toString("base64") });

function T(s, t, o) { s.addText(t, Object.assign({ isTextBox: true, fontFace: BODY, fontSize: 14, color: INK, margin: 0, valign: "top" }, o)); }
function box(s, x, y, w, h, accent) {
  s.addShape(p.ShapeType.rect, { x, y, w, h, fill: { color: accent ? ACCFILL : FILL }, line: { color: accent ? ACC : LINE, width: 1 } });
}
function H(s, title, q) {
  T(s, title, { x: 0.6, y: 0.4, w: 12.1, h: 0.65, fontFace: HEAD, fontSize: 28, bold: true, color: NAVY, valign: "middle" });
  if (q) T(s, q, { x: 0.6, y: 1.02, w: 12.1, h: 0.4, fontSize: 15, italic: true, color: MUTE });
}
function table(s, rows, o) {
  const hdr = rows[0].map(c => ({ text: c, options: { bold: true, color: WHITE, fill: NAVY } }));
  s.addTable([hdr, ...rows.slice(1)], Object.assign({ fontFace: BODY, fontSize: 14, color: INK, border: { pt: 1, color: "000000" },
    align: "center", valign: "middle", rowH: 0.44 }, o));
}
// labelled line: "Why:" text
function L(label, text, colour) {
  return [{ text: label + "  ", options: { bold: true, color: colour || NAVY } }, { text: text, options: { breakLine: true } }];
}
function lines(s, parts, o) { T(s, parts.flat(), Object.assign({ paraSpaceAfter: 5 }, o)); }

// token squares
function tokens(s, x, y, n, kept, sz) {
  sz = sz || 0.16;
  for (let i = 0; i < n; i++) {
    const on = kept(i);
    s.addShape(p.ShapeType.rect, { x: x + i * (sz + 0.05), y, w: sz, h: sz, fill: { color: on ? NAVY : WHITE },
      line: { color: on ? NAVY : "9AA3B5", width: 0.75, dashType: on ? "solid" : "dash" } });
  }
}
// pipeline of labelled steps; each step {t, tok (fn or null), accent}
function pipeline(s, x, y, steps, stepW, h) {
  stepW = stepW || 1.9; h = h || 0.85;
  const gap = 0.32;
  steps.forEach((st, k) => {
    const sx = x + k * (stepW + gap);
    box(s, sx, y, stepW, h, st.accent);
    if (st.tok) {
      tokens(s, sx + (stepW - (8 * 0.16 + 7 * 0.05)) / 2, y + 0.13, 8, st.tok, 0.16);
      const ty = h < 0.7 ? y + 0.3 : y + 0.42;
      T(s, st.t, { x: sx + 0.05, y: ty, w: stepW - 0.1, h: Math.max(h - (ty - y) - 0.02, 0.2), fontSize: h < 0.7 ? 10.5 : 12, align: "center", bold: true, color: st.accent ? ACC : NAVY });
    } else {
      T(s, st.t, { x: sx + 0.05, y, w: stepW - 0.1, h, fontSize: 12, align: "center", valign: "middle", bold: true, color: st.accent ? ACC : NAVY });
    }
    if (k < steps.length - 1)
      T(s, "→", { x: sx + stepW, y, w: gap, h, fontSize: 18, align: "center", valign: "middle", color: MUTE });
  });
}

// ============================================================ 1 TITLE
let s = p.addSlide();
T(s, "Visual Token Pruning and Text-Reading Performance\nin Vision-Language Models", { x: 0.8, y: 1.9, w: 11.7, h: 1.6, fontFace: HEAD, fontSize: 34, bold: true, color: NAVY });
T(s, "Phase 3 Investigation using Qwen2.5-VL-7B", { x: 0.8, y: 3.55, w: 11.7, h: 0.5, fontSize: 20, color: MUTE });
s.addShape(p.ShapeType.line, { x: 0.8, y: 4.35, w: 5.0, h: 0, line: { color: LINE, width: 1 } });
T(s, "P. S. Kedar", { x: 0.8, y: 4.6, w: 11.7, h: 0.4, fontSize: 17, bold: true });
T(s, "Supervised by Prof. Divya Saxena\nMentored by Aditya Sharma\nIIT Jodhpur", { x: 0.8, y: 5.05, w: 11.7, h: 1.1, fontSize: 14, color: MUTE, paraSpaceAfter: 2 });

// ============================================================ 2 SETUP
s = p.addSlide();
H(s, "Experimental Setup", "An image is converted into many visual tokens before the language model processes it.");
const rows2 = [
  { name: "Baseline", desc: "The original model without BTP. All visual tokens are kept.",
    why: "Tells us how well the model performs when we do not interfere with its visual information.",
    steps: [{ t: "Image", tok: i => true }, { t: "All visual tokens", tok: i => true }, { t: "Model" }, { t: "Answer" }] },
  { name: "Pruning (BTP)", desc: "BTP removes some visual tokens so the model processes less visual information.",
    why: "Some visual tokens remain, so the model can still use the image.",
    steps: [{ t: "Image", tok: i => true }, { t: "BTP pruning", tok: i => i % 2 === 0 }, { t: "Model" }, { t: "Answer" }] },
  { name: "Complete deletion", desc: "Removes every visual token still remaining at a particular layer.", accent: true,
    why: "After that layer, the model cannot receive any further visual information from the image.",
    steps: [{ t: "Image", tok: i => true }, { t: "Pruning: 12.5% remain", tok: i => i === 0 }, { t: "Complete deletion: 0 remain", tok: i => false, accent: true }, { t: "Answer" }] }
];
rows2.forEach((r, k) => {
  const y = 1.6 + k * 1.52;
  box(s, 0.6, y, 12.1, 1.38, r.accent);
  T(s, r.name, { x: 0.8, y: y + 0.12, w: 3.2, h: 0.35, fontFace: HEAD, fontSize: 17, bold: true, color: r.accent ? ACC : NAVY });
  T(s, r.desc, { x: 0.8, y: y + 0.5, w: 3.6, h: 0.45, fontSize: 12.5 });
  T(s, r.why, { x: 0.8, y: y + 0.95, w: 3.6, h: 0.4, fontSize: 11.5, italic: true, color: MUTE });
  pipeline(s, 4.6, y + 0.27, r.steps, 1.75, 0.85);
});
box(s, 0.6, 6.25, 12.1, 0.85, false);
T(s, [{ text: "Pruning", options: { bold: true, color: NAVY } }, { text: " = reducing visual information          " },
      { text: "Complete deletion", options: { bold: true, color: ACC } }, { text: " = cutting off visual information completely" }],
  { x: 0.8, y: 6.3, w: 11.7, h: 0.4, fontSize: 16, valign: "middle", align: "center" });
T(s, "This distinction is essential for every experiment that follows.", { x: 0.8, y: 6.7, w: 11.7, h: 0.3, fontSize: 12, italic: true, color: MUTE, align: "center" });

// ============================================================ 3 OBSERVATION
s = p.addSlide();
H(s, "Initial Observation", "We compared the original model (baseline) with BTP on Qwen2.5-VL-7B.");
const rt = (t) => ({ text: t, options: { bold: true, fill: ACCFILL } });
table(s, [
  ["Task", "Baseline", "BTP"],
  ["POPE", "87.6", "86.2"], ["MMBench", "83.7", "79.3"], ["GQA", "60.9", "55.9"], ["AI2D", "86.4", "79.6"],
  [rt("TextVQA *"), { text: "86.2", options: { fill: ACCFILL } }, { text: "23.3", options: { bold: true, color: ACC, fill: ACCFILL } }],
  [rt("DocVQA *"), { text: "94.7", options: { fill: ACCFILL } }, { text: "19.2", options: { bold: true, color: ACC, fill: ACCFILL } }],
  [rt("ChartQA *"), { text: "76.8", options: { fill: ACCFILL } }, { text: "29.0", options: { bold: true, color: ACC, fill: ACCFILL } }]
], { x: 0.6, y: 1.65, w: 5.4, colW: [2.4, 1.5, 1.5], rowH: 0.52 });
T(s, "* reading task", { x: 0.6, y: 5.9, w: 5.4, h: 0.3, fontSize: 12, italic: true, color: ACC });
box(s, 6.35, 1.65, 6.35, 1.3, false);
T(s, "General visual tasks", { x: 6.55, y: 1.78, w: 6.0, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY });
T(s, "POPE, MMBench, GQA, AI2D: performance decreases only moderately.", { x: 6.55, y: 2.18, w: 6.0, h: 0.7, fontSize: 14 });
box(s, 6.35, 3.1, 6.35, 1.95, false);
T(s, "Reading tasks: performance collapses", { x: 6.55, y: 3.22, w: 6.0, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: ACC });
T(s, "Tasks where the model must identify and understand text physically present inside the image.", { x: 6.55, y: 3.6, w: 6.0, h: 0.55, fontSize: 13.5 });
T(s, [{ text: "TextVQA", options: { bold: true } }, { text: "  questions answered from text in images", options: { breakLine: true } },
      { text: "DocVQA", options: { bold: true } }, { text: "  understanding text in documents", options: { breakLine: true } },
      { text: "ChartQA", options: { bold: true } }, { text: "  reading information from charts" }],
  { x: 6.55, y: 4.15, w: 6.0, h: 0.85, fontSize: 13 });
box(s, 6.35, 5.2, 6.35, 1.3, true);
T(s, "Main observation", { x: 6.55, y: 5.3, w: 6.0, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: ACC });
T(s, "The degradation is not uniform. It is especially severe when the model must read text from the image.", { x: 6.55, y: 5.68, w: 6.0, h: 0.75, fontSize: 14 });

// ============================================================ 4 POSSIBLE CAUSES
s = p.addSlide();
H(s, "Possible Causes", "Why does BTP make Qwen fail on reading tasks? We tested four explanations.");
const exps = [
  { q: "1. Are we keeping the wrong visual tokens?",
    why: "Perhaps BTP removes useful tokens and keeps unhelpful ones.",
    what: "Replaced BTP's token selection with other methods, including random selection.",
    res: "All methods stayed around 17–19% on TextVQA.",
    so: "Changing which tokens are selected does not recover performance." },
  { q: "2. Are text regions too similar or redundant?",
    why: "If text patches look alike, BTP may treat them as duplicates and remove them.",
    what: "Compared how similar text patches are to each other versus background patches.",
    res: "Text similarity 0.38, background similarity 0.44.",
    so: "Text was not more redundant than background. This explanation was not supported." },
  { q: "3. Are too many tokens being removed?",
    why: "Removing a large share of tokens might destroy the information needed for reading.",
    what: "Gradually changed token retention from 100% down to 12.5%.",
    res: "Even at 90% retention, TextVQA had already fallen to the same low level.",
    so: "The collapse happens even when almost all tokens are kept, so it is not the amount removed." },
  { q: "4. OCR-based text-token protection",
    why: "If BTP accidentally removes text, protecting the text tokens should restore performance.",
    what: "OCR gives the text locations in the image. We forced the visual tokens at those locations to survive pruning, a best-case test.",
    res: "19.7% with OCR-based protection, 19.7% without.",
    so: "Even explicitly protecting the text tokens does not recover performance." }
];
exps.forEach((e, k) => {
  const x = 0.6 + (k % 2) * 6.15, y = 1.55 + Math.floor(k / 2) * 2.5;
  box(s, x, y, 5.95, 2.38, false);
  T(s, e.q, { x: x + 0.2, y: y + 0.1, w: 5.55, h: 0.35, fontFace: HEAD, fontSize: 14.5, bold: true, color: NAVY });
  lines(s, [L("Why:", e.why), L("What:", e.what), L("Result:", e.res), L("So what:", e.so, ACC)],
    { x: x + 0.2, y: y + 0.5, w: 5.55, h: 1.82, fontSize: 11.5 });
});
T(s, "None of the four explanations can account for the collapse.", { x: 0.6, y: 6.6, w: 12.1, h: 0.4, fontSize: 15, bold: true, color: ACC });

// ============================================================ 5 CONTROL
s = p.addSlide();
H(s, "Control Experiment: 100% Token Retention", "Is the pruning itself responsible for the collapse?");
box(s, 0.6, 1.55, 12.1, 2.15, false);
T(s, "What does “BTP + 100% retention” mean?", { x: 0.8, y: 1.65, w: 11.7, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY });
T(s, "We keep the BTP procedure active, but every pruning stage is set to retain 100% of the visual tokens. BTP runs, but nothing is removed by pruning.", { x: 0.8, y: 2.02, w: 11.7, h: 0.45, fontSize: 13.5 });
T(s, "Normal BTP", { x: 0.8, y: 2.6, w: 2.2, h: 0.4, fontSize: 13, bold: true, valign: "middle" });
pipeline(s, 3.0, 2.55, [{ t: "100% tokens", tok: i => true }, { t: "Pruning stage", tok: i => i % 2 === 0 }, { t: "Fewer tokens", tok: i => i === 0 }], 1.75, 0.5);
T(s, "100% retention", { x: 0.8, y: 3.1, w: 2.2, h: 0.4, fontSize: 13, bold: true, valign: "middle", color: ACC });
pipeline(s, 3.0, 3.05, [{ t: "100% tokens", tok: i => true }, { t: "Pruning stage", tok: i => true }, { t: "100% remain", tok: i => true }], 1.75, 0.5);
lines(s, [L("Why:", "if pruning causes the collapse, removing the pruning should restore the original performance.")], { x: 0.6, y: 3.9, w: 12.1, h: 0.4, fontSize: 14 });
table(s, [["Condition", "TextVQA"], ["Baseline", "87.6%"], ["BTP + 100% retention", { text: "17.8%", options: { bold: true, color: ACC } }]],
  { x: 0.6, y: 4.35, w: 5.6, colW: [3.6, 2.0], fontSize: 15, rowH: 0.55 });
box(s, 6.55, 4.35, 6.15, 2.3, true);
T(s, "So what", { x: 6.75, y: 4.45, w: 5.8, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: ACC });
T(s, "The model still collapses even when no visual tokens are removed by pruning.\n\nThe major loss must be happening after the pruning stages.", { x: 6.75, y: 4.85, w: 5.8, h: 1.7, fontSize: 14 });

// ============================================================ 6 IMPLEMENTATION
s = p.addSlide();
H(s, "Implementation Finding", "What happens to the visual tokens after the pruning stages?");
pipeline(s, 0.6, 1.55, [
  { t: "Image", tok: i => true }, { t: "All visual tokens", tok: i => true }, { t: "BTP pruning: 12.5% remain", tok: i => i === 0 },
  { t: "Complete deletion", tok: i => false, accent: true }, { t: "0 visual tokens", accent: true }], 2.14, 0.85);
lines(s, [L("What we changed:", "kept the 12.5% pruning exactly the same, and switched off only the complete-deletion step.")], { x: 0.6, y: 2.6, w: 12.1, h: 0.4, fontSize: 14 });
table(s, [["Condition", "TextVQA"], ["No pruning", "85.65%"], ["BTP as released", { text: "17.25%", options: { bold: true, color: ACC } }], ["Complete deletion switched off", { text: "78.60%", options: { bold: true, color: NAVY } }]],
  { x: 0.6, y: 3.15, w: 5.6, colW: [3.6, 2.0], fontSize: 14, rowH: 0.5 });
box(s, 6.55, 3.15, 6.15, 1.55, false);
T(s, "Intuition", { x: 6.75, y: 3.25, w: 5.8, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY });
T(s, "If pruning were responsible, switching off deletion would leave the score low. Instead it rises from 17.25% to 78.60%, with the pruning still in place.", { x: 6.75, y: 3.62, w: 5.8, h: 1.0, fontSize: 13 });
box(s, 6.55, 4.85, 6.15, 1.2, true);
T(s, "Conclusion", { x: 6.75, y: 4.93, w: 5.8, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: ACC });
T(s, "Most of the TextVQA collapse comes from completely removing the remaining tokens, not from the earlier pruning.", { x: 6.75, y: 5.3, w: 5.8, h: 0.7, fontSize: 13 });
T(s, "Note: the paper specifies retaining 12.5% of visual tokens for Qwen at the final stage, while the implementation tested here removes the remaining tokens.", { x: 0.6, y: 6.35, w: 12.1, h: 0.6, fontSize: 12.5, italic: true, color: MUTE });

// ============================================================ 7 THRESHOLD
s = p.addSlide();
H(s, "Visual Token Access Threshold", "Complete deletion at one layer causes a large drop. Does the exact layer matter?");
lines(s, [L("What:", "we moved the complete-deletion point from early layers to later ones, and measured TextVQA at each point.")], { x: 0.6, y: 1.5, w: 12.1, h: 0.4, fontSize: 14 });
s.addImage(Object.assign({ x: 0.6, y: 1.95, w: 7.9, h: 3.95 }, img("fig9_threshold_single.png")));
box(s, 8.75, 1.95, 3.95, 1.55, false);
T(s, "Threshold", { x: 8.95, y: 2.05, w: 3.6, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY });
T(s, "The point where the model still keeps at least 50% of its original TextVQA performance.", { x: 8.95, y: 2.43, w: 3.6, h: 1.0, fontSize: 13 });
box(s, 8.75, 3.62, 3.95, 1.25, false);
T(s, [{ text: "Qwen2.5-VL-7B", options: { bold: true, color: NAVY, breakLine: true } }, { text: "≈ 22.5 of 28 layers  →  ≈ 80%", options: { breakLine: true } },
      { text: "InternVL2-2B", options: { bold: true, color: ACC, breakLine: true } }, { text: "≈ 18 of 24 layers  →  ≈ 75%" }],
  { x: 8.95, y: 3.72, w: 3.6, h: 1.1, fontSize: 13 });
T(s, "Relative threshold position: the threshold as a share of the model's layers.", { x: 8.75, y: 4.95, w: 3.95, h: 0.6, fontSize: 11, italic: true, color: MUTE });
box(s, 0.6, 6.05, 12.1, 1.0, true);
T(s, [{ text: "Remove visual information too early → reading collapses.   Keep it available longer → performance recovers sharply.", options: { bold: true, breakLine: true } },
      { text: "Measured on TextVQA, so these are task-specific measurements, not a universal law.", options: { italic: true, color: ACC } }],
  { x: 0.8, y: 6.13, w: 11.7, h: 0.85, fontSize: 13.5 });

// ============================================================ 8 COSTS
s = p.addSlide();
H(s, "Pruning and Deletion Costs", "Which part of the loss comes from pruning, and which from complete deletion?");
box(s, 0.6, 1.55, 5.2, 0.9, false);
T(s, [{ text: "Corrected BTP", options: { bold: true, color: NAVY } }, { text: " = normal BTP pruning to 12.5%, but without the complete-deletion step." }], { x: 0.8, y: 1.63, w: 4.8, h: 0.75, fontSize: 12.5 });
const defs = [["Pruning cost", "Baseline → Corrected BTP", "the cost of removing some visual tokens"],
              ["Deletion cost", "Corrected BTP → Released BTP", "the additional cost of removing all remaining tokens"],
              ["Dominant effect", "", "whichever causes the larger share of the total drop"]];
defs.forEach((d, k) => {
  const y = 2.6 + k * 1.05;
  box(s, 0.6, y, 5.2, 0.92, k === 1);
  T(s, d[0], { x: 0.8, y: y + 0.08, w: 2.0, h: 0.3, fontSize: 14, bold: true, color: k === 1 ? ACC : NAVY });
  if (d[1]) T(s, d[1], { x: 2.75, y: y + 0.08, w: 2.95, h: 0.3, fontSize: 12.5, bold: true });
  T(s, "This tells us " + d[2] + ".", { x: 0.8, y: y + 0.45, w: 4.8, h: 0.42, fontSize: 12 });
});
const B = t => ({ text: t, options: { bold: true } });
table(s, [
  ["Task", "Pruning cost", "Deletion cost", "Dominant"],
  ["TextVQA", "5.7", B("57.2"), B("Deletion")], ["DocVQA", "17.9", B("57.6"), B("Deletion")],
  ["ChartQA", B("31.4"), "16.4", B("Pruning")], ["AI2D", B("7.2"), "0.4", B("Pruning")],
  ["GQA", "2.1", "2.9", "Both small"], ["POPE", "1.3", "0.1", "Both small"], ["MMBench", "4.7", "0.3", "Both small"]
], { x: 6.1, y: 1.55, w: 6.6, colW: [1.7, 1.6, 1.6, 1.7], rowH: 0.5, fontSize: 13.5 });
T(s, "Points of performance lost, Qwen2.5-VL-7B.", { x: 6.1, y: 5.62, w: 6.6, h: 0.3, fontSize: 11, italic: true, color: MUTE });
box(s, 0.6, 5.95, 12.1, 1.1, true);
T(s, [{ text: "There is no single dominant effect for every task.", options: { bold: true, color: ACC, breakLine: true } },
      { text: "TextVQA and DocVQA: deletion dominates.   ChartQA: pruning dominates.   General tasks: both effects relatively small." }],
  { x: 0.8, y: 6.05, w: 11.7, h: 0.9, fontSize: 13.5 });

// ============================================================ 9 EVALUATION
s = p.addSlide();
H(s, "Evaluation Results", "Why did the standard evaluation not clearly reveal this problem?");
table(s, [
  ["Configuration", "Standard benchmarks", "Reading tasks"],
  ["Qwen, BTP as released", "96.1%", { text: "28.4%", options: { bold: true, color: ACC } }],
  ["Qwen, deletion switched off", "96.9%", { text: "77.9%", options: { bold: true } }],
  ["InternVL, pruning only", "94.9%", { text: "56.5%", options: { bold: true, color: ACC } }]
], { x: 0.6, y: 1.6, w: 7.2, colW: [3.2, 2.0, 2.0], fontSize: 15, rowH: 0.6 });
T(s, "Share of each model's original performance retained.", { x: 0.6, y: 4.08, w: 7.2, h: 0.3, fontSize: 11.5, italic: true, color: MUTE });
box(s, 8.1, 1.6, 4.6, 2.65, false);
T(s, [{ text: "Standard benchmarks", options: { bold: true, color: NAVY, breakLine: true } },
      { text: "Around 95–97% retained, so pruning appears largely successful.", options: { breakLine: true } },
      { text: " ", options: { breakLine: true, fontSize: 6 } },
      { text: "Reading tasks", options: { bold: true, color: ACC, breakLine: true } },
      { text: "Large losses become visible." }],
  { x: 8.3, y: 1.72, w: 4.25, h: 2.4, fontSize: 13.5 });
box(s, 0.6, 4.55, 12.1, 0.95, false);
T(s, [{ text: "Important comparison: ", options: { bold: true, color: NAVY } },
      { text: "InternVL does not have the complete-deletion issue, yet its reading performance still drops substantially. The blind spot is not only caused by the Qwen implementation." }],
  { x: 0.8, y: 4.65, w: 11.7, h: 0.8, fontSize: 14 });
box(s, 0.6, 5.7, 12.1, 1.0, true);
T(s, "A benchmark suite without a suitable text-reading task can miss substantial degradation in visual reading ability.", { x: 0.8, y: 5.7, w: 11.7, h: 1.0, fontSize: 16, bold: true, color: ACC, valign: "middle" });

// ============================================================ 10 CONCLUSION
s = p.addSlide();
H(s, "Conclusion and Next Steps");
[["Finding 1: Implementation", "In the Qwen configuration tested, complete removal of the remaining visual tokens causes most of the TextVQA collapse."],
 ["Finding 2: Model behaviour", "The model needs visual information to remain accessible up to a measurable point in the network. TextVQA threshold: Qwen ≈ 80%, InternVL ≈ 75%."],
 ["Finding 3: Evaluation", "Standard benchmarks can show healthy overall performance while visual text-reading ability has degraded substantially."]
].forEach((r, k) => {
  const y = 1.3 + k * 1.3;
  box(s, 0.6, y, 12.1, 1.15, k === 2);
  T(s, r[0], { x: 0.8, y: y + 0.1, w: 11.7, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: k === 2 ? ACC : NAVY });
  T(s, r[1], { x: 0.8, y: y + 0.5, w: 11.7, h: 0.6, fontSize: 14 });
});
T(s, "Next steps", { x: 0.6, y: 5.35, w: 12.1, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY });
T(s, [{ text: "1.  Measure the threshold on DocVQA and ChartQA", options: { breakLine: true } },
      { text: "2.  Repeat on a third architecture", options: { breakLine: true } },
      { text: "3.  Consolidate the results into the manuscript" }],
  { x: 0.6, y: 5.8, w: 12.1, h: 1.1, fontSize: 14, paraSpaceAfter: 4 });

p.writeFile({ fileName: "BTP_Phase3_Deck.pptx" }).then(f => console.log("WROTE", f));
