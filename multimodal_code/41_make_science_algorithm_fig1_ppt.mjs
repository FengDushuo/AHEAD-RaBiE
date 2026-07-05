import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

async function writeBlob(filePath, blob) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, Buffer.from(await blob.arrayBuffer()));
}

function argValue(name, fallback) {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx >= 0 && process.argv[idx + 1]) return process.argv[idx + 1];
  return fallback;
}

function addText(slide, name, text, position, style = {}) {
  const s = slide.shapes.add({
    geometry: "textbox",
    name,
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  s.text = text;
  s.text.style = {
    fontSize: 15,
    color: "#1F1F1F",
    ...style,
  };
  return s;
}

function addBox(slide, name, position, fill, line = "#4A4A4A", radius = "rounded-lg") {
  return slide.shapes.add({
    geometry: "roundRect",
    name,
    position,
    fill,
    line: { style: "solid", fill: line, width: 1.1 },
    borderRadius: radius,
  });
}

function addLabeledBox(slide, cfg) {
  const box = addBox(slide, cfg.name, cfg.position, cfg.fill, cfg.line || "#4A4A4A");
  addText(
    slide,
    `${cfg.name}_title`,
    cfg.title,
    {
      left: cfg.position.left + 14,
      top: cfg.position.top + 10,
      width: cfg.position.width - 28,
      height: 28,
    },
    { fontSize: cfg.titleSize || 17, bold: true, alignment: "center", color: cfg.titleColor || "#111111" },
  );
  if (cfg.body) {
    addText(
      slide,
      `${cfg.name}_body`,
      cfg.body,
      {
        left: cfg.position.left + 16,
        top: cfg.position.top + 44,
        width: cfg.position.width - 32,
        height: cfg.position.height - 52,
      },
      { fontSize: cfg.bodySize || 14, color: "#2A2A2A" },
    );
  }
  return box;
}

function connect(slide, from, to, fromSide = "right", toSide = "left", dashed = false) {
  const connector = slide.shapes.connect(from, to, {
    kind: "elbow",
    fromSide,
    toSide,
    line: { style: dashed ? "dash" : "solid", fill: "#333333", width: 1.2 },
    tail: { type: "arrow", width: "med", length: "med" },
  });
  connector.bringToFront();
  return connector;
}

function addCircle(slide, name, x, y, r, fill, line = "#333333") {
  return slide.shapes.add({
    geometry: "ellipse",
    name,
    position: { left: x - r, top: y - r, width: 2 * r, height: 2 * r },
    fill,
    line: { style: "solid", fill: line, width: 0.8 },
  });
}

function addChip(slide, name, text, x, y, w, fill, line = "#6B7280") {
  const chip = addBox(slide, name, { left: x, top: y, width: w, height: 38 }, fill, line, "rounded-md");
  addText(
    slide,
    `${name}_text`,
    text,
    { left: x + 8, top: y + 8, width: w - 16, height: 22 },
    { fontSize: 12.5, bold: true, alignment: "center", color: "#1F1F1F" },
  );
  return chip;
}

function addEquation(slide, name, text, x, y, w, h) {
  const box = addBox(slide, name, { left: x, top: y, width: w, height: h }, "#FFFFFF", "#A6A6A6", "rounded-md");
  addText(
    slide,
    `${name}_text`,
    text,
    { left: x + 12, top: y + 10, width: w - 24, height: h - 18 },
    { fontSize: 13.2, color: "#111111", alignment: "center" },
  );
  return box;
}

function addMiniNetwork(slide, left, top) {
  const nodes = [];
  const colors = ["#DCEAF7", "#F8F8F8", "#D7EAD8"];
  for (let layer = 0; layer < 3; layer++) {
    for (let j = 0; j < 3; j++) {
      nodes.push(addCircle(slide, `nn_${layer}_${j}`, left + layer * 52, top + j * 34, 8, colors[layer]));
    }
  }
  for (let layer = 0; layer < 2; layer++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 3; k++) {
        connect(slide, nodes[layer * 3 + j], nodes[(layer + 1) * 3 + k], "right", "left");
      }
    }
  }
  addText(slide, "nn_label", "shared multi-view backbone", { left, top: top + 104, width: 160, height: 22 }, { fontSize: 12, alignment: "center", color: "#555555" });
}

async function main() {
  const outDir = argValue("out-dir", "outputs_addh_science_algorithm_fig1");
  const outPptx = argValue("out-pptx", path.join(outDir, "addhout_science_algorithm_fig1_editable.pptx"));
  const previewDir = argValue("preview-dir", path.join(outDir, "preview"));
  await fs.mkdir(outDir, { recursive: true });
  await fs.mkdir(previewDir, { recursive: true });

  const presentation = Presentation.create({ slideSize: { width: 1600, height: 900 } });
  const slide = presentation.slides.add();
  slide.background.fill = "#FFFFFF";

  addText(
    slide,
    "title",
    "Multi-view fine-tuning and target-domain prediction for H adsorption energy",
    { left: 62, top: 34, width: 1250, height: 44 },
    { fontSize: 32, bold: true, color: "#111111" },
  );
  addText(
    slide,
    "subtitle",
    "The source-trained multi-view backbone is fine-tuned on the curated addH/addH-2 data and calibrated for CeO2/ZnO AddH-out prediction through chemistry-guided few-shot residual learning.",
    { left: 62, top: 80, width: 1410, height: 32 },
    { fontSize: 16, color: "#555555" },
  );

  const sourcePanel = addLabeledBox(slide, {
    name: "panel_source",
    title: "A. Source-domain fine-tuning",
    fill: "#F4F6F9",
    line: "#6B7280",
    position: { left: 60, top: 136, width: 480, height: 558 },
    body: "",
  });
  const representationPanel = addLabeledBox(slide, {
    name: "panel_representation",
    title: "B. Multi-view representation and fusion",
    fill: "#F2F8FC",
    line: "#507AA3",
    position: { left: 560, top: 136, width: 480, height: 558 },
    body: "",
  });
  const targetPanel = addLabeledBox(slide, {
    name: "panel_target",
    title: "C. CeO2/ZnO target prediction",
    fill: "#F7FAF5",
    line: "#5F8F69",
    position: { left: 1060, top: 136, width: 480, height: 558 },
    body: "",
  });

  // Source-domain panel.
  const trainData = addLabeledBox(slide, {
    name: "source_data",
    title: "Curated source labels",
    fill: "#FFFFFF",
    line: "#A6A6A6",
    position: { left: 88, top: 196, width: 190, height: 134 },
    titleSize: 14.5,
    body:
      "addH + addH-2\n" +
      "DFT H adsorption energies\n" +
      "hosts, dopants, sites\n" +
      "unified target scale",
    bodySize: 12.4,
  });
  const theta0 = addEquation(slide, "theta0_box", "multi-view initialization\nθ0", 322, 214, 170, 78);
  connect(slide, trainData, theta0);

  addMiniNetwork(slide, 170, 370);
  const lossEq = addEquation(
    slide,
    "source_loss",
    "θ* = arg minθ  (1/Ns) Σ Huberδ(yi - fθ(xi)) + λ||θ - θ0||²",
    84,
    548,
    414,
    76,
  );
  connect(slide, theta0, lossEq, "bottom", "top");

  // Representation panel.
  const chips = [
    addChip(slide, "chip_comp", "composition", 594, 216, 130, "#FFFFFF"),
    addChip(slide, "chip_geom", "local geometry", 744, 216, 130, "#FFFFFF"),
    addChip(slide, "chip_graph", "graph embedding", 894, 216, 130, "#FFFFFF"),
    addChip(slide, "chip_prior", "element prior", 670, 276, 130, "#FFFFFF"),
    addChip(slide, "chip_stats", "source statistics", 820, 276, 130, "#FFFFFF"),
  ];
  const encoder = addLabeledBox(slide, {
    name: "view_encoders",
    title: "view encoders φv",
    fill: "#DCEAF7",
    line: "#507AA3",
    position: { left: 662, top: 360, width: 276, height: 72 },
    titleSize: 15,
    body: "hv = φv(xv)",
    bodySize: 13,
  });
  for (const chip of chips) connect(slide, chip, encoder, "bottom", "top");
  const fusionEq = addEquation(
    slide,
    "fusion_equation",
    "αv = softmax(uT hv)\nzi = Σv αv hv\nŷMV = gθ*(zi)",
    638,
    484,
    324,
    104,
  );
  connect(slide, encoder, fusionEq, "bottom", "top");

  // Target-domain panel.
  const materialCard = addLabeledBox(slide, {
    name: "target_materials",
    title: "Target oxides",
    fill: "#FFFFFF",
    line: "#A6A6A6",
    position: { left: 1090, top: 202, width: 164, height: 112 },
    titleSize: 14.5,
    body: "CeO2\nZnO\nAddH-out structures",
    bodySize: 12.6,
  });
  addCircle(slide, "ceo2_dot", 1296, 236, 10, "#009E73", "#009E73");
  addText(slide, "ceo2_label", "CeO2", { left: 1314, top: 226, width: 54, height: 20 }, { fontSize: 12.5 });
  addCircle(slide, "zno_dot", 1296, 270, 10, "#CC79A7", "#CC79A7");
  addText(slide, "zno_label", "ZnO", { left: 1314, top: 260, width: 54, height: 20 }, { fontSize: 12.5 });

  const anchorEq = addEquation(
    slide,
    "anchor_equation",
    "ai = ŷMV,i + b(hosti, dopanti, regimei)",
    1092,
    356,
    386,
    64,
  );
  connect(slide, materialCard, anchorEq, "bottom", "top");
  connect(slide, fusionEq, anchorEq, "right", "left", true);

  const residualEq = addEquation(
    slide,
    "residual_equation",
    "ỹi = ai + rη(qi)\nη* = arg minη ΣC (yj - aj - rη(qj))² + γ||η||²",
    1092,
    462,
    386,
    100,
  );
  connect(slide, anchorEq, residualEq, "bottom", "top");

  const validationBox = addLabeledBox(slide, {
    name: "validation_box",
    title: "Frozen held-out evaluation",
    fill: "#D7EAD8",
    line: "#5F8F69",
    position: { left: 1092, top: 594, width: 386, height: 88 },
    titleSize: 14.5,
    body: "material-stratified repeated splits\nMAE, RMSE, Pearson, Spearman",
    bodySize: 12.2,
  });
  connect(slide, residualEq, validationBox, "bottom", "top");

  // Mini parity schematic inside target panel.
  const plotFrame = addBox(slide, "mini_parity_frame", { left: 1372, top: 214, width: 108, height: 86 }, "#FFFFFF", "#D0D0D0", "rounded-md");
  const p1 = addCircle(slide, "parity_low", 1392, 278, 3.5, "#CC79A7", "#CC79A7");
  const p2 = addCircle(slide, "parity_mid", 1420, 252, 3.5, "#009E73", "#009E73");
  const p3 = addCircle(slide, "parity_high", 1452, 230, 3.5, "#009E73", "#009E73");
  connect(slide, p1, p3, "right", "left", false);
  addText(slide, "mini_parity_label", "trend recovery", { left: 1378, top: 302, width: 95, height: 20 }, { fontSize: 10.5, alignment: "center", color: "#555555" });

  // Bottom claim strip.
  const claim = addBox(slide, "claim_strip", { left: 60, top: 724, width: 1480, height: 104 }, "#F7FBFA", "#6AA682", "rounded-lg");
  addText(
    slide,
    "claim_title",
    "Interpretation used in the manuscript",
    { left: 88, top: 742, width: 500, height: 24 },
    { fontSize: 17, bold: true, color: "#145A32" },
  );
  addText(
    slide,
    "claim_body",
    "The central claim is not a fully blind AddH-out extrapolation. The model is a multi-view predictor fine-tuned on the curated source-domain H adsorption data, then adapted to the CeO2/ZnO target domain through a frozen chemistry anchor and few-shot residual calibration. Performance claims are based on held-out AddH-out samples in repeated material-stratified splits.",
    { left: 88, top: 772, width: 1412, height: 42 },
    { fontSize: 14.8, color: "#1F1F1F" },
  );

  addText(
    slide,
    "figure_note",
    "Editable Fig. 1: boxes, equations, material markers, and arrows are native PowerPoint objects.",
    { left: 62, top: 848, width: 1000, height: 22 },
    { fontSize: 13.5, color: "#666666" },
  );

  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await writeBlob(path.join(previewDir, "slide-01.png"), png);
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(previewDir, "slide-01.layout.json"), await layout.text(), "utf8");
  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await writeBlob(path.join(previewDir, "deck-montage.webp"), montage);
  const inspect = await presentation.inspect({ kind: "slide,textbox,shape", maxChars: 16000 });
  await fs.writeFile(path.join(previewDir, "inspect.ndjson"), inspect.ndjson, "utf8");
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(outPptx);
  console.log(`[OK] wrote ${outPptx}`);
  console.log(`[OK] preview ${path.join(previewDir, "slide-01.png")}`);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
