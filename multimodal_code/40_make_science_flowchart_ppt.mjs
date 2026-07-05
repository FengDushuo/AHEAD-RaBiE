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
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: 16,
    color: "#222222",
    ...style,
  };
  return shape;
}

function addBox(slide, name, position, fill, stroke = "#4A4A4A") {
  return slide.shapes.add({
    geometry: "roundRect",
    name,
    position,
    fill,
    line: { style: "solid", fill: stroke, width: 1.3 },
    borderRadius: "rounded-lg",
  });
}

function addModule(slide, cfg) {
  const box = addBox(slide, cfg.name, cfg.position, cfg.fill, cfg.stroke || "#4A4A4A");
  addText(
    slide,
    `${cfg.name}_header`,
    cfg.header,
    {
      left: cfg.position.left + 14,
      top: cfg.position.top + 12,
      width: cfg.position.width - 28,
      height: 28,
    },
    { fontSize: 18, bold: true, color: cfg.headerColor || "#111111", alignment: "center" },
  );
  addText(
    slide,
    `${cfg.name}_body`,
    cfg.body,
    {
      left: cfg.position.left + 16,
      top: cfg.position.top + 48,
      width: cfg.position.width - 32,
      height: cfg.position.height - 58,
    },
    { fontSize: cfg.bodySize || 16, color: "#2A2A2A" },
  );
  return box;
}

function connect(slide, from, to, fromSide = "right", toSide = "left", dashed = false) {
  return slide.shapes.connect(from, to, {
    kind: "elbow",
    fromSide,
    toSide,
    line: { style: dashed ? "dash" : "solid", fill: "#333333", width: 1.5 },
    tail: { type: "arrow", width: "med", length: "med" },
  });
}

async function main() {
  const outDir = argValue("out-dir", "outputs_addh_science_flowchart_ppt");
  const outPptx = argValue("out-pptx", path.join(outDir, "addhout_science_fig1_editable_flowchart.pptx"));
  const previewDir = argValue("preview-dir", path.join(outDir, "preview"));
  await fs.mkdir(outDir, { recursive: true });
  await fs.mkdir(previewDir, { recursive: true });

  const presentation = Presentation.create({
    slideSize: { width: 1600, height: 900 },
  });

  const slide = presentation.slides.add();
  slide.background.fill = "#FFFFFF";

  addText(
    slide,
    "title",
    "AddH-out target-domain prediction workflow",
    { left: 64, top: 38, width: 1030, height: 52 },
    { fontSize: 38, bold: true, color: "#111111" },
  );
  addText(
    slide,
    "subtitle",
    "Strict source-domain learning is combined with chemistry-guided anchoring and frozen few-shot AddH-out calibration.",
    { left: 64, top: 90, width: 1320, height: 36 },
    { fontSize: 18, color: "#555555" },
  );

  const top = 150;
  const h = 330;
  const w = 276;
  const gap = 18;
  const x0 = 64;

  const data = addModule(slide, {
    name: "module_data",
    header: "1. Source and target data",
    fill: "#F1F1F1",
    position: { left: x0, top, width: w, height: h },
    body:
      "addH / addH-2 training labels\n" +
      "- energy.dat and Excel formats\n" +
      "- unified IDs, hosts, dopants\n" +
      "- normalized H adsorption energy\n\n" +
      "AddH-out target domain\n" +
      "- CeO2 and ZnO structures\n" +
      "- used only in frozen validation",
  });

  const features = addModule(slide, {
    name: "module_features",
    header: "2. Multimodal features",
    fill: "#DCEAF7",
    position: { left: x0 + (w + gap), top, width: w, height: h },
    body:
      "Composition and host descriptors\n" +
      "- host, dopant, site identity\n" +
      "- source-dopant statistics\n\n" +
      "Structure-aware descriptors\n" +
      "- local H/site geometry\n" +
      "- FairChem graph embeddings\n" +
      "- EquiformerV2 features\n\n" +
      "LLM elemental priors\n" +
      "- reducibility\n" +
      "- oxygen affinity\n" +
      "- H-binding rank prior",
  });

  const sourceModels = addModule(slide, {
    name: "module_source_models",
    header: "3. Source-domain learners",
    fill: "#DCEAF7",
    position: { left: x0 + 2 * (w + gap), top, width: w, height: h },
    body:
      "Base learners\n" +
      "- multiview ensemble\n" +
      "- single-view branches\n" +
      "- graph branches\n" +
      "- residual and delta heads\n\n" +
      "Model selection\n" +
      "- cross-validated OOF errors\n" +
      "- rank-trend calibration\n" +
      "- guarded superblend\n\n" +
      "Strict blind baseline\n" +
      "- no AddH-out labels",
  });

  const chemistry = addModule(slide, {
    name: "module_chemistry",
    header: "4. Chemistry-guided anchor",
    fill: "#F7E1D5",
    position: { left: x0 + 3 * (w + gap), top, width: w, height: h },
    body:
      "Frozen expert-rule layer\n" +
      "- redox-active dopant regimes\n" +
      "- CeO2/ZnO host shifts\n" +
      "- conservative anchor\n" +
      "- balanced anchor\n\n" +
      "Purpose\n" +
      "- correct source-target shift\n" +
      "- preserve rank trend\n" +
      "- constrain extrapolation",
  });

  const validation = addModule(slide, {
    name: "module_validation",
    header: "5. AddH-out validation",
    fill: "#D7EAD8",
    position: { left: x0 + 4 * (w + gap), top, width: w, height: h },
    body:
      "Frozen repeated evaluation\n" +
      "- material-stratified splits\n" +
      "- few-shot calibration subset\n" +
      "- held-out test subset\n\n" +
      "Reported metrics\n" +
      "- MAE and RMSE\n" +
      "- Pearson and Spearman\n" +
      "- calibration sensitivity\n" +
      "- leave-one-dopant-out",
  });

  connect(slide, data, features);
  connect(slide, features, sourceModels);
  connect(slide, sourceModels, chemistry);
  connect(slide, chemistry, validation);

  const lowerTop = 520;
  const calib = addModule(slide, {
    name: "module_calibration",
    header: "Few-shot target-domain calibration",
    fill: "#FFF2CC",
    stroke: "#7A6332",
    position: { left: 560, top: lowerTop, width: 420, height: 148 },
    body:
      "Only calibration-fold AddH-out labels are used.\n" +
      "Residual correction is learned relative to the chemistry-guided anchor.\n" +
      "The remaining AddH-out samples stay held out for evaluation.",
    bodySize: 16,
  });

  const safeguards = addModule(slide, {
    name: "module_safeguards",
    header: "Anti-leakage and claims",
    fill: "#F8F8F8",
    stroke: "#777777",
    position: { left: 1010, top: lowerTop, width: 454, height: 148 },
    body:
      "Strict baseline: trained only on addH/addH-2.\n" +
      "Few-shot result: reported from repeated held-out AddH-out splits.\n" +
      "Post-hoc bidirectional chemistry prior is treated only as an upper-bound/reference ablation.",
    bodySize: 16,
  });

  connect(slide, chemistry, calib, "bottom", "top");
  connect(slide, calib, validation, "right", "bottom");
  connect(slide, calib, safeguards, "right", "left", true);

  const metrics = addBox(
    slide,
    "result_band",
    { left: 64, top: 718, width: 1400, height: 96 },
    "#F4FAF7",
    "#53956F",
  );
  addText(
    slide,
    "result_header",
    "Main evidence used in the manuscript",
    { left: 92, top: 732, width: 390, height: 28 },
    { fontSize: 18, bold: true, color: "#145A32" },
  );
  addText(
    slide,
    "result_body",
    "Strict source-domain model shows severe AddH-out domain shift. Chemistry-guided anchoring restores most of the rank trend. "
      + "Frozen few-shot calibration further reduces held-out error and produces stable AddH-out rank prediction across repeated splits.",
    { left: 92, top: 762, width: 1310, height: 38 },
    { fontSize: 16.5, color: "#1F1F1F" },
  );

  addText(
    slide,
    "caption",
    "Editable figure: all boxes, labels, and arrows are native PowerPoint objects.",
    { left: 64, top: 834, width: 900, height: 24 },
    { fontSize: 16, color: "#666666" },
  );

  const preview = await presentation.export({ slide, format: "png", scale: 1 });
  await writeBlob(path.join(previewDir, "slide-01.png"), preview);
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(previewDir, "slide-01.layout.json"), await layout.text(), "utf8");

  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await writeBlob(path.join(previewDir, "deck-montage.webp"), montage);

  const snapshot = await presentation.inspect({
    kind: "slide,textbox,shape",
    maxChars: 12000,
  });
  await fs.writeFile(path.join(previewDir, "inspect.ndjson"), snapshot.ndjson, "utf8");

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(outPptx);
  console.log(`[OK] wrote ${outPptx}`);
  console.log(`[OK] preview ${path.join(previewDir, "slide-01.png")}`);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
