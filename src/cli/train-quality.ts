import { resolve } from "node:path";
import * as tf from "@tensorflow/tfjs";
import { loadArchetypeEncoder } from "../data/archetypes.js";
import { readSamplesJsonl } from "../data/jsonl.js";
import {
  createDecisionQualityModel,
  getDecisionQualityLabel,
  trainDecisionQualityModel,
} from "../model/decision-quality.js";
import { saveDecisionQualityModel } from "../model/decision-quality-storage.js";

const [
  datasetArg = "data/processed/human-moves.jsonl",
  modelArg = "models/decision-quality",
  archetypesArg = "data/processed/player-archetypes.json",
  epochsArg = "10",
] = process.argv.slice(2);

const epochs = Number(epochsArg);
if (!Number.isInteger(epochs) || epochs < 1 || epochs > 50) {
  throw new Error("Epoch count must be an integer between 1 and 50.");
}

const samples = [];
for await (const sample of readSamplesJsonl(resolve(datasetArg))) samples.push(sample);
const labeled = samples.filter((sample) => getDecisionQualityLabel(sample) !== undefined);
const classCounts = new Map<string, number>();
for (const sample of labeled) {
  const label = getDecisionQualityLabel(sample);
  if (label) classCounts.set(label, (classCounts.get(label) ?? 0) + 1);
}
if (labeled.length < 10 || classCounts.size < 2) {
  throw new Error(
    `Not enough annotated data: found ${labeled.length} usable labeled moves across ${classCounts.size} classes; need at least 10 moves and 2 classes.`,
  );
}

const archetypeModel = await loadArchetypeEncoder(resolve(archetypesArg));
console.log(`Training decision-quality model on ${labeled.length} explicitly annotated moves.`);
console.log("Unannotated moves are excluded; they are not assumed to be good.");
console.log(`Annotation distribution: ${[...classCounts].map(([label, count]) => `${label}=${count}`).join(", ")}`);
if (!archetypeModel) console.log("No archetype file found; fixed archetype inputs will remain unknown.");

await tf.ready();
const model = createDecisionQualityModel();
await trainDecisionQualityModel(model, labeled, { epochs, archetypeModel });
await saveDecisionQualityModel(model, resolve(modelArg), archetypeModel);
model.dispose();
console.log(`Saved decision-quality model and metadata to ${resolve(modelArg)}`);
