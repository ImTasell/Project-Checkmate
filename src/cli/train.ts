import { resolve } from "node:path";
import * as tf from "@tensorflow/tfjs";
import { loadArchetypeEncoder } from "../data/archetypes.js";
import { readSamplesJsonl } from "../data/jsonl.js";
import { createPolicyModel, trainPolicy } from "../model/policy.js";
import { savePolicyModel } from "../model/storage.js";

const [
  datasetArg = "data/processed/human-moves.jsonl",
  modelArg = "models/human-policy",
  archetypesArg = "data/processed/player-archetypes.json",
] = process.argv.slice(2);

const samples = [];
for await (const sample of readSamplesJsonl(resolve(datasetArg))) samples.push(sample);
const archetypeModel = await loadArchetypeEncoder(resolve(archetypesArg));

console.log(`Training human move policy on ${samples.length} positions...`);
console.log(archetypeModel
  ? `Archetype conditioning enabled with ${archetypeModel.archetypes.length} discovered styles.`
  : "No archetype file found; training with continuous style features only.");
await tf.ready();
const model = createPolicyModel();
await trainPolicy(model, samples, { archetypeModel });
await savePolicyModel(model, resolve(modelArg), archetypeModel);
model.dispose();
console.log(`Saved model and conditioning metadata to ${resolve(modelArg)}`);
