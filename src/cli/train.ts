import { resolve } from "node:path";
import * as tf from "@tensorflow/tfjs";
import { readSamplesJsonl } from "../data/jsonl.js";
import { createPolicyModel, trainPolicy } from "../model/policy.js";
import { savePolicyModel } from "../model/storage.js";

const [datasetArg = "data/processed/human-moves.jsonl", modelArg = "models/human-policy"] = process.argv.slice(2);
const samples = [];
for await (const sample of readSamplesJsonl(resolve(datasetArg))) samples.push(sample);

console.log(`Training human move policy on ${samples.length} positions...`);
await tf.ready();
const model = createPolicyModel();
await trainPolicy(model, samples);
await savePolicyModel(model, resolve(modelArg));
console.log(`Saved model to ${resolve(modelArg)}`);
