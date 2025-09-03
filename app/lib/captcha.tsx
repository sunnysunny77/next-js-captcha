"use server";

import * as tf from "@tensorflow/tfjs-node";
import path from "path";

const labels: string[] = ["0","1","2","3","4","5","6","7","8","9"];
let currentLabels: string[] = [];
let model: tf.GraphModel | null = null;

const loadModel = async (): Promise<void> => {
  if (!model) {
    const modelPath = path.join(process.cwd(), "tfjs_model/model.json");
    model = await tf.loadGraphModel(`file://${modelPath}`);
    console.log("Model loaded");
  }
};

export const getLabels = async (): Promise<string[]> => {
  currentLabels = Array.from(
    { length: 4 },
    () => labels[Math.floor(Math.random() * labels.length)]
  );
  return currentLabels;
};

const processImageNode = async (imageBuffer: Buffer): Promise<number | null> => {

  let img = tf.node.decodeImage(imageBuffer, 3).toFloat().div(255.0);

  const mask = img.greater(0.1);
  const coords = await tf.whereAsync(mask);

  if (coords.shape[0] === 0) {
    img.dispose();
    mask.dispose();
    coords.dispose();
    return null;
  }

  const ys = coords.slice([0, 0], [-1, 1]).squeeze();
  const xs = coords.slice([0, 1], [-1, 1]).squeeze();

  const minY: number = ys.min().arraySync() as number;
  const maxY: number = ys.max().arraySync() as number;
  const minX: number = xs.min().arraySync() as number;
  const maxX: number = xs.max().arraySync() as number;

  const width: number = maxX - minX + 1;
  const height: number = maxY - minY + 1;

  let imgTensor = img.mean(2).expandDims(2);
  img.dispose();
  mask.dispose();
  coords.dispose();
  ys.dispose();
  xs.dispose();

  imgTensor = imgTensor.slice([minY, minX, 0], [height, width, 1]);

  const scale: number = 20 / Math.max(height, width);
  const newHeight: number = Math.round(height * scale);
  const newWidth: number = Math.round(width * scale);
  imgTensor = imgTensor.resizeBilinear([newHeight, newWidth]);

  const top: number = Math.floor((28 - newHeight) / 2);
  const bottom: number = 28 - newHeight - top;
  const left: number = Math.floor((28 - newWidth) / 2);
  const right: number = 28 - newWidth - left;

  imgTensor = imgTensor
    .pad([[top, bottom], [left, right], [0, 0]])
    .expandDims(0);

  const prediction = model!.predict(imgTensor) as tf.Tensor;
  const maxIndex: number = prediction.argMax(-1).dataSync()[0];

  prediction.dispose();
  imgTensor.dispose();

  return maxIndex;
};

export const getClassify = async (
  images: string[]
): Promise<{ correctLabel: string; predictedLabel: string | null }[]> => {
  if (!model) await loadModel();

  if (!currentLabels || currentLabels.length !== images.length) {
    throw new Error("Server labels not set or mismatch");
  }

  return Promise.all(
    images.map(async (base64, i) => {
      const buffer = Buffer.from(base64, "base64");
      const predIndex = await processImageNode(buffer);
      const predLabel = predIndex !== null ? labels[predIndex] : null;

      return {
        correctLabel: currentLabels[i],
        predictedLabel: predLabel,
      };
    })
  );
};
