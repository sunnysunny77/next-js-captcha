"use server";
import * as tf from "@tensorflow/tfjs-node";
import path from "path";
import {createCanvas} from "canvas";

const labels: string[] = [
  "A","B","C","D","E","F","G","H","I","J","K","L","M",
  "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
];

const phoneticLabels = {
  "A": "ALPHA",   "B": "BRAVO",   "C": "CHARLIE", "D": "DELTA",
  "E": "ECHO",    "F": "FOXTROT", "G": "GOLF",    "H": "HOTEL",
  "I": "INDIA",   "J": "JULIET",  "K": "KILO",    "L": "LIMA",
  "M": "MIKE",    "N": "NOVEMBER","O": "OSCAR",   "P": "PAPA",
  "Q": "QUEBEC",  "R": "ROMEO",   "S": "SIERRA",  "T": "TANGO",
  "U": "UNIFORM", "V": "VICTOR",  "W": "WHISKEY", "X": "XRAY",
  "Y": "YANKEE",  "Z": "ZULU",
};

let currentLabels: string[] = [];

let model: tf.GraphModel | null = null;

const loadModel = async (): Promise<void> => {
  if (!model) {
    const modelPath = path.join(process.cwd(), "tfjs_model/model.json");
    model = await tf.loadGraphModel(`file://${modelPath}`);
    console.log("Model loaded");
  }
};

const drawPhoneticLabel = (label) => {
  const width = 122;
  const height = 61;
  const fill = "white";
  const dotCount = 50;
  const lineStyle = "rgba(0,0,0,0.34)";
  const lineWidth = 0.5;
  const fontSize = 18;
  const font = `bold ${fontSize}px Sans`;
  const overlap = 0.05;

  const word = phoneticLabels[label];
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = fill;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let i = 0; i < dotCount; i++) {
    const r = Math.floor(Math.random() * 256);
    const g = Math.floor(Math.random() * 256);
    const b = Math.floor(Math.random() * 256);
    const alpha = Math.random() < 0.5 ? 0.3 : 0.6;
    const radius = Math.random() * 3 + 1;
    ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
    ctx.beginPath();
    ctx.arc(Math.random() * canvas.width, Math.random() * canvas.height, radius, 0, 2 * Math.PI);
    ctx.fill();
  }

  ctx.strokeStyle = lineStyle;
  ctx.lineWidth = lineWidth;
  for (let j = 0; j < 2; j++) {
    ctx.beginPath();
    ctx.moveTo(0, Math.random() * canvas.height);
    for (let x = 0; x < canvas.width; x += 5) {
      ctx.lineTo(
        x,
        (canvas.height / 2) + Math.sin(x / 5 + Math.random() * 2) * 12 + (Math.random() * 20 - 10)
      );
    }
    ctx.stroke();
  }

  ctx.font = font;
  let totalWidth = 0;
  for (let char of word) {
    totalWidth += ctx.measureText(char).width * 0.8;
  }
  let x = (canvas.width - totalWidth) / 2;
  for (let char of word) {
    const angle = (Math.random() - 0.5) * 0.6;
    const offsetY = (Math.random() - 0.5) * 18;
    const min = 50;
    const max = 150;
    const r = Math.floor(Math.random() * (max - min) + min);
    const g = Math.floor(Math.random() * (max - min) + min);
    const b = Math.floor(Math.random() * (max - min) + min);
    const color = `rgba(${r},${g},${b},1)`
    ctx.save();
    ctx.fillStyle = color;
    ctx.translate(x, canvas.height / 2 + offsetY);
    ctx.rotate(angle);
    ctx.fillText(char, 0, 0);
    ctx.restore();

    const overlapCalc = -ctx.measureText(char).width * overlap;
    x += ctx.measureText(char).width * 0.8 + overlapCalc;
  }

  return canvas.toDataURL();
};

export const getLabels = async (): Promise<string[]> => {
  currentLabels = Array.from({ length: 4 },() => labels[Math.floor(Math.random() * labels.length)]);
  const labelImages = currentLabels.map(label => drawPhoneticLabel(label));
  return labelImages;
};

const processImageNode = async (data, shape) => {
  const tensor = tf.tensor(data, shape);

  const mask = tensor.greater(0.1);
  const coords = await tf.whereAsync(mask);

  if (coords.shape[0] === 0) {
    tensor.dispose();
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

  const sliced = tensor.slice([minY, minX, 0], [height, width, 1]);

  const scale: number = 20 / Math.max(height, width);
  const newHeight: number = Math.round(height * scale);
  const newWidth: number = Math.round(width * scale);
  const resized = sliced.resizeBilinear([newHeight, newWidth]);

  const top: number = Math.floor((28 - newHeight) / 2);
  const bottom: number = 28 - newHeight - top;
  const left: number = Math.floor((28 - newWidth) / 2);
  const right: number = 28 - newWidth - left;
  const input = resized.pad([[top, bottom], [left, right], [0, 0]]).expandDims(0);

  const prediction = model.predict(input) as tf.Tensor;
  const maxIndex = prediction.argMax(-1).dataSync()[0];

  tensor.dispose();
  mask.dispose();
  coords.dispose();
  ys.dispose();
  xs.dispose();
  sliced.dispose();
  resized.dispose();
  input.dispose();
  prediction.dispose();

  return maxIndex;
};

export const getClassify = async (tensorArrays) => {
  if (!model) await loadModel();

  if (!currentLabels || currentLabels.length !== tensorArrays.length) {
    throw new Error("Error");
  }

  return Promise.all(
    tensorArrays.map(async (index, i) => {
      const predIndex = await processImageNode(index.data, index.shape);
      return {
        correctLabel: currentLabels[i],
        predictedLabel: predIndex !== null ? labels[predIndex] : null,
      };
    })
  );
};
