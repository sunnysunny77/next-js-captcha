"use server";
import * as tf from "@tensorflow/tfjs-node";
import path from "path";
import {createCanvas} from "canvas";

const labels: string[] = [
  "A","B","C","D","E","F","G","H","I","J","K","L","M",
  "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
];

const phoneticLabels = {
  "A": "ALPHA",   "B": "BRAVO",   "C": "CHARLIE", "D": "DELTA",
  "E": "ECHO",    "F": "FOXTROT", "G": "GOLF",    "H": "HOTEL",
  "I": "INDIA",   "J": "JULIET",  "K": "KILO",    "L": "LIMA",
  "M": "MIKE",    "N": "NOVEMBER","O": "OSCAR",   "P": "PAPA",
  "Q": "QUEBEC",  "R": "ROMEO",   "S": "SIERRA",  "T": "TANGO",
  "U": "UNIFORM", "V": "VICTOR",  "W": "WHISKEY", "X": "XRAY",
  "Y": "YANKEE",  "Z": "ZULU"
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
  const word = phoneticLabels[label];
  const canvas = createCanvas(122, 61);
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const dotCount = 61;
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

  ctx.strokeStyle = "rgba(0,0,0,0.2)";
  ctx.lineWidth = 0.5;
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

  const fontSize = 18;
  ctx.font = `bold ${fontSize}px Sans`;

  let totalWidth = 0;
  for (let char of word) {
    totalWidth += ctx.measureText(char).width * 0.8;
  }

  let x = (canvas.width - totalWidth) / 2;

  for (let char of word) {
    const angle = (Math.random() - 0.5) * 0.6;
    const offsetY = (Math.random() - 0.5) * 18;
    const color = `rgba(${Math.floor(Math.random() * 256)},${Math.floor(Math.random() * 256)},${Math.floor(Math.random() * 256)},1)`;

    ctx.save();
    ctx.fillStyle = color;
    ctx.translate(x, canvas.height / 2 + offsetY);
    ctx.rotate(angle);
    ctx.fillText(char, 0, 0);
    ctx.restore();

    x += ctx.measureText(char).width * 0.8;
  }

  return canvas.toDataURL();
};

export const getLabels = async (): Promise<string[]> => {
  currentLabels = Array.from(
    { length: 4 },
    () => labels[Math.floor(Math.random() * labels.length)]
  );
  const phoneticImages = currentLabels.map((label) => drawPhoneticLabel(label));
  return phoneticImages; 
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
