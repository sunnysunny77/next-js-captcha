"use server";
import * as tf from "@tensorflow/tfjs-node";
import path from "path";
import {createCanvas} from "canvas";
import {cookies} from "next/headers";
import crypto from "crypto";

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

let model: tf.GraphModel | null = null;

const loadModel = async (): Promise<void> => {
  if (!model) {
    const modelPath = path.join(process.cwd(), "tfjs_model/model.json");
    model = await tf.loadGraphModel(`file://${modelPath}`);
    console.log("Model loaded");
  }
};

const drawPhoneticLabel = (index) => {
  const width = 122;
  const height = 61;
  const fill = "white";
  const dotCount = 50;
  const lineStyle = "rgba(0,0,0,0.34)";
  const lineWidth = 0.5;
  const fontSize = 18;
  const font = `bold ${fontSize}px Sans`;
  const spacing = 0.05;
  const label = phoneticLabels[index]

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
  const charWidths = [];
  for (const char of label) {
    const w = ctx.measureText(char).width;
    charWidths.push(w);
    totalWidth += w * (1 + spacing);
  }

  let x = (width - totalWidth) / 2;

  for (let i = 0; i < label.length; i++) {
    const char = label[i];
    const angle = (Math.random() - 0.5) * 0.6;
    const offsetY = (Math.random() - 0.5) * 18;
    const min = 50;
    const max = 150;
    const r = Math.floor(Math.random() * (max - min) + min);
    const g = Math.floor(Math.random() * (max - min) + min);
    const b = Math.floor(Math.random() * (max - min) + min);
    ctx.save();
    ctx.fillStyle = `rgba(${r},${g},${b},1)`;
    ctx.translate(x, height / 2 + offsetY);
    ctx.rotate(angle);
    ctx.fillText(char, 0, 0);
    ctx.restore();

    x += charWidths[i] * (1 + spacing);
  }

  return canvas.toDataURL();
};

export const getLabels = async (): Promise<string[]> => {
  try {
    const currentLabels = Array.from({ length: 4 },() => labels[Math.floor(Math.random() * labels.length)]);
    const labelImages = currentLabels.map(label => drawPhoneticLabel(label));
    const secret = process.env.REACT_APP_AUTH_SECRET;
    const hash = crypto.createHmac("sha256", secret).update(JSON.stringify(currentLabels)).digest("hex");
    const cookieStore = await cookies();
    cookieStore.set("App-Captcha", hash, {secure: true, httpOnly: true, sameSite: "strict"});

    return labelImages;
  } catch (err) {
    console.error(err);
  }
};

const processImageNode = async (data, shape) => {
  const tensor = tf.tensor(data, shape, "float32").div(255.0);
  const input = tensor.expandDims(0);
  const prediction = model.predict(input) as tf.Tensor;
  const maxIndex = prediction.argMax(-1).dataSync()[0];
  tensor.dispose();
  input.dispose();
  prediction.dispose();

  return maxIndex;
};

export const getClassify = async (tensorArrays) => {
  try {
    if (!model) await loadModel();

    const results = await Promise.all(
      tensorArrays.map(index => processImageNode(index.data, index.shape))
    );

    const currentLabels = results.map(result => labels[result]);

    const cookieStore = await cookies();
    const storedHash = cookieStore.get("App-Captcha").value;
    const secret = process.env.REACT_APP_AUTH_SECRET;
    const incomingHash = crypto.createHmac("sha256", secret).update(JSON.stringify(currentLabels)).digest("hex");
    const match = crypto.timingSafeEqual(
      Buffer.from(incomingHash, "hex"),
      Buffer.from(storedHash, "hex")
    );

    return { correct:  match };
  } catch (err) {
    console.error(err);
  }
};
