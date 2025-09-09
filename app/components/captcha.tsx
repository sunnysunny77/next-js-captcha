"use client";
import * as tf from "@tensorflow/tfjs";
import {useEffect, useRef, useState, useCallback} from "react";
import {getLabels, getClassify} from "@/lib/captcha";
import Image from "next/image";
import Spinner from "@/images/spinner.gif";

const SIZE = 140;
const INVERT = false;

const Captcha = () => {
  const quadRef = useRef(null);
  const predictBtnRef = useRef(null);
  const canvasesRef = useRef([]);
  const contextsRef = useRef([]);
  const drawingRef = useRef([false, false, false, false]);

  const [labels, setLabels] = useState(null);
  const [message, setMessage] = useState("Loading");
  const [disabled, setDisabled] = useState(false);

  const setRandomLabels = useCallback( async () => {
    try {
      const res = await getLabels();
      setLabels(res);
    } catch (err) {
      console.error(err);
      setMessage("Error");
    }
  },[]);

  const clear = async (text, reset) => {
    contextsRef.current.forEach(ctx => {
        if (INVERT) {
          ctx.fillStyle = "white";
          ctx.fillRect(0, 0, SIZE, SIZE);
        } else {
          ctx.clearRect(0, 0, SIZE, SIZE)
        }
    });
    if (reset) await setRandomLabels();
    setMessage(text);
  }

  const getCanvasCoords = (event, canvas) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
    };
  };

  useEffect(() => {
    canvasesRef.current = Array.from(document.querySelectorAll(".quad"));
    contextsRef.current = canvasesRef!.current.map(canvas => {
      canvas.width = SIZE;
      canvas.height = SIZE;
      const ctx = canvas.getContext("2d");
      return ctx;
    });

    const handlers = {};

    canvasesRef.current.forEach((canvas, i) => {
      const ctx = contextsRef.current[i];

      const onPointerDown = event => {
        if (["mouse", "pen", "touch"].includes(event.pointerType)) {
          drawingRef.current[i] = true;
          const { x, y } = getCanvasCoords(event, canvas);
          ctx.strokeStyle = INVERT ? "black" : "white";
          ctx.lineWidth = Math.max(10, canvas.width / 16);
          ctx.lineCap = "round";
          ctx.lineJoin = "round";
          ctx.beginPath();
          ctx.moveTo(x, y);
          event.preventDefault();
        }
      };

      const onPointerMove = event => {
        if (drawingRef.current[i]) {
          const { x, y } = getCanvasCoords(event, canvas);
          ctx.lineTo(x, y);
          ctx.stroke();
          event.preventDefault();
        }
      };

      const onPointerUp = () => {
        drawingRef.current[i] = false;
      };

      canvas.addEventListener("pointerdown", onPointerDown);
      canvas.addEventListener("pointermove", onPointerMove);
      ["pointerup", "pointercancel", "pointerleave"].forEach(evt =>
        canvas.addEventListener(evt, onPointerUp)
      );

      handlers[i] = { onPointerDown, onPointerMove, onPointerUp };
    });

    setRandomLabels();
    setMessage("Draw a capital letter in the boxes");

    return () => {
      canvasesRef.current.forEach((canvas, i) => {
        const obj = handlers[i];
        if (!obj) return;
        canvas.removeEventListener("pointerdown", obj.onPointerDown);
        canvas.removeEventListener("pointermove", obj.onPointerMove);
        ["pointerup", "pointercancel", "pointerleave"].forEach(event =>
          canvas.removeEventListener(event, obj.onPointerUp)
        );
      });
    };
  }, [setRandomLabels]);

   const handleSubmit = async () => {
    try {
      setDisabled(true);
      setMessage("Checking");
      const tensors = canvasesRef.current.map(canvas =>{
        const img = tf.browser.fromPixels(canvas, 1).toFloat().div(255.0);
        return INVERT ? tf.sub(1.0, img) : img;
      });
      const tensorData = tensors.map(tensor => ({
        data: Array.from(new Uint8Array(tensor.mul(255).dataSync())),
        shape: tensor.shape
      }));
      const res = await getClassify(tensorData);
      tensors.forEach(tensor => tensor.dispose());
      const correct = res.every(r => r.correct);
      clear(correct ? "Correct" : "Incorrect", true);
    } catch (err) {
      console.error(err);
      setMessage("Error");
    } finally {
      setDisabled(false);
    }
  };

  return (

    <div className="hr-container d-flex flex-column" id="container">

      <h1 className="text-center mb-2">Handwritten recognition</h1>

      <div ref={quadRef} className="mb-4" id="canvas-wrapper">

        <div className="before"></div>
        <canvas className="quad"></canvas>
        <canvas className="quad"></canvas>
        <canvas className="quad"></canvas>
        <canvas className="quad"></canvas>
        <div className="after"></div>
        <div className="fill"></div>

      </div>

      <div className="d-flex flex-wrap justify-content-center mb-4">

        <button className="btn btn-success m-2 button" onClick={() => clear("Draw a capital letter in the boxes", true)}>Reset</button>

        <button className="btn btn-success m-2 button" disabled={disabled} onClick={handleSubmit} ref={predictBtnRef}>Submit</button>

      </div>

      <div className="output-container mb-3">

        <div className="label-grid">

          {labels ? labels.map((label, i) => (<Image key={i} width="125" height="60" src={label} alt="canvas"/>)): <Image className="spinner" width="250" height="125" src={Spinner} alt="spinner"/>}

        </div>

      </div>

      <div className="text-center alert alert-success p-2 w-100 mb-0" role="alert">{message}</div>

    </div>

  );

};

export default Captcha;
