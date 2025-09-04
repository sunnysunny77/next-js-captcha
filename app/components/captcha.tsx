"use client";
import {useEffect, useRef, useState} from "react";
import {getLabels, getClassify} from "@/lib/captcha";

const SIZE = 140;

const Captcha = () => {

  const quadRef = useRef(null);
  const predictBtnRef = useRef(null);
  const canvasesRef = useRef([]);
  const contextsRef = useRef([]);
  const drawingRef = useRef([false, false, false, false]);

  const [labels, setLabels] = useState([]);
  const [message, setMessage] = useState("Loading model...");
  const [disabled, setDisabled] = useState(false);

  const setRandomLabels = async () => {
    try {
      const labels = await getLabels();
      setLabels(labels);
    } catch (err) {
      console.error(err);
      setMessage("Error fetching labels");
    }
  };

  const clear = async (text, reset) => {
    contextsRef.current.forEach(ctx => {
      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, SIZE, SIZE);
    });
    if (reset) await setRandomLabels();
    setMessage(text);
  };

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
    canvasesRef.current = Array.from(quadRef!.current!.children);
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
          ctx.strokeStyle = "white";
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

    clear("Draw the required characters", true);

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
  }, []);

  const handleSubmit = async () => {
    try {
      setDisabled(true);
      setMessage("Checking...");

      const images = canvasesRef.current.map(c => c.toDataURL("image/png").split(",")[1]);
      const results = await getClassify(images);

      let correct = results.every(prediction => prediction.predictedLabel === prediction.correctLabel);

      correct ? setMessage("Correct") : setMessage("Incorrect");
    } catch (err) {
      console.error(err);
      setMessage("Error");
    } finally {
      setDisabled(false);
    }
  };

  return (

    <div className="container" id="container">

      <h1>Handwritten recognition</h1>

      <div ref={quadRef} id="canvas-wrapper">

        <canvas className="quad"></canvas>
        <canvas className="quad"></canvas>
        <canvas className="quad"></canvas>
        <canvas className="quad"></canvas>

      </div>

      <div className="buttons">

        <button onClick={() => clear("Draw the required characters", true)}>Reset</button>

        <button disabled={disabled} onClick={handleSubmit} ref={predictBtnRef}>Submit</button>

      </div>

      <div className="label-grid">

        {labels.map((label, i) => (<div key={i}>{label}</div>))}

      </div>

      <div>{message}</div>

    </div>

  );

};

export default Captcha;
