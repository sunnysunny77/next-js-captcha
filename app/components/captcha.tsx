"use client";
import {useEffect, useRef} from "react";
import {getLabels, getClassify} from "@/lib/captcha";

const SIZE = 140;

const Captcha = () => {

  const outputRef = useRef(null);
  const messageRef = useRef(null);
  const predictBtnRef = useRef(null);
  const canvasesRef = useRef([]);
  const contextsRef = useRef([]);
  const drawingRef = useRef([false, false, false, false]);

  const setRandomLabels = async () => {
    try {
      const labels = await getLabels();
      if (outputRef.current) {
        outputRef.current.innerHTML = labels.map(l => `<div>${l}</div>`).join("");
      }
    } catch (err) {
      console.error(err);
      if (messageRef.current) messageRef.current.innerText = "Error fetching labels";
    }
  };

  const clear = async (text, reset) => {
    contextsRef.current.forEach(ctx => {
      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, SIZE, SIZE);
    });
    if (reset) await setRandomLabels();
    if (messageRef.current) messageRef.current.innerText = text;
  };

   const getCanvasCoords = (e, canvas) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  useEffect(() => {
    canvasesRef.current = Array.from(document.querySelectorAll(".quad"));
    contextsRef.current = canvasesRef.current.map(c => {
      c.width = SIZE;
      c.height = SIZE;
      const ctx = c.getContext("2d");
      if (!ctx) throw new Error("Canvas context not found");
      return ctx;
    });

    const handlers: { [key: number]: { [key: string]: EventListener } } = {};

    canvasesRef.current.forEach((canvas, i) => {
      const ctx = contextsRef.current[i];

      const onPointerDown = (e: PointerEvent) => {
        if (["mouse", "pen", "touch"].includes(e.pointerType)) {
          drawingRef.current[i] = true;
          const { x, y } = getCanvasCoords(e, canvas);
          ctx.strokeStyle = "white";
          ctx.lineWidth = Math.max(10, canvas.width / 16);
          ctx.lineCap = "round";
          ctx.lineJoin = "round";
          ctx.beginPath();
          ctx.moveTo(x, y);
          e.preventDefault();
        }
      };

      const onPointerMove = (e: PointerEvent) => {
        if (drawingRef.current[i]) {
          const { x, y } = getCanvasCoords(e, canvas);
          ctx.lineTo(x, y);
          ctx.stroke();
          e.preventDefault();
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
        const h = handlers[i];
        if (h) {
          canvas.removeEventListener("pointerdown", h.onPointerDown);
          canvas.removeEventListener("pointermove", h.onPointerMove);
          ["pointerup", "pointercancel", "pointerleave"].forEach(evt =>
            canvas.removeEventListener(evt, h.onPointerUp)
          );
        }
      });
    };
  }, []);

  const handleSubmit = async () => {
    try {
      if (predictBtnRef.current) predictBtnRef.current.disabled = true;
      if (messageRef.current) messageRef.current.innerText = "Checking...";

      const images = canvasesRef.current.map(c => c.toDataURL("image/png").split(",")[1]);
      const results = await getClassify(images);

      let allCorrect = results.every(p => p.predictedLabel === p.correctLabel);

      if (messageRef.current)
        messageRef.current.innerText = allCorrect ? "All Correct!" : "Some answers are incorrect";
    } catch (err) {
      console.error(err);
      if (messageRef.current) messageRef.current.innerText = "Error";
    } finally {
      if (predictBtnRef.current) predictBtnRef.current.disabled = false;
    }
  };

  return (

    <div className="container" id="container">

      <h1>Handwritten recognition</h1>

      <div id="canvas-wrapper">

        <canvas className="quad"></canvas>
        <canvas className="quad"></canvas>
        <canvas className="quad"></canvas>
        <canvas className="quad"></canvas>

      </div>

      <div className="buttons">

        <button onClick={() => clear("Draw the required characters", true)}>Reset</button>

        <button onClick={handleSubmit} ref={predictBtnRef}>Submit</button>

      </div>

      <div ref={outputRef} className="label-grid"></div>

      <div ref={messageRef}>Loading model...</div>

    </div>

  );

};

export default Captcha;
