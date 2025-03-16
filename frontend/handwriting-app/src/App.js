import React, { useRef, useState } from "react";
import html2canvas from "html2canvas"; // html2canvasのインポート

const Canvas = () => {
  const canvasRef = useRef(null); // Canvasを参照
  const [isDrawing, setIsDrawing] = useState(false); // 描画状態

  // 描画を開始
  const startDrawing = (e) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.beginPath();
    ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    setIsDrawing(true);
  };

  // 描画を停止
  const stopDrawing = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.closePath();
    setIsDrawing(false);
  };

  // 描画を続ける
  const draw = (e) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    ctx.stroke();
  };

  // クリアボタン
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  // ダウンロードボタン (手書き画像をダウンロード)
  const downloadCanvas = () => {
    const canvas = canvasRef.current;
    html2canvas(canvas).then((canvasImage) => {
      const link = document.createElement("a");
      link.href = canvasImage.toDataURL(); // 手書きの画像をData URLとして取得
      link.download = "handwritten_image.png"; // 画像としてダウンロード
      link.click();
    });
  };

  return (
    <div>
      <canvas
        ref={canvasRef}
        width="300"
        height="300"
        style={{ border: "1px solid black" }}
        onMouseDown={startDrawing}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
        onMouseMove={draw}
      />
      <div>
        <button onClick={clearCanvas}>Clear</button>
        <button onClick={downloadCanvas}>Download</button>
      </div>
    </div>
  );
};

const App = () => {
  return (
    <div>
      <h1>Handwriting Canvas</h1>
      <Canvas />
    </div>
  );
};

export default App;
