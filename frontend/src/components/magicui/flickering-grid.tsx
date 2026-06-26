import React, { useEffect, useRef } from "react";
import { cn } from "../../lib/utils";

interface FlickeringGridProps extends React.HTMLAttributes<HTMLDivElement> {
  squareSize?: number;
  gridGap?: number;
  flickerSpeed?: number;
  maxOpacity?: number;
  color?: string; // hex color
}

export function FlickeringGrid({
  squareSize = 4,
  gridGap = 6,
  flickerSpeed = 0.5,
  maxOpacity = 0.3,
  color = "#003527", // Default to Curated Monolith primary
  className,
  ...props
}: FlickeringGridProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Convert hex to rgb for opacity manipulation
    const hexToRgb = (hex: string) => {
      const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
      return result
        ? `${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(
            result[3], 16
          )}`
        : "0, 0, 0";
    };
    const rgbColor = hexToRgb(color);

    let width = 0;
    let height = 0;
    let cols = 0;
    let rows = 0;
    let squares: Float32Array;

    const resize = () => {
      width = container.clientWidth;
      height = container.clientHeight;
      canvas.width = width;
      canvas.height = height;

      cols = Math.ceil(width / (squareSize + gridGap));
      rows = Math.ceil(height / (squareSize + gridGap));

      // Re-initialize square opacities
      squares = new Float32Array(cols * rows);
      for (let i = 0; i < squares.length; i++) {
        squares[i] = Math.random() * maxOpacity;
      }
    };

    resize();
    window.addEventListener("resize", resize);

    let animationFrameId: number;
    let lastTime = 0;

    const draw = (time: number) => {
      const deltaTime = (time - lastTime) / 1000;
      lastTime = time;

      ctx.clearRect(0, 0, width, height);

      for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
          const index = i * rows + j;
          
          // Randomly update opacities
          if (Math.random() < flickerSpeed * deltaTime) {
            squares[index] = Math.random() * maxOpacity;
          }

          const opacity = squares[index];
          if (opacity > 0) {
            ctx.fillStyle = `rgba(${rgbColor}, ${opacity})`;
            ctx.fillRect(
              i * (squareSize + gridGap),
              j * (squareSize + gridGap),
              squareSize,
              squareSize
            );
          }
        }
      }

      animationFrameId = requestAnimationFrame(draw);
    };

    animationFrameId = requestAnimationFrame(draw);

    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(animationFrameId);
    };
  }, [squareSize, gridGap, flickerSpeed, maxOpacity, color]);

  return (
    <div
      ref={containerRef}
      className={cn("absolute inset-0 w-full h-full pointer-events-none overflow-hidden", className)}
      {...props}
    >
      <canvas ref={canvasRef} className="w-full h-full block" />
    </div>
  );
}
