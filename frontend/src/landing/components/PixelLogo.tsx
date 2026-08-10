import React from 'react'

const FONT: Record<string, string[]> = {
  D: [
    "#### ",
    "#   #",
    "#   #",
    "#   #",
    "#   #",
    "#   #",
    "#### "
  ],
  O: [
    " ### ",
    "#   #",
    "#   #",
    "#   #",
    "#   #",
    "#   #",
    " ### "
  ],
  C: [
    " ####",
    "#    ",
    "#    ",
    "#    ",
    "#    ",
    "#    ",
    " ####"
  ],
  U: [
    "#   #",
    "#   #",
    "#   #",
    "#   #",
    "#   #",
    "#   #",
    " ### "
  ],
  M: [
    "#   #",
    "## ##",
    "# # #",
    "#   #",
    "#   #",
    "#   #",
    "#   #"
  ],
  I: [
    " ### ",
    "  #  ",
    "  #  ",
    "  #  ",
    "  #  ",
    "  #  ",
    " ### "
  ],
  N: [
    "#   #",
    "##  #",
    "# # #",
    "# # #",
    "#  ##",
    "#   #",
    "#   #"
  ],
  R: [
    "#### ",
    "#   #",
    "#   #",
    "#### ",
    "# #  ",
    "#  # ",
    "#   #"
  ],
  A: [
    " ### ",
    "#   #",
    "#   #",
    "#####",
    "#   #",
    "#   #",
    "#   #"
  ],
  G: [
    " ####",
    "#    ",
    "#    ",
    "# ###",
    "#   #",
    "#   #",
    " ####"
  ],
  ' ': [
    "     ",
    "     ",
    "     ",
    "     ",
    "     ",
    "     ",
    "     "
  ]
}

export default function PixelLogo({ text = "DOCUMIND", className = "" }: { text?: string, className?: string }) {
  const letters = text.toUpperCase().split('')
  
  // Calculate total width
  // Each letter is 5 units wide, plus 1 unit gap between letters
  const width = letters.length * 5 + (letters.length - 1)
  const height = 7

  return (
    <div className={`relative ${className}`}>
      <svg 
        viewBox={`0 0 ${width} ${height}`} 
        className="w-full h-auto drop-shadow-[0_0_20px_rgba(99,102,241,0.6)]"
        style={{ shapeRendering: 'crispEdges' }}
      >
        <defs>
          <linearGradient id="pixelGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#60a5fa" /> {/* blue-400 */}
            <stop offset="50%" stopColor="#c084fc" /> {/* purple-400 */}
            <stop offset="100%" stopColor="#60a5fa" /> {/* blue-400 */}
          </linearGradient>
        </defs>
        
        {letters.map((char, charIdx) => {
          const pattern = FONT[char] || FONT[' ']
          const xOffset = charIdx * 6 // 5 width + 1 gap
          
          return pattern.map((row, y) => (
            row.split('').map((pixel, x) => (
              pixel === '#' ? (
                <rect 
                  key={`${charIdx}-${x}-${y}`}
                  x={xOffset + x} 
                  y={y} 
                  width="0.9" // 0.9 width + 0.1 gap for distinct square blocks
                  height="0.9" 
                  fill="url(#pixelGradient)" 
                />
              ) : null
            ))
          ))
        })}
      </svg>
    </div>
  )
}
