import React from 'react'

const FONT: Record<string, string[]> = {
  A: [
    " ### ",
    "#   #",
    "#   #",
    "#####",
    "#   #",
    "#   #",
    "#   #"
  ],
  B: [
    "#### ",
    "#   #",
    "#   #",
    "#### ",
    "#   #",
    "#   #",
    "#### "
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
  D: [
    "#### ",
    "#   #",
    "#   #",
    "#   #",
    "#   #",
    "#   #",
    "#### "
  ],
  E: [
    "#####",
    "#    ",
    "#    ",
    "#### ",
    "#    ",
    "#    ",
    "#####"
  ],
  F: [
    "#####",
    "#    ",
    "#    ",
    "#### ",
    "#    ",
    "#    ",
    "#    "
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
  H: [
    "#   #",
    "#   #",
    "#   #",
    "#####",
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
  J: [
    "  ###",
    "    #",
    "    #",
    "    #",
    "#   #",
    "#   #",
    " ### "
  ],
  K: [
    "#   #",
    "#  # ",
    "# #  ",
    "##   ",
    "# #  ",
    "#  # ",
    "#   #"
  ],
  L: [
    "#    ",
    "#    ",
    "#    ",
    "#    ",
    "#    ",
    "#    ",
    "#####"
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
  N: [
    "#   #",
    "##  #",
    "# # #",
    "# # #",
    "#  ##",
    "#   #",
    "#   #"
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
  P: [
    "#### ",
    "#   #",
    "#   #",
    "#### ",
    "#    ",
    "#    ",
    "#    "
  ],
  Q: [
    " ### ",
    "#   #",
    "#   #",
    "#   #",
    "# # #",
    "#  # ",
    " ## #"
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
  S: [
    " ####",
    "#    ",
    "#    ",
    " ### ",
    "    #",
    "    #",
    "#### "
  ],
  T: [
    "#####",
    "  #  ",
    "  #  ",
    "  #  ",
    "  #  ",
    "  #  ",
    "  #  "
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
  V: [
    "#   #",
    "#   #",
    "#   #",
    "#   #",
    " # # ",
    " # # ",
    "  #  "
  ],
  W: [
    "#   #",
    "#   #",
    "#   #",
    "# # #",
    "# # #",
    "# # #",
    " # # "
  ],
  X: [
    "#   #",
    "#   #",
    " # # ",
    "  #  ",
    " # # ",
    "#   #",
    "#   #"
  ],
  Y: [
    "#   #",
    "#   #",
    " # # ",
    "  #  ",
    "  #  ",
    "  #  ",
    "  #  "
  ],
  Z: [
    "#####",
    "    #",
    "   # ",
    "  #  ",
    " #   ",
    "#    ",
    "#####"
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

export default function PixelLogo({ text = "KNOWVEX", className = "" }: { text?: string, className?: string }) {
  const letters = text.toUpperCase().split('')
  
  // Calculate total width
  // Each letter is 5 units wide, plus 1 unit gap between letters
  const width = letters.length * 5 + (letters.length - 1)
  const height = 7

  return (
    <div className={`relative ${className}`}>
      <svg 
        viewBox={`0 0 ${width} ${height}`} 
        className="w-full h-auto text-primary"
        aria-hidden="true"
        style={{ shapeRendering: 'crispEdges' }}
      >
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
                  fill="currentColor" 
                />
              ) : null
            ))
          ))
        })}
      </svg>
    </div>
  )
}
