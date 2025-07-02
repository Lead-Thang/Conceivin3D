import { type NextRequest, NextResponse } from "next/server"

export async function POST(req: NextRequest) {
  try {
    // Parse request body safely
    let body
    try {
      body = await req.json()
    } catch (parseError) {
      console.error("Failed to parse request body:", parseError)
      return NextResponse.json({ error: "Invalid JSON in request body" }, { status: 400 })
    }

    const { messages, userId } = body

    console.log({
      message: "Grok AI Assistant Request Received",
      userId: userId || "anonymous",
      timestamp: new Date().toISOString(),
      messageCount: Array.isArray(messages) ? messages.length : 0,
    })

    // Validate messages array
    if (!Array.isArray(messages)) {
      return NextResponse.json({ error: "Messages must be an array" }, { status: 400 })
    }

    // Get the latest message safely
    const latestMessage = messages.length > 0 ? messages[messages.length - 1]?.content || "" : ""

    if (!latestMessage.trim()) {
      return NextResponse.json({
        message: "I didn't receive a message. Could you please try again?",
        timestamp: new Date().toISOString(),
      })
    }

    // Try to use Grok API if available, otherwise fallback to simple responses
    let response: string

    if (process.env.XAI_API_KEY) {
      try {
        response = await generateGrokResponse(latestMessage, messages)
      } catch (grokError) {
        console.warn("Grok API failed, using fallback:", grokError)
        response = generateSimpleResponse(latestMessage)
      }
    } else {
      response = generateSimpleResponse(latestMessage)
    }

    // Return successful response
    return NextResponse.json({
      message: response,
      timestamp: new Date().toISOString(),
      success: true,
      powered_by: process.env.XAI_API_KEY ? "Grok AI" : "Fallback AI",
    })
  } catch (error) {
    console.error("AI Assistant error:", {
      message: error instanceof Error ? error.message : "Unknown error",
      stack: error instanceof Error ? error.stack : undefined,
      timestamp: new Date().toISOString(),
    })

    return NextResponse.json(
      {
        error: "Internal server error",
        message: "I'm having trouble processing your request. Please try again.",
        timestamp: new Date().toISOString(),
      },
      { status: 500 },
    )
  }
}

async function generateGrokResponse(message: string, messages: any[]): Promise<string> {
  const response = await fetch("https://api.x.ai/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.XAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "grok-beta",
      messages: [
        {
          role: "system",
          content: `You are Conceivo, the AI assistant for Conceivin3D — a professional 3D design platform.

You are an expert in 3D modeling, CAD operations, engineering design, and product development. 

When users ask about 3D modeling actions, provide helpful guidance and suggest specific steps they can take in the 3D modeling interface.

Keep responses concise but informative. Be encouraging and provide practical, actionable advice.

Focus on:
- 3D modeling techniques
- CAD operations (extrude, revolve, sweep, loft)
- Design optimization
- Material selection
- Manufacturing considerations
- Best practices

Always be helpful, professional, and encouraging.`,
        },
        ...messages.slice(-5), // Include last 5 messages for context
      ],
      max_tokens: 500,
      temperature: 0.7,
    }),
  })

  if (!response.ok) {
    throw new Error(`Grok API error: ${response.status}`)
  }

  const data = await response.json()
  return data.choices[0]?.message?.content || "I'm having trouble generating a response right now."
}

function generateSimpleResponse(message: string): string {
  const lowerMessage = message.toLowerCase().trim()

  // Handle empty or very short messages
  if (lowerMessage.length < 2) {
    return "Could you please provide more details about what you'd like to do?"
  }

  // 3D modeling responses
  if (lowerMessage.includes("add") || lowerMessage.includes("create")) {
    if (lowerMessage.includes("box") || lowerMessage.includes("cube")) {
      return "I'll help you add a box to your scene. You can use the toolbar on the left or click the box tool in the Primitives section. The box will appear at a random position and you can modify it using the Properties panel on the right."
    }
    if (lowerMessage.includes("sphere") || lowerMessage.includes("ball")) {
      return "Great! To add a sphere, look for the sphere tool in the Primitives section of the toolbar. Once created, you can adjust its position, scale, and color in the Properties panel."
    }
    if (lowerMessage.includes("cylinder")) {
      return "I'll help you create a cylinder. Find the cylinder tool in the Primitives section. You can then modify its height, radius, and other properties using the controls on the right."
    }
    return "I can help you add 3D objects like boxes, spheres, cylinders, cones, and torus shapes. What would you like to create? You can find all the tools in the Primitives section of the toolbar."
  }

  if (lowerMessage.includes("delete") || lowerMessage.includes("remove")) {
    return "To delete an object: 1) Click on the object to select it (it will show a cyan wireframe), 2) Use the Delete button in the Properties panel on the right, or 3) Use voice commands like 'delete this'."
  }

  if (lowerMessage.includes("color") || lowerMessage.includes("material")) {
    return "To change colors: 1) Select an object by clicking on it, 2) In the Properties panel on the right, find the Color section, 3) Click the color picker to choose a new color. You can also use voice commands like 'make this red' or 'change color to blue'."
  }

  if (lowerMessage.includes("help") || lowerMessage.includes("how")) {
    return "I'm here to help with 3D modeling! Here's what you can do:\n\n• **Create**: Add objects using the Primitives toolbar\n• **Select**: Click objects to select them\n• **Modify**: Use the Properties panel to change position, scale, rotation, and color\n• **Transform**: Use tools for moving, rotating, and scaling\n• **Voice**: Try commands like 'add a red sphere' or 'make this bigger'\n\nWhat would you like to learn about?"
  }

  // Greetings
  if (lowerMessage.includes("hello") || lowerMessage.includes("hi") || lowerMessage.includes("hey")) {
    return "Hello! I'm your Conceivin3D assistant powered by Grok AI. I'm here to help you with 3D modeling, creating objects, and using the tools. What would you like to create today?"
  }

  // Default response
  return "I'm your 3D modeling assistant powered by Grok AI! I can help you create objects, modify properties, and use the tools. What would you like to do? Try asking about adding shapes, changing colors, or using specific tools."
}

// Handle CORS preflight requests
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization",
    },
  })
}
