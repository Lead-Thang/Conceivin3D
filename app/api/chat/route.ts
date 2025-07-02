import { streamText } from "ai"
import { createOpenAI } from "@ai-sdk/openai"
import type { NextRequest } from "next/server"

// Create xAI/Grok client
const xai = createOpenAI({
  apiKey: process.env.XAI_API_KEY,
  baseURL: "https://api.x.ai/v1",
})

export async function POST(req: NextRequest) {
  try {
    const { messages, userId } = await req.json()

    console.log({
      message: "Grok AI Assistant Request Received",
      userId,
      timestamp: new Date().toISOString(),
      messageCount: messages.length,
    })

    const result = await streamText({
      model: xai("grok-beta"),
      system: `You are Conceivo, the AI assistant for Conceivin3D — a professional 3D design platform powered by Grok AI.

You are an expert in:
1. 3D modeling techniques and CAD operations
2. Engineering design principles
3. Product development and prototyping
4. Materials science and manufacturing
5. Design optimization and analysis
6. 3D printing and fabrication
7. Technical drawing and documentation

When users ask about 3D modeling actions, you can provide both guidance and structured commands:

**Structured Commands** (for direct execution):
- {"action": "add", "type": "box", "position": [x, y, z], "color": "#hex"}
- {"action": "delete"}
- {"action": "scale", "factor": 1.2}
- {"action": "color", "color": "#hex"}
- {"action": "move", "axis": "x", "value": 2.0}

**Supported Object Types**: box, sphere, cylinder, cone, torus, plane, wedge

**Your Personality**:
- Professional yet approachable
- Encouraging and supportive
- Detail-oriented with practical advice
- Focus on both form and function
- Emphasize best practices and industry standards

Always provide specific, actionable guidance for 3D modeling tasks. When discussing technical concepts, explain them clearly and suggest concrete steps users can take.`,
      messages,
    })

    return result.toDataStreamResponse()
  } catch (error) {
    console.error({
      message: "Grok AI Assistant error",
      error: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
      timestamp: new Date().toISOString(),
    })

    return new Response(
      JSON.stringify({
        error: "Failed to process request with Grok AI",
        details: error instanceof Error ? error.message : "Unknown error",
      }),
      {
        status: 500,
        headers: {
          "Content-Type": "application/json",
        },
      },
    )
  }
}

export async function OPTIONS() {
  return new Response(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization",
    },
  })
}
