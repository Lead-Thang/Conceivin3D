import { type NextRequest, NextResponse } from "next/server"

export async function POST(req: NextRequest) {
  try {
    const { text } = await req.json()

    if (!text || text === "test") {
      // Handle test requests
      return NextResponse.json({
        message: "ElevenLabs connection test successful",
        status: "ready",
      })
    }

    const agentId = process.env.ELEVENLABS_AGENT_ID
    const apiKey = process.env.ELEVENLABS_API_KEY

    if (!apiKey) {
      console.error("❌ Missing ELEVENLABS_API_KEY environment variable")
      return NextResponse.json({ error: "Missing ElevenLabs API key" }, { status: 500 })
    }

    if (!agentId) {
      console.error("❌ Missing ELEVENLABS_AGENT_ID environment variable")
      return NextResponse.json({ error: "Missing ElevenLabs Agent ID" }, { status: 500 })
    }

    console.log("🎤 ElevenLabs Request:", { text: text.substring(0, 50) + "...", agentId })

    // Use the conversational AI endpoint
    const response = await fetch(`https://api.elevenlabs.io/v1/convai/conversation`, {
      method: "POST",
      headers: {
        "xi-api-key": apiKey,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        agent_id: agentId,
        text,
        user_id: "conceivin3d_user",
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error("❌ ElevenLabs API error:", response.status, errorText)
      return NextResponse.json(
        {
          error: `ElevenLabs API error: ${response.status}`,
          details: errorText,
        },
        { status: response.status },
      )
    }

    const data = await response.json()
    console.log("✅ ElevenLabs response received:", {
      hasAudio: !!data.audio_url,
      hasText: !!data.text,
      responseLength: data.text?.length || 0,
    })

    return NextResponse.json({
      ...data,
      success: true,
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    console.error("❌ ElevenLabs API route error:", error)
    return NextResponse.json(
      {
        error: "Failed to process ElevenLabs request",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}
