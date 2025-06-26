// app/api/generate-3d/route.ts

import { NextRequest } from "next/server";
// Adjust path based on actual location
import {
  generate3DShape,
  applyTextureToMesh,
} from "../../../src/lib/hunyuan3d-integration";
import { Conceivo3DIntegration } from "/conceivin3d/ai/integrations/conceivo_3d_integration";

const conceivo = new Conceivo3DIntegration();

export async function POST(req: NextRequest) {
  try {
    const { prompt } = await req.json();

    if (!prompt) {
      return new Response(JSON.stringify({ error: "Prompt is required." }), {
        status: 400,
      });
    }

    // Use Conceivo AI to process the 3D command and generate code
    const generatedCode = await conceivo.process_3d_command(prompt);

    if (!generatedCode) {
      return new Response(
        JSON.stringify({ error: "Failed to generate 3D code from prompt." }),
        { status: 500 }
      );
    }

    // For now, we'll just return the generated code
    // In a real implementation, this would execute the code or queue a generation task
    return new Response(JSON.stringify({ code: generatedCode }), {
      status: 200,
    });
  } catch (error) {
    console.error("Error generating 3D from prompt:", error);
    return new Response(
      JSON.stringify({ error: "Failed to generate 3D from prompt." }),
      { status: 500 }
    );
  }
}
