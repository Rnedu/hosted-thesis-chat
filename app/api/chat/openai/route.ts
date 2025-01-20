import { checkApiKey, getServerProfile } from "@/lib/server/server-chat-helpers";
import { ChatSettings } from "@/types";
import { OpenAIStream, StreamingTextResponse } from "ai";
import { ServerRuntime } from "next";
import OpenAI from "openai";
import { ChatCompletionCreateParamsBase } from "openai/resources/chat/completions.mjs";
import { PineconeClient } from "@pinecone-database/pinecone";

export const runtime: ServerRuntime = "edge";

// Initialize Pinecone client
const pinecone = new PineconeClient();
await pinecone.init({
  apiKey: process.env.PINECONE_API_KEY || "", // Add your Pinecone API key here
  environment: process.env.PINECONE_ENVIRONMENT || "", // Add your Pinecone environment
});
const index = pinecone.Index("thesis"); // Replace with your Pinecone index name

async function retrieveContext(query: string): Promise<string> {
  const embeddingModel = "text-embedding-ada-002"; // OpenAI's embedding model
  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY || "",
  });
  const embeddingResponse = await openai.embeddings.create({
    model: embeddingModel,
    input: query,
  });

  const vector = embeddingResponse.data[0].embedding;
  
  const searchResponse = await index.query({
    vector,
    topK: 5, // Number of relevant documents to retrieve
    includeMetadata: true,
  });

  const context = searchResponse.matches
    .map((match) => match.metadata?.content || "")
    .join("\n");

  return context;
}

export async function POST(request: Request) {
  const json = await request.json();
  const { chatSettings, messages } = json as {
    chatSettings: ChatSettings;
    messages: any[];
  };

  try {
    const profile = await getServerProfile();

    checkApiKey(profile.openai_api_key, "OpenAI");

    const openai = new OpenAI({
      apiKey: profile.openai_api_key || "",
      organization: profile.openai_organization_id,
    });

    const queryMessage = messages[messages.length - 1]?.content || "";
    const context = await retrieveContext(queryMessage); // Retrieve relevant context

    const systemMessage = {
      role: "system",
      content: `You are a Socratic tutor. Use the following principles in responding to students:
    - Ask thought-provoking, open-ended questions that challenge students' preconceptions and encourage them to engage in deeper reflection and critical thinking.
    - Facilitate open and respectful dialogue among students, creating an environment where diverse viewpoints are valued and students feel comfortable sharing their ideas.
    - Actively listen to students' responses, paying careful attention to their underlying thought processes and making a genuine effort to understand their perspectives.
    - Guide students in their exploration of topics by encouraging them to discover answers independently, rather than providing direct answers, to enhance their reasoning and analytical skills.
    - Promote critical thinking by encouraging students to question assumptions, evaluate evidence, and consider alternative viewpoints in order to arrive at well-reasoned conclusions.
    - Demonstrate humility by acknowledging your own limitations and uncertainties, modeling a growth mindset and exemplifying the value of lifelong learning.
    - Incorporate the following contextual information:\n\n${context}`,
    };

    const updatedMessages = [systemMessage, ...messages.filter((msg) => msg.role !== "system")];

    const response = await openai.chat.completions.create({
      model: chatSettings.model as ChatCompletionCreateParamsBase["model"],
      messages: updatedMessages as ChatCompletionCreateParamsBase["messages"],
      temperature: chatSettings.temperature,
      max_tokens:
        chatSettings.model === "gpt-4-vision-preview" || chatSettings.model === "gpt-4o"
          ? 4096
          : null,
      stream: true,
    });

    const stream = OpenAIStream(response);

    return new StreamingTextResponse(stream);
  } catch (error: any) {
    let errorMessage = error.message || "An unexpected error occurred";
    const errorCode = error.status || 500;

    if (errorMessage.toLowerCase().includes("api key not found")) {
      errorMessage = "OpenAI API Key not found. Please set it in your profile settings.";
    } else if (errorMessage.toLowerCase().includes("incorrect api key")) {
      errorMessage = "OpenAI API Key is incorrect. Please fix it in your profile settings.";
    }

    return new Response(JSON.stringify({ message: errorMessage }), {
      status: errorCode,
    });
  }
}
