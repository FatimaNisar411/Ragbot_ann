<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# RAG Bot React Project Instructions

This is a React project built with Vite and TypeScript for a RAG (Retrieval-Augmented Generation) chatbot interface.

## Project Context
- Modern React application with TypeScript
- Clean, minimal UI design with glassmorphism effects and dark mode
- Chat interface for interacting with an intelligent RAG bot
- Document upload functionality for personalized Q&A on user's own documents
- Backend handles automatic context injection and memory management
- Session-specific document management with hybrid knowledge search
- Uses modern React patterns like hooks and functional components
- Styled with custom CSS and neutral gray color scheme
- ChatGPT-inspired dark mode with smooth theme transitions

## Code Style Guidelines
- Use functional components with React hooks
- Follow TypeScript best practices
- Use proper error handling for API calls
- Implement responsive design principles
- Keep components focused and reusable
- Use semantic HTML elements for accessibility

## Backend Intelligence
- **Context Injection**: Backend automatically adds recent Q&A context to follow-up questions
- **Memory Management**: Keeps last 10 exchanges, auto-cleans older conversations
- **Smart Prompting**: LLM sees both current question + conversation history
- **Fallback Handling**: Works with or without conversation IDs seamlessly

## API Integration
- **Main endpoint**: `http://127.0.0.1:5000/ask`
- **Request format**: `{ query: string, conversation_id?: string }`
- **Response format**: `{ answer: string, sources?: string[], retrieved_docs?: string[], conversation_id?: string }`
- **Upload endpoint**: `/upload-document` with FormData for file uploads
- **Session docs endpoint**: `/session-documents` to list uploaded documents
- **Clear endpoint**: `/clear-conversation` to reset conversation context and uploaded docs
- **Frontend responsibilities**: Store conversationId, handle file uploads, distinguish source types, send with requests, update from responses
- **Backend handles**: All contextualization, memory management, document storage, and intelligent prompting
