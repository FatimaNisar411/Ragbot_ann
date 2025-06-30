# RAG Bot React Interface

A modern React application built with Vite and TypeScript for interacting with a RAG (Retrieval-Augmented Generation) chatbot with intelligent conversation context management.

## Features

- **Smart Conversation Management**: Automatic context injection and memory management
- **Document Upload**: Upload your own PDF, TXT, MD, DOC, or DOCX files for personalized Q&A
- **Hybrid Knowledge**: Combines base knowledge with your uploaded documents seamlessly
- **Session Isolation**: Each conversation has its own uploaded documents
- **Clean Chat Interface**: Modern, minimal design with glassmorphism effects
- **Dark Mode**: ChatGPT-inspired dark theme with smooth transitions
- **Enhanced Source Citations**: Distinguishes between uploaded documents (📎) and base documents (📄)
- **Context Persistence**: Conversations maintain context across multiple questions
- **Conversation Reset**: Clear conversation history and uploaded documents
- **Responsive Design**: Works seamlessly on desktop and mobile
- **TypeScript**: Full type safety and modern React patterns
- **Real-time Feedback**: Loading states, error handling, and upload progress

## Backend Intelligence

The application leverages intelligent backend features for enhanced conversation quality:

- **Context Injection**: Backend automatically adds recent Q&A context to follow-up questions
- **Memory Management**: Keeps last 10 exchanges, auto-cleans older conversations  
- **Smart Prompting**: LLM sees both current question + conversation history
- **Fallback Handling**: Works seamlessly with or without conversation IDs
- **Automatic Cleanup**: Backend manages conversation lifecycle intelligently

## Getting Started

### Prerequisites

- Node.js (v20.19.0 or higher recommended)
- npm or yarn

### Installation

1. Clone the repository or navigate to the project directory
2. Install dependencies:
   ```bash
   npm install
   ```

### Development

Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:5173`

### Building for Production

Create a production build:
```bash
npm run build
```

Preview the production build:
```bash
npm run preview
```

## API Integration

The app integrates with a backend API that provides intelligent conversation management and document upload capabilities:

### Main Endpoint: `/ask`
```json
POST http://127.0.0.1:5000/ask
{
  "query": "Your question here",
  "conversation_id": "optional-conversation-id"
}
```

**Response:**
```json
{
  "answer": "The response text",
  "sources": ["file1.pdf", "📎 uploaded_doc.pdf"],
  "retrieved_docs": ["Document content snippets..."],
  "conversation_id": "generated-or-existing-conversation-id"
}
```

### Document Upload: `/upload-document`
```json
POST http://127.0.0.1:5000/upload-document
FormData with:
- file: The document file
- conversation_id: (optional) existing conversation ID
```

**Response:**
```json
{
  "message": "Document uploaded successfully",
  "filename": "uploaded_doc.pdf",
  "conversation_id": "conversation-id"
}
```

### Session Documents: `/session-documents`
```json
GET http://127.0.0.1:5000/session-documents?conversation_id=conv_id
```

**Response:**
```json
{
  "documents": [
    {
      "filename": "uploaded_doc.pdf",
      "upload_time": "2025-06-30T12:00:00Z"
    }
  ]
}
```

### Clear Conversation: `/clear-conversation`
```json
POST http://127.0.0.1:5000/clear-conversation
{
  "conversation_id": "conversation-id-to-clear"
}
```

### Frontend Responsibilities
- Store and manage conversation IDs in React state
- Handle file uploads with FormData
- Display uploaded vs base document sources differently
- Send conversation IDs with each request for context continuity
- Update conversation IDs from backend responses
- Handle conversation clearing and document cleanup

### Backend Intelligence
- Automatically injects recent Q&A context into follow-up questions
- Manages conversation memory (last 10 exchanges)
- Provides intelligent prompting with conversation history
- Handles session-specific document upload and storage
- Combines base knowledge with uploaded documents in responses
- Provides hybrid search across both base and uploaded documents
- Handles conversation lifecycle and cleanup automatically

## Project Structure

```
src/
├── App.tsx          # Main application component
├── App.css          # Application styles
├── main.tsx         # Application entry point
└── index.css        # Global styles
```

## Technologies Used

- **React 18** - UI library with modern hooks and functional components
- **TypeScript** - Type safety and enhanced developer experience
- **Vite** - Fast build tool and development server
- **CSS3** - Modern styling with glassmorphism effects and smooth transitions

## Development Notes

- The application automatically handles conversation context through backend intelligence
- Dark mode preference is persisted in localStorage
- Conversation IDs are managed automatically for seamless context continuity
- The clear conversation button (🗑️) appears only when a conversation is active
- All API calls include proper error handling and loading states
- Document upload supports PDF, TXT, MD, DOC, and DOCX files
- Uploaded documents are session-specific and automatically cleaned up
- Source citations clearly distinguish between uploaded (📎) and base (📄) documents

## Usage

1. **Start a conversation**: Ask any question about the base knowledge
2. **Upload documents**: Click the 📎 button to upload your own documents
3. **Ask about uploads**: Questions will search both base knowledge and your uploaded files
4. **View sources**: Expand sources to see which documents were referenced
5. **Clear session**: Use the 🗑️ button to reset conversation and remove uploaded documents
