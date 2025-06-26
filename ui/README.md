# RAG Bot React Interface

A modern React application built with Vite and TypeScript for interacting with a RAG (Retrieval-Augmented Generation) chatbot.

## Features

- Clean, minimal chat interface
- Real-time messaging with typing indicators
- Source citation display with expandable details
- Responsive design
- TypeScript for type safety
- Modern React patterns with hooks

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

The app expects a backend API endpoint at `/ask` that accepts POST requests with the following structure:

```json
{
  "query": "Your question here"
}
```

The API should respond with:
```json
{
  "answer": "The response text",
  "sources": ["file1.pdf", "file2.txt"],
  "retrieved_docs": ["Document content snippets..."]
}
```

## Project Structure

```
src/
├── App.tsx          # Main application component
├── App.css          # Application styles
├── main.tsx         # Application entry point
└── index.css        # Global styles
```

## Technologies Used

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool and development server
- **CSS3** - Styling

## Customization

The interface is designed to be easily customizable. Key areas for modification:

- **Styling**: Update `App.css` for visual changes
- **API Integration**: Modify the `askQuestion` function in `App.tsx`
- **Message Format**: Adjust the `Message` interface and rendering logic
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default tseslint.config([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
