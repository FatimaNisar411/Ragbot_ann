import { useState, useRef, useEffect } from 'react'
import './App.css'

interface Message {
  id: string
  type: 'user' | 'bot'
  content: string
  sources?: string[]
  retrieved_docs?: string[]
  isLoading?: boolean
  isError?: boolean
}

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      type: 'bot',
      content: "Hello! I'm ready to answer questions about the documents in your knowledge base. What would you like to know?"
    }
  ])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set())
  const [isDarkMode, setIsDarkMode] = useState(() => {
    // Check if user has a saved preference, otherwise use system preference
    const savedTheme = localStorage.getItem('theme')
    if (savedTheme) {
      return savedTheme === 'dark'
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches
  })
  const chatContainerRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Apply dark mode class to body
  useEffect(() => {
    document.body.classList.toggle('dark-mode', isDarkMode)
    localStorage.setItem('theme', isDarkMode ? 'dark' : 'light')
  }, [isDarkMode])

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode)
  }

  const scrollToBottom = () => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight
    }
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const toggleSources = (messageId: string) => {
    setExpandedSources(prev => {
      const newSet = new Set(prev)
      if (newSet.has(messageId)) {
        newSet.delete(messageId)
      } else {
        newSet.add(messageId)
      }
      return newSet
    })
  }

  const askQuestion = async () => {
    const question = inputValue.trim()
    if (!question || isLoading) return

    const userMessageId = `user_${Date.now()}`
    const botMessageId = `bot_${Date.now()}`

    // Add user message
    const userMessage: Message = {
      id: userMessageId,
      type: 'user',
      content: question
    }

    // Add loading bot message
    const loadingMessage: Message = {
      id: botMessageId,
      type: 'bot',
      content: 'Thinking...',
      isLoading: true
    }

    setMessages(prev => [...prev, userMessage, loadingMessage])
    setInputValue('')
    setIsLoading(true)

    try {
      const response = await fetch('http://127.0.0.1:5000/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: question })
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const data = await response.json()

      // Replace loading message with actual response
      setMessages(prev => prev.map(msg => 
        msg.id === botMessageId 
          ? {
              ...msg,
              content: data.answer,
              sources: data.sources,
              retrieved_docs: data.retrieved_docs,
              isLoading: false
            }
          : msg
      ))

    } catch (error) {
      console.error('Error:', error)
      
      // Replace loading message with error
      setMessages(prev => prev.map(msg => 
        msg.id === botMessageId 
          ? {
              ...msg,
              content: 'Sorry, I encountered an error. Please try again.',
              isLoading: false,
              isError: true
            }
          : msg
      ))
    }

    setIsLoading(false)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      askQuestion()
    }
  }

  return (
    <div className="container">
      <div className="header">
        <button 
          className="theme-toggle" 
          onClick={toggleDarkMode}
          aria-label="Toggle dark mode"
        >
          {isDarkMode ? '☀️' : '🌙'}
        </button>
        <h1>RAG Bot</h1>
        <p>Ask questions about your documents</p>
      </div>
      
      <div className="chat-container" ref={chatContainerRef}>
        {messages.map((message) => (
          <div key={message.id} className="message">
            {message.type === 'user' ? (
              <div className="user-message">
                <div className="label">You</div>
                <div>{message.content}</div>
              </div>
            ) : (
              <div className="bot-message">
                <div className="label">Assistant</div>
                <div className={`bot-response ${message.isLoading ? 'loading' : ''} ${message.isError ? 'error' : ''}`}>
                  {message.content}
                </div>
                {((message.sources && message.sources.length > 0) || 
                  (message.retrieved_docs && message.retrieved_docs.length > 0)) && !message.isLoading && (
                  <div className="sources-toggle">
                    <button 
                      className="sources-button" 
                      onClick={() => toggleSources(message.id)}
                    >
                      {expandedSources.has(message.id) ? 'Hide sources' : 'Show sources'}
                    </button>
                  </div>
                )}
                {expandedSources.has(message.id) && !message.isLoading && (
                  <div className="sources-content show">
                    {message.sources && message.sources.length > 0 && (
                      <>
                        <h4>Source Files</h4>
                        {message.sources.map((source, index) => (
                          <div key={index} className="source-item">📄 {source}</div>
                        ))}
                      </>
                    )}
                    {message.retrieved_docs && message.retrieved_docs.length > 0 && (
                      <>
                        <h4 style={{ marginTop: '16px' }}>Referenced Content</h4>
                        {message.retrieved_docs.map((doc, index) => (
                          <div 
                            key={index} 
                            className="source-item retrieved-doc" 
                            style={{ 
                              marginBottom: '8px', 
                              padding: '8px', 
                              borderRadius: '4px', 
                              whiteSpace: 'pre-wrap', 
                              fontFamily: 'inherit' 
                            }}
                          >
                            {doc.substring(0, 200)}{doc.length > 200 ? '...' : ''}
                          </div>
                        ))}
                      </>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
      
      <div className="input-container">
        <input
          ref={inputRef}
          type="text"
          className="question-input"
          placeholder="Ask a question..."
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={isLoading}
        />
        <button 
          onClick={askQuestion} 
          className="ask-button"
          disabled={isLoading || !inputValue.trim()}
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </div>
    </div>
  )
}

export default App
