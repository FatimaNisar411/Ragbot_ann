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

interface UploadedDocument {
  filename: string
  upload_time: string
}

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      type: 'bot',
      content: "Hello! I'm ready to answer questions about the documents in your knowledge base. You can also upload your own documents using the 📎 button to ask questions about them!"
    }
  ])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set())
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [uploadedDocuments, setUploadedDocuments] = useState<UploadedDocument[]>([])
  const [isUploading, setIsUploading] = useState(false)
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

  // Fetch session documents when conversation ID changes
  useEffect(() => {
    if (conversationId) {
      fetchSessionDocuments()
    } else {
      setUploadedDocuments([])
    }
  }, [conversationId])

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

  const clearConversation = async () => {
    try {
      if (conversationId) {
        const response = await fetch('http://127.0.0.1:5000/clear-conversation', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ conversation_id: conversationId })
        })
        
        if (response.ok) {
          setConversationId(null)
          setUploadedDocuments([])
        }
      } else {
        // If no conversation ID, just reset locally
        setConversationId(null)
        setUploadedDocuments([])
      }
      
      // Reset messages to just the welcome message
      setMessages([{
        id: 'welcome',
        type: 'bot',
        content: "Hello! I'm ready to answer questions about the documents in your knowledge base. You can also upload your own documents using the 📎 button to ask questions about them!"
      }])
    } catch (error) {
      console.error('Error clearing conversation:', error)
      // Still reset locally even if the server call fails
      setConversationId(null)
      setUploadedDocuments([])
      setMessages([{
        id: 'welcome',
        type: 'bot',
        content: "Hello! I'm ready to answer questions about the documents in your knowledge base. You can also upload your own documents using the 📎 button to ask questions about them!"
      }])
    }
  }

  const uploadDocument = async (file: File) => {
    if (!file) return

    setIsUploading(true)
    
    try {
      const formData = new FormData()
      formData.append('file', file)
      if (conversationId) {
        formData.append('conversation_id', conversationId)
      }

      const response = await fetch('http://127.0.0.1:5000/upload-document', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status}`)
      }

      const data = await response.json()
      
      // Update conversation ID if provided by backend
      if (data.conversation_id) {
        setConversationId(data.conversation_id)
      }

      // Refresh uploaded documents list
      await fetchSessionDocuments()

      // Add upload success message
      const uploadMessage: Message = {
        id: `upload_${Date.now()}`,
        type: 'bot',
        content: `✅ Successfully uploaded "${file.name}". You can now ask questions about this document!`
      }
      setMessages(prev => [...prev, uploadMessage])

    } catch (error) {
      console.error('Upload error:', error)
      
      // Add upload error message
      const errorMessage: Message = {
        id: `upload_error_${Date.now()}`,
        type: 'bot',
        content: `❌ Failed to upload "${file.name}". Please try again.`,
        isError: true
      }
      setMessages(prev => [...prev, errorMessage])
    }

    setIsUploading(false)
  }

  const fetchSessionDocuments = async () => {
    if (!conversationId) return

    try {
      const response = await fetch(`http://127.0.0.1:5000/session-documents?conversation_id=${conversationId}`)
      if (response.ok) {
        const data = await response.json()
        setUploadedDocuments(data.documents || [])
      }
    } catch (error) {
      console.error('Error fetching session documents:', error)
    }
  }

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      uploadDocument(file)
      // Reset the input
      event.target.value = ''
    }
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
      const requestBody: { query: string; conversation_id?: string } = { query: question }
      if (conversationId) {
        requestBody.conversation_id = conversationId
      }

      const response = await fetch('http://127.0.0.1:5000/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const data = await response.json()

      // Update conversation ID if provided by backend
      if (data.conversation_id) {
        setConversationId(data.conversation_id)
      }

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
        <div className="header-controls">
          <button 
            className="theme-toggle" 
            onClick={toggleDarkMode}
            aria-label="Toggle dark mode"
          >
            {isDarkMode ? '☀️' : '🌙'}
          </button>
          {conversationId && (
            <button 
              className="clear-button" 
              onClick={clearConversation}
              aria-label="Clear conversation"
              title="Start a new conversation"
            >
              🗑️
            </button>
          )}
        </div>
        <h1>RAG Bot</h1>
        <p>Ask questions about your documents</p>

        {/* Uploaded Documents Display */}
        {uploadedDocuments.length > 0 && (
          <div className="uploaded-docs-display">
            <h4>📎 Uploaded Documents ({uploadedDocuments.length})</h4>
            <div className="uploaded-docs-list">
              {uploadedDocuments.map((doc, index) => (
                <div key={index} className="uploaded-doc-item">
                  📄 {doc.filename}
                  <span className="upload-time">
                    {new Date(doc.upload_time).toLocaleTimeString()}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
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
                        {message.sources.map((source, index) => {
                          // Check if this source is an uploaded document
                          const isUploaded = uploadedDocuments.some(doc => source.includes(doc.filename))
                          return (
                            <div key={index} className={`source-item ${isUploaded ? 'uploaded-source' : 'base-source'}`}>
                              {isUploaded ? '📎' : '📄'} {source}
                              {isUploaded && <span className="source-type">(uploaded)</span>}
                            </div>
                          )
                        })}
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
          type="file"
          id="file-upload"
          accept=".pdf,.txt,.md,.doc,.docx"
          onChange={handleFileChange}
          disabled={isUploading}
          style={{ display: 'none' }}
        />
        <label 
          htmlFor="file-upload" 
          className={`attach-button ${isUploading ? 'uploading' : ''}`}
          title="Attach document"
        >
          📎
        </label>
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
