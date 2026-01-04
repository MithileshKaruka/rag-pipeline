import { useState, useEffect } from 'react';
import axios from 'axios';
import { API_ENDPOINTS } from './config';
import './App.css';

function App() {
  // State management
  const [activeTab, setActiveTab] = useState('query'); // 'query' or 'ingest'
  const [healthStatus, setHealthStatus] = useState(null);
  const [collections, setCollections] = useState([]);
  const [selectedCollection, setSelectedCollection] = useState('knowledge_base');

  // Query tab state
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState([]);
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryError, setQueryError] = useState('');

  // Ingest tab state
  const [ingestMode, setIngestMode] = useState('text'); // 'text' or 'file'
  const [ingestText, setIngestText] = useState('');
  const [ingestFile, setIngestFile] = useState(null);
  const [ingestCollection, setIngestCollection] = useState('knowledge_base');
  const [ingestLoading, setIngestLoading] = useState(false);
  const [ingestSuccess, setIngestSuccess] = useState('');
  const [ingestError, setIngestError] = useState('');

  // New collection state
  const [newCollectionName, setNewCollectionName] = useState('');
  const [showNewCollection, setShowNewCollection] = useState(false);

  // Check backend health on component mount
  useEffect(() => {
    checkHealth();
    loadCollections();
  }, []);

  /**
   * Check backend API health
   */
  const checkHealth = async () => {
    try {
      const response = await axios.get(API_ENDPOINTS.HEALTH);
      setHealthStatus(response.data);
    } catch (err) {
      console.error('Health check failed:', err);
      setHealthStatus({ status: 'error', message: err.message });
    }
  };

  /**
   * Load available collections
   */
  const loadCollections = async () => {
    try {
      const response = await axios.get(API_ENDPOINTS.COLLECTIONS);
      setCollections(response.data);
      if (response.data.length > 0) {
        setSelectedCollection(response.data[0]);
        setIngestCollection(response.data[0]);
      }
    } catch (err) {
      console.error('Failed to load collections:', err);
    }
  };

  /**
   * Create new collection
   */
  const handleCreateCollection = async () => {
    if (!newCollectionName.trim()) {
      setIngestError('Please enter a collection name');
      return;
    }

    try {
      await axios.post(API_ENDPOINTS.CREATE_COLLECTION, {
        collection_name: newCollectionName
      });
      setIngestSuccess(`Collection '${newCollectionName}' created successfully`);
      setNewCollectionName('');
      setShowNewCollection(false);
      loadCollections();
      setTimeout(() => setIngestSuccess(''), 3000);
    } catch (err) {
      setIngestError(err.response?.data?.detail || 'Failed to create collection');
      setTimeout(() => setIngestError(''), 3000);
    }
  };

  /**
   * Submit query to RAG backend
   */
  const handleQuery = async (e) => {
    e.preventDefault();

    if (!question.trim()) {
      setQueryError('Please enter a question');
      return;
    }

    setQueryLoading(true);
    setQueryError('');
    setAnswer('');
    setSources([]);

    try {
      const response = await axios.post(API_ENDPOINTS.QUERY, {
        question: question,
        collection_name: selectedCollection,
        n_results: 3
      });

      setAnswer(response.data.answer);
      setSources(response.data.sources);
    } catch (err) {
      console.error('Query error:', err);
      setQueryError(
        err.response?.data?.detail ||
        err.message ||
        'Failed to get answer. Please try again.'
      );
    } finally {
      setQueryLoading(false);
    }
  };

  /**
   * Handle text ingestion
   */
  const handleIngestText = async (e) => {
    e.preventDefault();

    if (!ingestText.trim()) {
      setIngestError('Please enter text to ingest');
      return;
    }

    setIngestLoading(true);
    setIngestError('');
    setIngestSuccess('');

    try {
      const response = await axios.post(API_ENDPOINTS.INGEST_TEXT, {
        text: ingestText,
        collection_name: ingestCollection,
        chunk_size: 1000,
        chunk_overlap: 200
      });

      setIngestSuccess(response.data.message);
      setIngestText('');
      setTimeout(() => setIngestSuccess(''), 5000);
    } catch (err) {
      console.error('Ingest error:', err);
      setIngestError(
        err.response?.data?.detail ||
        err.message ||
        'Failed to ingest text. Please try again.'
      );
    } finally {
      setIngestLoading(false);
    }
  };

  /**
   * Handle file ingestion
   */
  const handleIngestFile = async (e) => {
    e.preventDefault();

    if (!ingestFile) {
      setIngestError('Please select a file to upload');
      return;
    }

    setIngestLoading(true);
    setIngestError('');
    setIngestSuccess('');

    try {
      const formData = new FormData();
      formData.append('file', ingestFile);
      formData.append('collection_name', ingestCollection);
      formData.append('chunk_size', '1000');
      formData.append('chunk_overlap', '200');

      const response = await axios.post(API_ENDPOINTS.INGEST_FILE, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setIngestSuccess(response.data.message);
      setIngestFile(null);
      // Reset file input
      document.getElementById('file-input').value = '';
      setTimeout(() => setIngestSuccess(''), 5000);
    } catch (err) {
      console.error('File ingest error:', err);
      setIngestError(
        err.response?.data?.detail ||
        err.message ||
        'Failed to ingest file. Please try again.'
      );
    } finally {
      setIngestLoading(false);
    }
  };

  /**
   * Clear query results
   */
  const handleClearQuery = () => {
    setQuestion('');
    setAnswer('');
    setSources([]);
    setQueryError('');
  };

  // Render health status indicator
  const renderHealthStatus = () => {
    if (!healthStatus) return null;

    const statusColor = healthStatus.status === 'healthy' ? '#4caf50' : '#ff9800';

    return (
      <div className="health-status" style={{ borderColor: statusColor }}>
        <span className="status-dot" style={{ backgroundColor: statusColor }}></span>
        <span>Backend: {healthStatus.status}</span>
        {healthStatus.model && <span className="model-badge">Model: {healthStatus.model}</span>}
      </div>
    );
  };

  return (
    <div className="App">
      {/* Header */}
      <header className="app-header">
        <h1>🤖 RAG Application</h1>
        <p>Query your knowledge base or ingest new documents</p>
        {renderHealthStatus()}
      </header>

      {/* Tab Navigation */}
      <div className="tab-navigation">
        <button
          className={`tab-button ${activeTab === 'query' ? 'active' : ''}`}
          onClick={() => setActiveTab('query')}
        >
          🔍 Query
        </button>
        <button
          className={`tab-button ${activeTab === 'ingest' ? 'active' : ''}`}
          onClick={() => setActiveTab('ingest')}
        >
          📤 Ingest
        </button>
      </div>

      {/* Main Content */}
      <main className="app-main">
        {/* Query Tab */}
        {activeTab === 'query' && (
          <div className="tab-content">
            <h2>Ask Questions</h2>

            <form onSubmit={handleQuery} className="query-form">
              <div className="form-group">
                <label htmlFor="collection">Collection:</label>
                <select
                  id="collection"
                  value={selectedCollection}
                  onChange={(e) => setSelectedCollection(e.target.value)}
                  className="collection-select"
                  disabled={queryLoading}
                >
                  {collections.length > 0 ? (
                    collections.map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))
                  ) : (
                    <option value="knowledge_base">knowledge_base</option>
                  )}
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="question">Your Question:</label>
                <textarea
                  id="question"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="What would you like to know?"
                  className="question-input"
                  rows="4"
                  disabled={queryLoading}
                />
              </div>

              <div className="button-group">
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={queryLoading || !question.trim()}
                >
                  {queryLoading ? 'Thinking...' : 'Ask Question'}
                </button>
                <button
                  type="button"
                  onClick={handleClearQuery}
                  className="btn btn-secondary"
                  disabled={queryLoading}
                >
                  Clear
                </button>
              </div>
            </form>

            {/* Loading Indicator */}
            {queryLoading && (
              <div className="loading">
                <div className="spinner"></div>
                <p>Processing your question...</p>
              </div>
            )}

            {/* Error Message */}
            {queryError && (
              <div className="error-message">
                <strong>Error:</strong> {queryError}
              </div>
            )}

            {/* Answer Section */}
            {answer && (
              <div className="answer-section">
                <h2>Answer:</h2>
                <div className="answer-content">
                  {answer}
                </div>

                {/* Source Documents */}
                {sources && sources.length > 0 && (
                  <div className="sources-section">
                    <h3>Source Documents:</h3>
                    {sources.map((source, index) => (
                      <div key={index} className="source-card">
                        <div className="source-header">
                          <span className="source-number">Source {index + 1}</span>
                          {source.metadata && (
                            <span className="source-metadata">
                              {Object.entries(source.metadata).map(([key, value]) => (
                                <span key={key} className="metadata-item">
                                  {key}: {value}
                                </span>
                              ))}
                            </span>
                          )}
                        </div>
                        <div className="source-content">
                          {source.content}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Ingest Tab */}
        {activeTab === 'ingest' && (
          <div className="tab-content">
            <h2>Ingest Documents</h2>

            {/* Collection Selection with Create New Option */}
            <div className="form-group">
              <label htmlFor="ingest-collection">Target Collection:</label>
              <div className="collection-row">
                <select
                  id="ingest-collection"
                  value={ingestCollection}
                  onChange={(e) => setIngestCollection(e.target.value)}
                  className="collection-select"
                  disabled={ingestLoading}
                >
                  {collections.length > 0 ? (
                    collections.map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))
                  ) : (
                    <option value="knowledge_base">knowledge_base</option>
                  )}
                </select>
                <button
                  type="button"
                  onClick={() => setShowNewCollection(!showNewCollection)}
                  className="btn btn-secondary btn-small"
                >
                  + New Collection
                </button>
              </div>
            </div>

            {/* New Collection Form */}
            {showNewCollection && (
              <div className="new-collection-form">
                <input
                  type="text"
                  value={newCollectionName}
                  onChange={(e) => setNewCollectionName(e.target.value)}
                  placeholder="Enter collection name"
                  className="text-input"
                />
                <button
                  onClick={handleCreateCollection}
                  className="btn btn-primary btn-small"
                >
                  Create
                </button>
                <button
                  onClick={() => {
                    setShowNewCollection(false);
                    setNewCollectionName('');
                  }}
                  className="btn btn-secondary btn-small"
                >
                  Cancel
                </button>
              </div>
            )}

            {/* Ingest Mode Selection */}
            <div className="ingest-mode-selector">
              <button
                className={`mode-button ${ingestMode === 'text' ? 'active' : ''}`}
                onClick={() => setIngestMode('text')}
              >
                📝 Text Input
              </button>
              <button
                className={`mode-button ${ingestMode === 'file' ? 'active' : ''}`}
                onClick={() => setIngestMode('file')}
              >
                📁 File Upload
              </button>
            </div>

            {/* Text Ingest Form */}
            {ingestMode === 'text' && (
              <form onSubmit={handleIngestText} className="ingest-form">
                <div className="form-group">
                  <label htmlFor="ingest-text">Enter Text to Ingest:</label>
                  <textarea
                    id="ingest-text"
                    value={ingestText}
                    onChange={(e) => setIngestText(e.target.value)}
                    placeholder="Paste your text here... (will be automatically chunked and stored)"
                    className="text-area-large"
                    rows="10"
                    disabled={ingestLoading}
                  />
                  <small className="help-text">
                    Text will be split into chunks of ~1000 characters with 200 character overlap
                  </small>
                </div>

                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={ingestLoading || !ingestText.trim()}
                >
                  {ingestLoading ? 'Ingesting...' : 'Ingest Text'}
                </button>
              </form>
            )}

            {/* File Ingest Form */}
            {ingestMode === 'file' && (
              <form onSubmit={handleIngestFile} className="ingest-form">
                <div className="form-group">
                  <label htmlFor="file-input">Select File (.txt or .md):</label>
                  <input
                    id="file-input"
                    type="file"
                    accept=".txt,.md"
                    onChange={(e) => setIngestFile(e.target.files[0])}
                    className="file-input"
                    disabled={ingestLoading}
                  />
                  {ingestFile && (
                    <div className="file-info">
                      Selected: <strong>{ingestFile.name}</strong> ({(ingestFile.size / 1024).toFixed(2)} KB)
                    </div>
                  )}
                  <small className="help-text">
                    Supported formats: .txt, .md
                  </small>
                </div>

                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={ingestLoading || !ingestFile}
                >
                  {ingestLoading ? 'Uploading...' : 'Upload & Ingest'}
                </button>
              </form>
            )}

            {/* Success Message */}
            {ingestSuccess && (
              <div className="success-message">
                <strong>Success:</strong> {ingestSuccess}
              </div>
            )}

            {/* Error Message */}
            {ingestError && (
              <div className="error-message">
                <strong>Error:</strong> {ingestError}
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>Powered by Ollama + ChromaDB</p>
      </footer>
    </div>
  );
}

export default App;
