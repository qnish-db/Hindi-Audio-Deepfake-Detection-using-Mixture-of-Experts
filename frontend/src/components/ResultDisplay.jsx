import { useState } from 'react';
import XAIVisualizations from './XAIComponents_Final';

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const ResultDisplay = ({ result, fileName, fileSize, onReset }) => {
  const [xaiData, setXaiData] = useState(null);
  const [xaiLoading, setXaiLoading] = useState(false);
  const [xaiError, setXaiError] = useState('');
  console.log('ResultDisplay received:', { result, fileName, fileSize });
  
  // Early validation
  if (!result || typeof result !== 'object') {
    console.error('Invalid result:', result);
    return (
      <div style={{ 
        textAlign: 'center', 
        padding: '80px 40px',
        color: '#ef4444',
        fontSize: '18px',
        fontFamily: "'Montserrat', sans-serif"
      }}>
        <p>Error: No data received from server</p>
        <button
          onClick={onReset}
          style={{
            marginTop: '24px',
            padding: '12px 24px',
            background: '#1a1a1a',
            border: '1px solid #333',
            borderRadius: '8px',
            color: '#fff',
            cursor: 'pointer',
            fontFamily: "'Montserrat', sans-serif"
          }}
        >
          Try Again
        </button>
      </div>
    );
  }

  const formatSize = (bytes) => {
    if (!bytes) return '—';
    const mb = bytes / (1024 * 1024);
    return mb < 1 ? Math.round(bytes / 1024) + "KB" : mb.toFixed(1) + "MB";
  };
  
  const truncateFilename = (name, maxLength = 30) => {
    if (!name || name.length <= maxLength) return name || 'audio.wav';
    const ext = name.split('.').pop();
    const nameWithoutExt = name.substring(0, name.lastIndexOf('.'));
    const truncated = nameWithoutExt.substring(0, maxLength - ext.length - 3);
    return `${truncated}...${ext}`;
  };

  const labelText = (lbl) => (lbl === 1 ? "FAKE" : "REAL");

  const handleGenerateXAI = async () => {
    if (!fileName) return;
    setXaiLoading(true);
    setXaiError('');
    try {
      const file = new File([new Blob()], fileName, { type: 'audio/wav' });
      const fd = new FormData();
      fd.append('file', file);
      const r = await fetch(`${API}/xai`, { method: 'POST', body: fd });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const json = await r.json();
      if (json.error) {
        setXaiError(json.error);
      } else {
        setXaiData(json);
      }
    } catch (e) {
      setXaiError(e.message);
    } finally {
      setXaiLoading(false);
    }
  };

  // Check if this is a language gate rejection
  if (result.error_code === "not_hindi") {
    return (
      <div style={{ 
        width: '100%', 
        minHeight: '100vh',
        background: 'transparent',
        padding: '60px 40px',
        fontFamily: "'Montserrat', sans-serif"
      }}>
        <div style={{ maxWidth: '900px', margin: '0 auto' }}>
          
          {/* Rejection Card */}
          <div style={{
            background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.05) 100%)',
            border: '2px solid rgba(239, 68, 68, 0.3)',
            borderRadius: '24px',
            padding: '60px',
            marginBottom: '40px',
            textAlign: 'center'
          }}>
            {/* Error Icon */}
            <div style={{
              width: '80px',
              height: '80px',
              background: 'rgba(239, 68, 68, 0.2)',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 24px'
            }}>
              <svg width="40" height="40" fill="#ef4444" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>

            <h2 style={{ 
              color: '#ef4444', 
              fontSize: '32px', 
              fontWeight: '700', 
              marginBottom: '16px'
            }}>
              Language Check Failed
            </h2>
            <p style={{ 
              color: '#bbb', 
              fontSize: '16px', 
              marginBottom: '32px', 
              lineHeight: '1.6'
            }}>
              {result.message || 'Audio is not in Hindi. Please upload a Hindi audio file.'}
            </p>

            {/* Language Details */}
            <div style={{
              background: 'rgba(0, 0, 0, 0.3)',
              border: '1px solid #2a2a2a',
              borderRadius: '16px',
              padding: '24px',
              marginBottom: '32px'
            }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px', textAlign: 'center' }}>
                <div>
                  <div style={{ color: '#666', fontSize: '12px', marginBottom: '8px' }}>Detected Language</div>
                  <div style={{ color: '#fff', fontSize: '24px', fontWeight: '700' }}>
                    {result.lid_debug?.detected_lang?.toUpperCase() || '—'}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#666', fontSize: '12px', marginBottom: '8px' }}>Hindi Probability</div>
                  <div style={{ color: '#ef4444', fontSize: '24px', fontWeight: '700' }}>
                    {result.lid_debug?.p_hi ? (result.lid_debug.p_hi * 100).toFixed(1) + '%' : '0%'}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#666', fontSize: '12px', marginBottom: '8px' }}>Speech Content</div>
                  <div style={{ color: '#fff', fontSize: '24px', fontWeight: '700' }}>
                    {result.lid_debug?.speech_fraction ? (result.lid_debug.speech_fraction * 100).toFixed(0) + '%' : '—'}
                  </div>
                </div>
              </div>
            </div>

            {/* File Info */}
            <div style={{ color: '#666', fontSize: '14px', marginBottom: '32px' }}>
              File: {truncateFilename(fileName) || 'audio.wav'} • {fileSize ? formatSize(fileSize) : '—'}
            </div>

            {/* Action Button */}
            <button
              onClick={onReset}
              className="shimmer-btn"
              style={{
                background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
                border: '1px solid #333',
                borderRadius: '16px',
                padding: '16px 40px',
                color: 'white',
                fontSize: '16px',
                fontWeight: '700',
                cursor: 'pointer',
                position: 'relative',
                overflow: 'hidden'
              }}
            >
              <div className="shimmer" />
              <span style={{ position: 'relative', zIndex: 1 }}>Try Another File</span>
            </button>
          </div>

          {/* Technical Details (Collapsible) */}
          {result.lid_debug && (
            <details style={{ marginTop: '20px' }}>
              <summary style={{
                color: '#888',
                fontSize: '16px',
                fontWeight: '600',
                cursor: 'pointer',
                padding: '16px',
                background: 'rgba(255, 255, 255, 0.02)',
                borderRadius: '12px',
                border: '1px solid #2a2a2a'
              }}>
                Technical Details
              </summary>
              <div style={{
                marginTop: '16px',
                background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
                border: '1px solid #2a2a2a',
                borderRadius: '16px',
                padding: '24px',
                fontSize: '14px',
                fontFamily: 'monospace',
                color: '#888'
              }}>
                <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                  {JSON.stringify(result.lid_debug, null, 2)}
                </pre>
              </div>
            </details>
          )}
        </div>
      </div>
    );
  }

  // Normal inference result (Hindi audio passed)
  // Safety check: ensure we have the required fields
  if (result.prob_fake === undefined || result.label === undefined) {
    console.error('Invalid result format:', result);
    return (
      <div style={{ 
        textAlign: 'center', 
        padding: '80px 40px',
        color: '#ef4444',
        fontSize: '18px',
        fontFamily: "'Montserrat', sans-serif"
      }}>
        <p>Error: Invalid response from server</p>
        <button
          onClick={onReset}
          style={{
            marginTop: '24px',
            padding: '12px 24px',
            background: '#1a1a1a',
            border: '1px solid #333',
            borderRadius: '8px',
            color: '#fff',
            cursor: 'pointer',
            fontFamily: "'Montserrat', sans-serif"
          }}
        >
          Try Again
        </button>
      </div>
    );
  }

  const isFake = result.label === 1;
  const confidence = result.prob_fake !== undefined ? (Number(result.prob_fake) * 100).toFixed(2) : '0.00';

  return (
    <div style={{ 
      width: '100%', 
      minHeight: '100vh',
      background: 'transparent',
      padding: '60px 40px',
      fontFamily: "'Montserrat', sans-serif"
    }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        
        {/* Hero Result Card */}
        <div style={{
          background: isFake 
            ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.05) 100%)'
            : 'linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(22, 163, 74, 0.05) 100%)',
          border: `2px solid ${isFake ? 'rgba(239, 68, 68, 0.3)' : 'rgba(34, 197, 94, 0.3)'}`,
          borderRadius: '32px',
          padding: '80px 60px',
          marginBottom: '40px',
          position: 'relative',
          overflow: 'hidden'
        }}>
          {/* Background Pattern */}
          <div style={{
            position: 'absolute',
            top: 0,
            right: 0,
            width: '400px',
            height: '400px',
            background: isFake 
              ? 'radial-gradient(circle, rgba(239, 68, 68, 0.15) 0%, transparent 70%)'
              : 'radial-gradient(circle, rgba(34, 197, 94, 0.15) 0%, transparent 70%)',
            pointerEvents: 'none'
          }} />

          <div style={{ position: 'relative', zIndex: 1 }}>
            {/* Status Badge */}
            <div style={{ 
              display: 'inline-flex',
              alignItems: 'center',
              gap: '12px',
              background: isFake ? 'rgba(239, 68, 68, 0.2)' : 'rgba(34, 197, 94, 0.2)',
              border: `1px solid ${isFake ? 'rgba(239, 68, 68, 0.4)' : 'rgba(34, 197, 94, 0.4)'}`,
              borderRadius: '50px',
              padding: '12px 28px',
              marginBottom: '40px'
            }}>
              <div style={{
                width: '12px',
                height: '12px',
                borderRadius: '50%',
                background: isFake ? '#ef4444' : '#22c55e',
                boxShadow: `0 0 20px ${isFake ? '#ef4444' : '#22c55e'}`
              }} />
              <span style={{ 
                color: isFake ? '#ef4444' : '#22c55e',
                fontSize: '16px',
                fontWeight: '700',
                letterSpacing: '1px'
              }}>
                {isFake ? 'DEEPFAKE DETECTED' : 'AUTHENTIC AUDIO'}
              </span>
            </div>

            {/* Main Result */}
            <div style={{ 
              fontSize: '120px', 
              fontWeight: '900',
              color: isFake ? '#ef4444' : '#22c55e',
              lineHeight: '1',
              marginBottom: '30px',
              letterSpacing: '-2px'
            }}>
              {labelText(result.label)}
            </div>

            {/* Confidence Bar */}
            <div style={{ marginTop: '50px' }}>
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '16px'
              }}>
                <span style={{ color: '#888', fontSize: '18px', fontWeight: '600' }}>Confidence Score</span>
                <span style={{ 
                  color: '#fff',
                  fontSize: '36px',
                  fontWeight: '900'
                }}>{confidence}%</span>
              </div>
              <div style={{
                width: '100%',
                height: '16px',
                background: 'rgba(255, 255, 255, 0.1)',
                borderRadius: '50px',
                overflow: 'hidden',
                position: 'relative'
              }}>
                <div style={{
                  width: `${confidence}%`,
                  height: '100%',
                  background: isFake 
                    ? 'linear-gradient(90deg, #ef4444 0%, #dc2626 100%)'
                    : 'linear-gradient(90deg, #22c55e 0%, #16a34a 100%)',
                  borderRadius: '50px',
                  transition: 'width 1s ease-out',
                  boxShadow: `0 0 20px ${isFake ? '#ef4444' : '#22c55e'}`
                }} />
              </div>
            </div>
          </div>
        </div>

        {/* Info Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px', marginBottom: '40px' }}>
          {/* File Info Card */}
          <div style={{
            background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
            border: '1px solid #2a2a2a',
            borderRadius: '24px',
            padding: '32px'
          }}>
            <div style={{ 
              width: '48px',
              height: '48px',
              background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
              border: '1px solid #333',
              borderRadius: '12px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '20px'
            }}>
              <svg width="24" height="24" fill="white" viewBox="0 0 20 20">
                <path d="M18 3a1 1 0 00-1.196-.98l-10 2A1 1 0 006 5v9.114A4.369 4.369 0 005 14c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2V7.82l8-1.6v5.894A4.37 4.37 0 0015 12c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2V3z" />
              </svg>
            </div>
            <h3 style={{ color: '#fff', fontSize: '20px', fontWeight: '700', marginBottom: '16px' }}>Audio File</h3>
            <p style={{ color: '#666', fontSize: '14px', marginBottom: '8px' }}>{truncateFilename(fileName)}</p>
            <div style={{ display: 'flex', gap: '16px', marginTop: '16px' }}>
              <div>
                <div style={{ color: '#666', fontSize: '12px', marginBottom: '4px' }}>Size</div>
                <div style={{ color: '#fff', fontSize: '16px', fontWeight: '600' }}>{formatSize(fileSize)}</div>
              </div>
              <div>
                <div style={{ color: '#666', fontSize: '12px', marginBottom: '4px' }}>Duration</div>
                <div style={{ color: '#fff', fontSize: '16px', fontWeight: '600' }}>{result.debug?.audio_sec}s</div>
              </div>
            </div>
          </div>

          {/* Threshold Card */}
          <div style={{
            background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
            border: '1px solid #2a2a2a',
            borderRadius: '24px',
            padding: '32px'
          }}>
            <div style={{ 
              width: '48px',
              height: '48px',
              background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
              border: '1px solid #333',
              borderRadius: '12px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '20px'
            }}>
              <svg width="24" height="24" fill="white" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M6 2a1 1 0 00-1 1v1H4a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-1V3a1 1 0 10-2 0v1H7V3a1 1 0 00-1-1zm0 5a1 1 0 000 2h8a1 1 0 100-2H6z" clipRule="evenodd" />
              </svg>
            </div>
            <h3 style={{ color: '#fff', fontSize: '20px', fontWeight: '700', marginBottom: '16px' }}>Threshold</h3>
            <p style={{ color: '#fff', fontSize: '24px', fontWeight: '700', fontFamily: 'monospace' }}>
              {typeof result.threshold_used === 'number' ? result.threshold_used.toFixed(6) : '—'}
            </p>
            <p style={{ color: '#666', fontSize: '14px', marginTop: '8px' }}>{result.threshold_source || 'Default'}</p>
          </div>

          {/* Device Card */}
          <div style={{
            background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
            border: '1px solid #2a2a2a',
            borderRadius: '24px',
            padding: '32px'
          }}>
            <div style={{ 
              width: '48px',
              height: '48px',
              background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
              border: '1px solid #333',
              borderRadius: '12px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '20px'
            }}>
              <svg width="24" height="24" fill="white" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M3 5a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2h-2.22l.123.489.804.804A1 1 0 0113 18H7a1 1 0 01-.707-1.707l.804-.804L7.22 15H5a2 2 0 01-2-2V5zm5.771 7H5V5h10v7H8.771z" clipRule="evenodd" />
              </svg>
            </div>
            <h3 style={{ color: '#fff', fontSize: '20px', fontWeight: '700', marginBottom: '16px' }}>Processing</h3>
            <p style={{ color: '#fff', fontSize: '18px', fontWeight: '600', marginBottom: '8px' }}>{result.meta?.device || 'CPU'}</p>
            <p style={{ color: '#666', fontSize: '14px' }}>Total: {result.debug?.t_overall_ms}ms</p>
          </div>
        </div>

        {/* Action Button */}
        <button
          onClick={onReset}
          className="shimmer-btn"
          style={{
            width: '100%',
            background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
            border: '1px solid #333',
            borderRadius: '16px',
            padding: '20px',
            color: 'white',
            fontSize: '18px',
            fontWeight: '700',
            cursor: 'pointer',
            position: 'relative',
            overflow: 'hidden',
            boxShadow: '0 10px 40px rgba(0, 0, 0, 0.5)'
          }}
        >
          <div className="shimmer" />
          <span style={{ position: 'relative', zIndex: 1 }}>Analyze New File</span>
        </button>

        {/* Language Check Info */}
        {result.language_check && (
          <div style={{
            background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(22, 163, 74, 0.05) 100%)',
            border: '1px solid rgba(34, 197, 94, 0.3)',
            borderRadius: '24px',
            padding: '32px',
            marginTop: '40px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
              <svg width="24" height="24" fill="#22c55e" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              <h3 style={{ color: '#22c55e', fontSize: '20px', fontWeight: '700', margin: 0 }}>Language Verified: Hindi</h3>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
              <div>
                <div style={{ color: '#666', fontSize: '12px', marginBottom: '6px' }}>Detected Language</div>
                <div style={{ color: '#fff', fontSize: '18px', fontWeight: '600' }}>
                  {result.language_check.detected_lang?.toUpperCase() || '—'}
                </div>
              </div>
              <div>
                <div style={{ color: '#666', fontSize: '12px', marginBottom: '6px' }}>Hindi Confidence</div>
                <div style={{ color: '#22c55e', fontSize: '18px', fontWeight: '600' }}>
                  {result.language_check.p_hi ? (result.language_check.p_hi * 100).toFixed(1) + '%' : '—'}
                </div>
              </div>
              <div>
                <div style={{ color: '#666', fontSize: '12px', marginBottom: '6px' }}>Speech Content</div>
                <div style={{ color: '#fff', fontSize: '18px', fontWeight: '600' }}>
                  {result.language_check.speech_fraction ? (result.language_check.speech_fraction * 100).toFixed(0) + '%' : '—'}
                </div>
              </div>
              <div>
                <div style={{ color: '#666', fontSize: '12px', marginBottom: '6px' }}>LID Time</div>
                <div style={{ color: '#fff', fontSize: '18px', fontWeight: '600' }}>
                  {result.language_check.t_lid_ms || result.debug?.t_lid_ms || '—'} <span style={{ fontSize: '12px', color: '#666' }}>ms</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Performance Metrics */}
        <div style={{
          background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
          border: '1px solid #2a2a2a',
          borderRadius: '24px',
          padding: '40px',
          marginTop: '40px'
        }}>
          <h3 style={{ color: '#fff', fontSize: '24px', fontWeight: '700', marginBottom: '32px' }}>Performance Metrics</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '24px', marginBottom: '32px' }}>
            <div style={{ padding: '20px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '12px', border: '1px solid #2a2a2a' }}>
              <div style={{ color: '#666', fontSize: '12px', marginBottom: '8px' }}>Language Check</div>
              <div style={{ color: '#fff', fontSize: '24px', fontWeight: '700' }}>{result.debug?.t_lid_ms || '—'} <span style={{ fontSize: '14px', color: '#666' }}>ms</span></div>
            </div>
            <div style={{ padding: '20px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '12px', border: '1px solid #2a2a2a' }}>
              <div style={{ color: '#666', fontSize: '12px', marginBottom: '8px' }}>HuBERT Time</div>
              <div style={{ color: '#fff', fontSize: '24px', fontWeight: '700' }}>{result.debug?.t_hubert_ms || '—'} <span style={{ fontSize: '14px', color: '#666' }}>ms</span></div>
            </div>
            <div style={{ padding: '20px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '12px', border: '1px solid #2a2a2a' }}>
              <div style={{ color: '#666', fontSize: '12px', marginBottom: '8px' }}>Wav2Vec2 Time</div>
              <div style={{ color: '#fff', fontSize: '24px', fontWeight: '700' }}>{result.debug?.t_w2v_ms || '—'} <span style={{ fontSize: '14px', color: '#666' }}>ms</span></div>
            </div>
            <div style={{ padding: '20px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '12px', border: '1px solid #2a2a2a' }}>
              <div style={{ color: '#666', fontSize: '12px', marginBottom: '8px' }}>MoE Time</div>
              <div style={{ color: '#fff', fontSize: '24px', fontWeight: '700' }}>{result.debug?.t_moe_ms || '—'} <span style={{ fontSize: '14px', color: '#666' }}>ms</span></div>
            </div>
          </div>

          {/* Gate Weights */}
          {result.gate && Object.keys(result.gate).length > 0 && (
            <div style={{ marginTop: '24px' }}>
              <h4 style={{ color: '#888', fontSize: '16px', fontWeight: '600', marginBottom: '16px' }}>Gate Weights</h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>
                {Object.entries(result.gate).map(([k,v]) => (
                  <div key={k} style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between',
                    padding: '12px 16px',
                    background: 'rgba(255, 255, 255, 0.02)',
                    borderRadius: '8px'
                  }}>
                    <span style={{ color: '#888', fontSize: '14px' }}>{k}</span>
                    <span style={{ color: '#fff', fontSize: '14px', fontWeight: '600', fontFamily: 'monospace' }}>{Number(v).toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Model Info */}
          <div style={{ marginTop: '32px', paddingTop: '24px', borderTop: '1px solid #2a2a2a' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px', fontSize: '14px' }}>
              <div>
                <span style={{ color: '#666' }}>Checkpoint:</span>
                <span style={{ color: '#fff', marginLeft: '8px', fontFamily: 'monospace' }}>{result.meta?.checkpoint || '—'}</span>
              </div>
              <div>
                <span style={{ color: '#666' }}>Run Name:</span>
                <span style={{ color: '#fff', marginLeft: '8px', fontFamily: 'monospace' }}>{result.meta?.run_name || '—'}</span>
              </div>
              <div>
                <span style={{ color: '#666' }}>Best Val EER:</span>
                <span style={{ color: '#fff', marginLeft: '8px', fontFamily: 'monospace' }}>{result.meta?.best_val_eer ? Number(result.meta.best_val_eer).toFixed(4) : '—'}</span>
              </div>
              {result.truth_label_str && (
                <div>
                  <span style={{ color: '#666' }}>Ground Truth:</span>
                  <span style={{ color: '#fff', marginLeft: '8px', fontFamily: 'monospace' }}>{result.truth_label_str}</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* XAI BUTTON */}
        {!xaiData && !xaiLoading && (
          <div style={{
            textAlign: 'center',
            marginTop: '60px'
          }}>
            <button
              onClick={handleGenerateXAI}
              style={{
                padding: '24px 48px',
                background: isFake 
                  ? 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)'
                  : 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)',
                border: 'none',
                borderRadius: '16px',
                color: 'white',
                fontSize: '20px',
                fontWeight: '700',
                cursor: 'pointer',
                boxShadow: `0 10px 40px ${isFake ? 'rgba(239, 68, 68, 0.4)' : 'rgba(34, 197, 94, 0.4)'}`,
                transition: 'all 0.3s ease',
                letterSpacing: '0.5px'
              }}
              onMouseOver={(e) => e.target.style.transform = 'translateY(-2px)'}
              onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}
            >
              🔍 Generate Explainability Analysis
            </button>
            <p style={{ color: '#666', fontSize: '14px', marginTop: '16px' }}>
              Detailed AI decision explanations, temporal analysis, and visual breakdowns
            </p>
          </div>
        )}

        {/* XAI LOADING */}
        {xaiLoading && (
          <div style={{
            textAlign: 'center',
            padding: '80px 40px',
            marginTop: '40px'
          }}>
            <div style={{
              width: '80px',
              height: '80px',
              border: '6px solid #333',
              borderTop: `6px solid ${isFake ? '#ef4444' : '#22c55e'}`,
              borderRadius: '50%',
              margin: '0 auto 24px',
              animation: 'spin 1s linear infinite'
            }} />
            <p style={{ color: '#888', fontSize: '18px', fontWeight: '600' }}>Generating XAI Analysis...</p>
            <p style={{ color: '#666', fontSize: '14px', marginTop: '8px' }}>This may take 10-30 seconds</p>
          </div>
        )}

        {/* XAI ERROR */}
        {xaiError && (
          <div style={{
            marginTop: '40px',
            padding: '24px',
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: '16px',
            color: '#ef4444',
            textAlign: 'center'
          }}>
            <p style={{ fontSize: '16px', fontWeight: '600' }}>XAI Error: {xaiError}</p>
            <button
              onClick={handleGenerateXAI}
              style={{
                marginTop: '16px',
                padding: '12px 24px',
                background: '#ef4444',
                border: 'none',
                borderRadius: '8px',
                color: 'white',
                fontSize: '14px',
                fontWeight: '600',
                cursor: 'pointer'
              }}
            >
              Retry
            </button>
          </div>
        )}

        {/* XAI VISUALIZATIONS */}
        {xaiData && (
          <XAIVisualizations 
            xaiData={xaiData} 
            isFake={isFake}
            fileName={fileName}
          />
        )}

      </div>
    </div>
  );
};

export default ResultDisplay;
