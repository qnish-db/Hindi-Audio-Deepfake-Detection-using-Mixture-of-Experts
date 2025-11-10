import React from 'react';

// Import clean, minimal components
import {
  IntegratedGradientsCard,
  SHAPCard,
  LRPCard,
  ImprovedWaveformCard,
  ImprovedTemporalCard
} from './XAI_Clean';

// Main XAI Visualizations Component with staggered delays
const XAIVisualizations = ({ xaiData, isFake, fileName }) => {
  if (!xaiData) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: '#888' }}>
        <p>No XAI data available</p>
      </div>
    );
  }

  const accentColor = isFake ? '#ef4444' : '#22c55e';
  const basicXAI = xaiData.basic_xai || {};
  const advancedXAI = xaiData.advanced_xai || {};

  // Check if we have any data to display
  const hasData = (basicXAI && Object.keys(basicXAI).length > 0) || 
                  (advancedXAI && Object.keys(advancedXAI).length > 0);
  
  if (!hasData) {
    return (
      <div style={{ 
        padding: '60px', 
        textAlign: 'center', 
        color: '#888',
        background: 'rgba(255, 255, 255, 0.02)',
        borderRadius: '24px',
        marginTop: '40px'
      }}>
        <p style={{ fontSize: '18px', marginBottom: '12px' }}>⚠️ XAI Analysis Incomplete</p>
        <p style={{ fontSize: '14px', color: '#666' }}>The explainability analysis did not return valid data. Please try again.</p>
      </div>
    );
  }

  // Staggered delays for smooth sequential appearance
  const baseDelay = 0;
  const delayIncrement = 1400; // 1.4 seconds between each graph

  return (
    <div style={{ marginTop: '80px' }}>
      {/* Section Header */}
      <div style={{
        textAlign: 'center',
        marginBottom: '80px',
        padding: '60px',
        background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%)',
        borderRadius: '32px',
        border: '2px solid rgba(255, 255, 255, 0.15)',
        position: 'relative',
        overflow: 'hidden',
        animation: 'fadeIn 1s ease-out forwards'
      }}>
        <div style={{
          position: 'absolute',
          top: '-50%',
          left: '-50%',
          width: '200%',
          height: '200%',
          background: `radial-gradient(circle, ${accentColor}10 0%, transparent 50%)`,
          animation: 'pulse 4s ease-in-out infinite',
          pointerEvents: 'none'
        }} />
        
        <div style={{ position: 'relative', zIndex: 1 }}>
          <h2 style={{
            color: '#fff',
            fontSize: '56px',
            fontWeight: '900',
            marginBottom: '20px',
            letterSpacing: '-2px',
            textShadow: '0 2px 40px rgba(255, 255, 255, 0.1)'
          }}>
            🔬 Explainable AI Analysis
          </h2>
          <p style={{ color: '#888', fontSize: '20px', maxWidth: '900px', margin: '0 auto', lineHeight: '1.8' }}>
            Deep dive into how the AI made its decision. Multiple analysis methods reveal
            different aspects of the model's reasoning process.
          </p>
          <div style={{
            marginTop: '28px',
            color: '#666',
            fontSize: '15px',
            fontFamily: 'monospace',
            fontWeight: '700'
          }}>
            Total Processing Time: {xaiData.processing_time_ms || 0}ms
          </div>
        </div>
      </div>

      {/* 1. INTEGRATED GRADIENTS - Temporal Attribution (Most Important) */}
      {advancedXAI.integrated_gradients && (
        <IntegratedGradientsCard
          data={advancedXAI.integrated_gradients}
          accentColor={accentColor}
          delay={baseDelay + delayIncrement * 0}
          audioDuration={basicXAI.temporal_heatmap?.timestamps?.[basicXAI.temporal_heatmap.timestamps.length - 1]}
          isFake={isFake}
        />
      )}

      {/* 2. SHAP - Frequency Band Analysis */}
      {advancedXAI.shap_approximation && (
        <SHAPCard
          data={advancedXAI.shap_approximation}
          accentColor={accentColor}
          delay={baseDelay + delayIncrement * 1}
          isFake={isFake}
        />
      )}

      {/* LRP REMOVED PER USER REQUEST */}

      {/* 3. WAVEFORM - Audio Visualization */}
      {basicXAI.temporal_heatmap && (
        <ImprovedWaveformCard
          data={basicXAI.temporal_heatmap}
          accentColor={accentColor}
          delay={baseDelay + delayIncrement * 2}
          isFake={isFake}
        />
      )}

      {/* 4. TEMPORAL CONSISTENCY - REPLACES EXPERT AGREEMENT */}
      {basicXAI.temporal_heatmap && (
        <ImprovedTemporalCard
          data={basicXAI.temporal_heatmap}
          accentColor={accentColor}
          delay={baseDelay + delayIncrement * 3}
          isFake={isFake}
        />
      )}

      {/* Processing Time Summary */}
      <div style={{
        background: 'rgba(255, 255, 255, 0.02)',
        border: '2px solid #2a2a2a',
        borderRadius: '24px',
        padding: '32px',
        textAlign: 'center',
        color: '#666',
        fontSize: '16px',
        fontWeight: '700',
        animation: 'fadeIn 1s ease-out forwards',
        animationDelay: `${baseDelay + delayIncrement * 4}ms`,
        opacity: 0
      }}>
        ✅ XAI Analysis Complete | Basic: {basicXAI.processing_time_ms || 0}ms | Advanced: {advancedXAI.processing_time_ms || 0}ms
      </div>

      {/* Global CSS Animations */}
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes pulse {
          0%, 100% { opacity: 0.5; transform: scale(1); }
          50% { opacity: 1; transform: scale(1.05); }
        }
      `}</style>
    </div>
  );
};

export default XAIVisualizations;
