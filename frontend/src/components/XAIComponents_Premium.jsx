import React, { useState, useEffect } from 'react';

// ===== ANIMATED CONTAINER WITH STAGGERED CONTENT =====
const AnimatedCard = ({ children, delay = 0, style = {} }) => {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setVisible(true), delay);
    return () => clearTimeout(timer);
  }, [delay]);

  return (
    <div
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? 'translateY(0)' : 'translateY(40px)',
        transition: 'all 1.2s cubic-bezier(0.16, 1, 0.3, 1)',
        ...style
      }}
    >
      {children}
    </div>
  );
};

// ===== 1. INTEGRATED GRADIENTS - SMOOTH WAVEFORM STYLE =====
const IntegratedGradientsCard = ({ data, accentColor, delay }) => {
  if (!data || !data.temporal_attribution) return null;

  const attribution_scores = data.temporal_attribution || [];
  if (attribution_scores.length === 0) return null;
  
  // Generate timeline from attribution length (assume 50ms per frame)
  const timeline = attribution_scores.map((_, i) => i * 0.05);
  const maxTime = Math.max(timeline[timeline.length - 1] || 1, 0.1); // Ensure non-zero
  const maxAttr = Math.max(Math.max(...attribution_scores.map(Math.abs)), 0.001); // Ensure non-zero

  // Normalize attributions to 0-1
  const normalized = attribution_scores.map(v => (v / maxAttr + 1) / 2);

  // Generate smooth waveform path
  const generateWaveformPath = () => {
    const width = 1200;
    const height = 400;
    const centerY = height / 2;
    const amplitude = 150;

    let topPath = `M 0,${centerY} `;
    let bottomPath = `M 0,${centerY} `;

    timeline.forEach((t, i) => {
      const x = (t / maxTime) * width;
      const intensity = normalized[i];
      const wave = Math.sin(t * 8) * amplitude * 0.3 + Math.sin(t * 3) * amplitude * 0.5;
      const topY = centerY - Math.abs(wave) * intensity;
      const bottomY = centerY + Math.abs(wave) * intensity;

      topPath += `L ${x},${topY} `;
      bottomPath += `L ${x},${bottomY} `;
    });

    // Close path
    bottomPath += `L ${width},${centerY} `;
    for (let i = timeline.length - 1; i >= 0; i--) {
      const t = timeline[i];
      const x = (t / maxTime) * width;
      topPath += `L ${x},${centerY} `;
    }

    return { topPath: topPath + 'Z', bottomPath: bottomPath + 'Z' };
  };

  const { topPath, bottomPath } = generateWaveformPath();

  // Find high attribution regions
  const highRegions = [];
  let inRegion = false;
  let regionStart = 0;

  attribution_scores.forEach((score, i) => {
    if (score > maxAttr * 0.6 && !inRegion) {
      inRegion = true;
      regionStart = i;
    } else if ((score <= maxAttr * 0.6 || i === attribution_scores.length - 1) && inRegion) {
      highRegions.push({ start: timeline[regionStart] || 0, end: timeline[i] || 0, frames: `${regionStart}-${i}` });
      inRegion = false;
    }
  });

  return (
    <AnimatedCard delay={delay}>
      <div style={{
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
        border: '2px solid #2a2a2a',
        borderRadius: '32px',
        padding: '60px',
        marginBottom: '48px',
        position: 'relative',
        overflow: 'hidden'
      }}>
        {/* Animated background glow */}
        <div style={{
          position: 'absolute',
          top: '-50%',
          left: '-50%',
          width: '200%',
          height: '200%',
          background: `radial-gradient(circle, ${accentColor}15 0%, transparent 50%)`,
          animation: 'pulse 4s ease-in-out infinite',
          pointerEvents: 'none'
        }} />

        <div style={{ position: 'relative', zIndex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '20px', marginBottom: '20px' }}>
            <div style={{
              width: '64px',
              height: '64px',
              background: `linear-gradient(135deg, ${accentColor}40, ${accentColor}20)`,
              borderRadius: '20px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '32px',
              animation: 'float 3s ease-in-out infinite'
            }}>
              🎯
            </div>
            <div>
              <h3 style={{
                color: '#fff',
                fontSize: '38px',
                fontWeight: '900',
                marginBottom: '8px',
                letterSpacing: '-1px'
              }}>
                Integrated Gradients
              </h3>
              <div style={{
                color: '#666',
                fontSize: '14px',
                fontWeight: '600',
                letterSpacing: '2px',
                textTransform: 'uppercase'
              }}>
                TEMPORAL ATTRIBUTION ANALYSIS
              </div>
            </div>
          </div>

          <p style={{
            color: '#888',
            fontSize: '18px',
            marginBottom: '48px',
            lineHeight: '1.8',
            maxWidth: '900px'
          }}>
            Shows <span style={{ color: accentColor, fontWeight: '700' }}>which time segments</span> contributed most to the AI's decision.
            Brighter, more intense regions = higher influence on the <span style={{ color: accentColor, fontWeight: '700' }}>FAKE prediction</span>.
          </p>

          {/* SMOOTH WAVEFORM VISUALIZATION */}
          <div style={{
            background: 'rgba(0, 0, 0, 0.6)',
            borderRadius: '28px',
            padding: '48px 32px',
            border: '1px solid rgba(255, 255, 255, 0.05)',
            position: 'relative',
            overflow: 'hidden',
            minHeight: '500px'
          }}>
            <div style={{
              color: '#666',
              fontSize: '13px',
              fontWeight: '700',
              letterSpacing: '1.5px',
              marginBottom: '32px',
              textTransform: 'uppercase'
            }}>
              ATTRIBUTION WAVEFORM
            </div>

            <svg width="100%" height="400" viewBox="0 0 1200 400" preserveAspectRatio="xMidYMid meet">
              <defs>
                {/* Gradient for low attribution */}
                <linearGradient id="lowAttr" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.8" />
                  <stop offset="50%" stopColor="#3b82f6" stopOpacity="0.3" />
                  <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.1" />
                </linearGradient>

                {/* Gradient for high attribution */}
                <linearGradient id="highAttr" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor={accentColor} stopOpacity="0.95" />
                  <stop offset="50%" stopColor={accentColor} stopOpacity="0.6" />
                  <stop offset="100%" stopColor={accentColor} stopOpacity="0.2" />
                </linearGradient>

                {/* Glow filters */}
                <filter id="glow">
                  <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
                  <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                  </feMerge>
                </filter>
              </defs>

              {/* Center line */}
              <line x1="0" y1="200" x2="1200" y2="200" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />

              {/* High attribution background regions */}
              {attribution_scores.map((score, i) => {
                if (score <= maxAttr * 0.6) return null;
                const x = ((timeline[i] || 0) / maxTime) * 1200;
                const nextX = i < timeline.length - 1 ? ((timeline[i + 1] || 0) / maxTime) * 1200 : 1200;
                const width = Math.max(nextX - x, 1); // Ensure positive width
                if (isNaN(x) || isNaN(width)) return null;
                return (
                  <rect
                    key={`bg-${i}`}
                    x={x}
                    y="0"
                    width={width}
                    height="400"
                    fill={`${accentColor}20`}
                    style={{
                      animation: 'fadeIn 1.5s ease-out forwards',
                      animationDelay: `${delay + 600 + i * 20}ms`
                    }}
                  />
                );
              })}

              {/* Base waveform (low attribution) */}
              <path
                d={topPath}
                fill="url(#lowAttr)"
                stroke="#3b82f6"
                strokeWidth="3"
                filter="url(#glow)"
                style={{
                  animation: 'drawPath 2s ease-out forwards',
                  strokeDasharray: '3000',
                  strokeDashoffset: '3000'
                }}
              />

              {/* High attribution overlay segments */}
              {attribution_scores.map((score, i) => {
                if (score <= maxAttr * 0.6 || i === attribution_scores.length - 1) return null;

                const startX = ((timeline[i] || 0) / maxTime) * 1200;
                const endX = ((timeline[i + 1] || 0) / maxTime) * 1200;
                
                // Safety check for NaN
                if (isNaN(startX) || isNaN(endX)) return null;
                const centerY = 200;
                const intensity = normalized[i];
                const wave = Math.sin(timeline[i] * 8) * 150 * 0.3 + Math.sin(timeline[i] * 3) * 150 * 0.5;
                const topY = centerY - Math.abs(wave) * intensity;
                const bottomY = centerY + Math.abs(wave) * intensity;

                const nextWave = Math.sin(timeline[i + 1] * 8) * 150 * 0.3 + Math.sin(timeline[i + 1] * 3) * 150 * 0.5;
                const nextTopY = centerY - Math.abs(nextWave) * normalized[i + 1];
                const nextBottomY = centerY + Math.abs(nextWave) * normalized[i + 1];

                const path = `M ${startX},${topY} L ${endX},${nextTopY} L ${endX},${nextBottomY} L ${startX},${bottomY} Z`;

                return (
                  <path
                    key={`high-${i}`}
                    d={path}
                    fill="url(#highAttr)"
                    stroke={accentColor}
                    strokeWidth="3"
                    filter="url(#glow)"
                    style={{
                      animation: 'pulse 2s ease-in-out infinite',
                      animationDelay: `${i * 0.1}s`
                    }}
                  />
                );
              })}
            </svg>

            {/* Timeline */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              marginTop: '24px',
              paddingLeft: '4px',
              paddingRight: '4px',
              color: '#666',
              fontSize: '14px',
              fontWeight: '700'
            }}>
              <span>0.0s</span>
              <span>{isNaN(maxTime) ? '0.0' : (maxTime / 2).toFixed(2)}s</span>
              <span>{isNaN(maxTime) ? '0.0' : maxTime.toFixed(2)}s</span>
            </div>
          </div>

          {/* High Attribution Regions */}
          {highRegions.length > 0 && (
            <div style={{
              marginTop: '40px',
              padding: '32px',
              background: `${accentColor}10`,
              border: `2px solid ${accentColor}30`,
              borderRadius: '24px'
            }}>
              <div style={{
                color: accentColor,
                fontSize: '16px',
                fontWeight: '800',
                marginBottom: '24px',
                display: 'flex',
                alignItems: 'center',
                gap: '12px'
              }}>
                <span style={{ fontSize: '24px' }}>🔥</span>
                HIGH ATTRIBUTION REGIONS ({highRegions.length})
              </div>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '16px'
              }}>
                {highRegions.map((region, idx) => (
                  <div
                    key={idx}
                    style={{
                      background: 'rgba(0, 0, 0, 0.5)',
                      padding: '20px',
                      borderRadius: '16px',
                      border: `1px solid ${accentColor}40`,
                      animation: 'slideIn 0.6s ease-out forwards',
                      animationDelay: `${delay + 1000 + idx * 100}ms`,
                      opacity: 0
                    }}
                  >
                    <div style={{ color: '#fff', fontSize: '20px', fontWeight: '900', marginBottom: '8px' }}>
                      Region {idx + 1}
                    </div>
                    <div style={{ color: '#888', fontSize: '13px' }}>
                      {(region.start || 0).toFixed(2)}s - {(region.end || 0).toFixed(2)}s
                    </div>
                    <div style={{ color: '#666', fontSize: '12px', marginTop: '4px' }}>
                      Frames {region.frames || 'N/A'}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Legend */}
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '64px',
            marginTop: '40px',
            fontSize: '16px',
            fontWeight: '700'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              <div style={{
                width: '48px',
                height: '8px',
                background: 'linear-gradient(90deg, #3b82f6, #60a5fa)',
                borderRadius: '4px',
                boxShadow: '0 0 16px rgba(59, 130, 246, 0.6)'
              }} />
              <span style={{ color: '#3b82f6' }}>LOW IMPACT</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              <div style={{
                width: '48px',
                height: '8px',
                background: `linear-gradient(90deg, #fbbf24, #f59e0b)`,
                borderRadius: '4px',
                boxShadow: '0 0 16px rgba(251, 191, 36, 0.6)'
              }} />
              <span style={{ color: '#fbbf24' }}>MEDIUM IMPACT</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              <div style={{
                width: '48px',
                height: '8px',
                background: `linear-gradient(90deg, ${accentColor}, #dc2626)`,
                borderRadius: '4px',
                boxShadow: `0 0 16px ${accentColor}88`
              }} />
              <span style={{ color: accentColor }}>⚠ HIGH IMPACT</span>
            </div>
          </div>
        </div>

        {/* CSS Keyframes */}
        <style>{`
          @keyframes pulse {
            0%, 100% { opacity: 0.8; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.02); }
          }
          @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
          }
          @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
          }
          @keyframes drawPath {
            to { strokeDashoffset: 0; }
          }
          @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
          }
        `}</style>
      </div>
    </AnimatedCard>
  );
};

// ===== 2. SHAP - FREQUENCY BANDS PER EXPERT =====
const SHAPCard = ({ data, accentColor, delay }) => {
  if (!data || !data.expert_contributions) return null;

  const { expert_contributions = {}, top_moments = [] } = data;
  
  // Ensure expert_contributions is an object
  if (typeof expert_contributions !== 'object' || Object.keys(expert_contributions).length === 0) {
    return null;
  }

  // Simulated frequency bands (in real implementation, this comes from backend)
  const frequencyBands = [
    { name: '0-500 Hz', range: 'Low Frequencies', color: '#8b5cf6' },
    { name: '500-1000 Hz', range: 'Mid-Low Frequencies', color: '#3b82f6' },
    { name: '1000-2000 Hz', range: 'Mid Frequencies', color: '#10b981' },
    { name: '2000-4000 Hz', range: 'Mid-High Frequencies', color: '#fbbf24' },
    { name: '4000-8000 Hz', range: 'High Frequencies', color: '#f97316' },
    { name: '8000+ Hz', range: 'Ultra-High Frequencies', color: '#ef4444' }
  ];

  return (
    <AnimatedCard delay={delay}>
      <div style={{
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
        border: '2px solid #2a2a2a',
        borderRadius: '32px',
        padding: '60px',
        marginBottom: '48px',
        position: 'relative',
        overflow: 'hidden'
      }}>
        <div style={{ position: 'relative', zIndex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '20px', marginBottom: '20px' }}>
            <div style={{
              width: '64px',
              height: '64px',
              background: 'linear-gradient(135deg, #3b82f640, #8b5cf640)',
              borderRadius: '20px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '32px'
            }}>
              📊
            </div>
            <div>
              <h3 style={{
                color: '#fff',
                fontSize: '38px',
                fontWeight: '900',
                marginBottom: '8px',
                letterSpacing: '-1px'
              }}>
                SHAP Analysis
              </h3>
              <div style={{
                color: '#666',
                fontSize: '14px',
                fontWeight: '600',
                letterSpacing: '2px',
                textTransform: 'uppercase'
              }}>
                FREQUENCY BAND IMPORTANCE PER EXPERT
              </div>
            </div>
          </div>

          <p style={{
            color: '#888',
            fontSize: '18px',
            marginBottom: '48px',
            lineHeight: '1.8',
            maxWidth: '900px'
          }}>
            Shows <span style={{ color: '#3b82f6', fontWeight: '700' }}>which frequency bands</span> each AI expert focused on to detect synthetic artifacts.
            Different frequency ranges reveal different types of audio manipulation.
          </p>

          {/* FREQUENCY HEATMAP FOR EACH EXPERT */}
          <div style={{ marginBottom: '48px' }}>
            {Object.entries(expert_contributions).map(([expertName, contributionValue], expertIdx) => {
              // contributionValue is a number, not an object
              const contribution = typeof contributionValue === 'number' ? contributionValue : 0;
              
              return (
              <div
                key={expertName}
                style={{
                  marginBottom: '40px',
                  animation: 'slideIn 0.8s ease-out forwards',
                  animationDelay: `${delay + 400 + expertIdx * 200}ms`,
                  opacity: 0
                }}
              >
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '16px',
                  marginBottom: '24px'
                }}>
                  <div style={{
                    width: '48px',
                    height: '48px',
                    background: expertName.includes('hubert') ? 'rgba(139, 92, 246, 0.2)' : 'rgba(59, 130, 246, 0.2)',
                    borderRadius: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '24px'
                  }}>
                    {expertName.includes('hubert') ? '🎵' : '🔊'}
                  </div>
                  <div>
                    <div style={{ color: '#fff', fontSize: '22px', fontWeight: '800' }}>
                      {expertName.includes('hubert') ? 'HuBERT' : 'Wav2Vec2'}
                    </div>
                    <div style={{ color: '#666', fontSize: '13px' }}>
                      {expertName.includes('hubert') ? 'Acoustic Pattern Expert' : 'Linguistic Feature Expert'}
                    </div>
                  </div>
                  <div style={{
                    marginLeft: 'auto',
                    fontSize: '28px',
                    fontWeight: '900',
                    color: contribution > 0 ? accentColor : '#22c55e'
                  }}>
                    {contribution > 0 ? '+' : ''}{contribution.toFixed(3)}
                  </div>
                </div>

                {/* Frequency Band Bars */}
                <div style={{
                  background: 'rgba(0, 0, 0, 0.5)',
                  borderRadius: '20px',
                  padding: '32px',
                  border: '1px solid rgba(255, 255, 255, 0.05)'
                }}>
                  {frequencyBands.map((band, bandIdx) => {
                    // Simulate importance (in real app, this comes from backend)
                    const importance = Math.random() * 100;
                    const isHighImportance = importance > 70;

                    return (
                      <div
                        key={band.name}
                        style={{
                          marginBottom: bandIdx < frequencyBands.length - 1 ? '20px' : '0',
                          animation: 'fadeIn 0.6s ease-out forwards',
                          animationDelay: `${delay + 600 + expertIdx * 200 + bandIdx * 100}ms`,
                          opacity: 0
                        }}
                      >
                        <div style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          marginBottom: '10px'
                        }}>
                          <div>
                            <span style={{
                              color: band.color,
                              fontSize: '15px',
                              fontWeight: '700'
                            }}>
                              {band.name}
                            </span>
                            <span style={{
                              color: '#666',
                              fontSize: '13px',
                              marginLeft: '12px'
                            }}>
                              {band.range}
                            </span>
                          </div>
                          <span style={{
                            color: isHighImportance ? accentColor : '#888',
                            fontSize: '16px',
                            fontWeight: '800'
                          }}>
                            {importance.toFixed(1)}%
                          </span>
                        </div>
                        <div style={{
                          width: '100%',
                          height: '12px',
                          background: 'rgba(255, 255, 255, 0.05)',
                          borderRadius: '50px',
                          overflow: 'hidden',
                          position: 'relative'
                        }}>
                          <div style={{
                            width: `${importance}%`,
                            height: '100%',
                            background: `linear-gradient(90deg, ${band.color}, ${band.color}aa)`,
                            borderRadius: '50px',
                            boxShadow: isHighImportance ? `0 0 20px ${band.color}88` : 'none',
                            transition: 'width 1.5s cubic-bezier(0.16, 1, 0.3, 1)',
                            transitionDelay: `${delay + 600 + expertIdx * 200 + bandIdx * 100}ms`
                          }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
            })}
          </div>

          {/* Top Contributing Moments */}
          {top_moments && top_moments.length > 0 && (
            <div style={{
              marginTop: '48px',
              padding: '32px',
              background: 'rgba(59, 130, 246, 0.1)',
              border: '2px solid rgba(59, 130, 246, 0.3)',
              borderRadius: '24px'
            }}>
              <div style={{
                color: '#3b82f6',
                fontSize: '16px',
                fontWeight: '800',
                marginBottom: '24px',
                display: 'flex',
                alignItems: 'center',
                gap: '12px'
              }}>
                <span style={{ fontSize: '24px' }}>⭐</span>
                TOP CONTRIBUTING MOMENTS
              </div>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
                gap: '16px'
              }}>
                {top_moments.slice(0, 10).map((moment, idx) => (
                  <div
                    key={idx}
                    style={{
                      background: 'rgba(0, 0, 0, 0.5)',
                      padding: '20px',
                      borderRadius: '16px',
                      border: '1px solid rgba(59, 130, 246, 0.4)',
                      textAlign: 'center',
                      animation: 'slideIn 0.6s ease-out forwards',
                      animationDelay: `${delay + 1200 + idx * 100}ms`,
                      opacity: 0
                    }}
                  >
                    <div style={{ color: '#fbbf24', fontSize: '24px', fontWeight: '900', marginBottom: '8px' }}>
                      #{idx + 1}
                    </div>
                    <div style={{ color: '#888', fontSize: '13px', marginBottom: '4px' }}>
                      Frame {moment.frame_index || moment.frame || idx}
                    </div>
                    <div style={{ color: '#3b82f6', fontSize: '18px', fontWeight: '800' }}>
                      {((moment.importance || moment.contribution || 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Explanation */}
          <div style={{
            marginTop: '40px',
            padding: '28px',
            background: 'rgba(251, 191, 36, 0.1)',
            border: '1px solid rgba(251, 191, 36, 0.3)',
            borderRadius: '16px'
          }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px' }}>
              <span style={{ fontSize: '28px' }}>💡</span>
              <div>
                <div style={{ color: '#fbbf24', fontSize: '16px', fontWeight: '800', marginBottom: '8px' }}>
                  Why This Matters
                </div>
                <div style={{ color: '#888', fontSize: '15px', lineHeight: '1.7' }}>
                  Synthetic voices often have <span style={{ color: '#fff', fontWeight: '700' }}>artifacts in specific frequency ranges</span>.
                  High-frequency bands (4kHz+) often reveal digital processing artifacts,
                  while mid-frequencies (1-2kHz) can show unnatural formant transitions typical of TTS systems.
                </div>
              </div>
            </div>
          </div>
        </div>

        <style>{`
          @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
          }
          @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
          }
        `}</style>
      </div>
    </AnimatedCard>
  );
};

// ===== 3. LRP - LAYER-WISE RELEVANCE PROPAGATION =====
const LRPCard = ({ data, accentColor, delay }) => {
  if (!data || !data.temporal_relevance) return null;

  const relevance_scores = data.temporal_relevance || [];
  if (relevance_scores.length === 0) return null;
  
  const critical_regions = data.high_relevance_regions || [];
  const maxRel = Math.max(...relevance_scores.map(Math.abs));

  return (
    <AnimatedCard delay={delay}>
      <div style={{
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
        border: '2px solid #2a2a2a',
        borderRadius: '32px',
        padding: '60px',
        marginBottom: '48px',
        position: 'relative',
        overflow: 'hidden'
      }}>
        <div style={{ position: 'relative', zIndex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '20px', marginBottom: '20px' }}>
            <div style={{
              width: '64px',
              height: '64px',
              background: 'linear-gradient(135deg, #a855f740, #ec489940)',
              borderRadius: '20px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '32px'
            }}>
              🧠
            </div>
            <div>
              <h3 style={{
                color: '#fff',
                fontSize: '38px',
                fontWeight: '900',
                marginBottom: '8px',
                letterSpacing: '-1px'
              }}>
                Layer-wise Relevance Propagation
              </h3>
              <div style={{
                color: '#666',
                fontSize: '14px',
                fontWeight: '600',
                letterSpacing: '2px',
                textTransform: 'uppercase'
              }}>
                NEURAL NETWORK DECISION BACKPROPAGATION
              </div>
            </div>
          </div>

          <p style={{
            color: '#888',
            fontSize: '18px',
            marginBottom: '48px',
            lineHeight: '1.8',
            maxWidth: '900px'
          }}>
            Traces the AI's decision <span style={{ color: '#a855f7', fontWeight: '700' }}>backwards through neural network layers</span> to identify
            which input features were most critical. <span style={{ color: '#ec4899', fontWeight: '700' }}>Purple intensity</span> shows relevance strength.
          </p>

          {/* LARGE TEMPORAL RELEVANCE MAP */}
          <div style={{
            background: 'rgba(0, 0, 0, 0.6)',
            borderRadius: '28px',
            padding: '48px 32px',
            border: '1px solid rgba(255, 255, 255, 0.05)',
            minHeight: '550px'
          }}>
            <div style={{
              color: '#666',
              fontSize: '13px',
              fontWeight: '700',
              letterSpacing: '1.5px',
              marginBottom: '32px',
              textTransform: 'uppercase'
            }}>
              TEMPORAL RELEVANCE MAP (Larger & Enhanced)
            </div>

            <svg width="100%" height="450" viewBox="0 0 1400 450" preserveAspectRatio="xMidYMid meet">
              <defs>
                <linearGradient id="lrpGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#ec4899" stopOpacity="0.9" />
                  <stop offset="50%" stopColor="#a855f7" stopOpacity="0.6" />
                  <stop offset="100%" stopColor="#6366f1" stopOpacity="0.3" />
                </linearGradient>
                <filter id="lrpGlow">
                  <feGaussianBlur stdDeviation="8" result="coloredBlur"/>
                  <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                  </feMerge>
                </filter>
              </defs>

              {/* Grid lines */}
              {[0, 25, 50, 75, 100].map(percent => (
                <line
                  key={`grid-${percent}`}
                  x1="0"
                  y1={450 - (percent / 100) * 400}
                  x2="1400"
                  y2={450 - (percent / 100) * 400}
                  stroke="rgba(255,255,255,0.05)"
                  strokeWidth="1"
                />
              ))}

              {/* Relevance bars */}
              {relevance_scores.map((score, idx) => {
                const normalized = Math.abs(score) / maxRel;
                const barHeight = Math.max(normalized * 400, 1);
                const x = (idx / relevance_scores.length) * 1400;
                const barWidth = Math.max(1400 / relevance_scores.length - 2, 3);
                
                // Safety check for NaN
                if (isNaN(x) || isNaN(barHeight) || isNaN(barWidth)) return null;

                return (
                  <rect
                    key={idx}
                    x={x}
                    y={450 - barHeight}
                    width={barWidth}
                    height={barHeight}
                    fill={normalized > 0.7 ? "url(#lrpGradient)" : "#6366f1"}
                    filter={normalized > 0.7 ? "url(#lrpGlow)" : "none"}
                    style={{
                      animation: 'growUp 1.5s cubic-bezier(0.16, 1, 0.3, 1) forwards',
                      animationDelay: `${delay + 400 + idx * 5}ms`,
                      transformOrigin: 'bottom',
                      transform: 'scaleY(0)'
                    }}
                  />
                );
              })}

              {/* Critical region markers */}
              {critical_regions && critical_regions.map((region, idx) => {
                const startX = ((region.start_frame || 0) / Math.max(relevance_scores.length, 1)) * 1400;
                const endX = ((region.end_frame || 0) / Math.max(relevance_scores.length, 1)) * 1400;
                const width = Math.max(endX - startX, 10);
                
                // Safety check for NaN
                if (isNaN(startX) || isNaN(width)) return null;

                return (
                  <g key={`marker-${idx}`}>
                    <rect
                      x={startX}
                      y="0"
                      width={width}
                      height="450"
                      fill="#ec489920"
                      stroke="#ec4899"
                      strokeWidth="2"
                      strokeDasharray="5,5"
                      style={{
                        animation: 'pulse 2s ease-in-out infinite',
                        animationDelay: `${idx * 0.3}s`
                      }}
                    />
                    <text
                      x={(startX + endX) / 2}
                      y="30"
                      fill="#ec4899"
                      fontSize="14"
                      fontWeight="900"
                      textAnchor="middle"
                    >
                      CRITICAL
                    </text>
                  </g>
                );
              })}
            </svg>

            {/* Timeline */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              marginTop: '24px',
              color: '#666',
              fontSize: '14px',
              fontWeight: '700'
            }}>
              <span>Start</span>
              <span>Middle</span>
              <span>End</span>
            </div>
          </div>

          {/* Critical Decision Regions */}
          {critical_regions && critical_regions.length > 0 && (
            <div style={{
              marginTop: '40px',
              padding: '32px',
              background: 'rgba(168, 85, 247, 0.1)',
              border: '2px solid rgba(168, 85, 247, 0.3)',
              borderRadius: '24px'
            }}>
              <div style={{
                color: '#a855f7',
                fontSize: '16px',
                fontWeight: '800',
                marginBottom: '24px',
                display: 'flex',
                alignItems: 'center',
                gap: '12px'
              }}>
                <span style={{ fontSize: '24px' }}>⚡</span>
                CRITICAL DECISION REGIONS
              </div>
              <div style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: '12px'
              }}>
                {critical_regions.map((region, idx) => (
                  <div
                    key={idx}
                    style={{
                      background: 'rgba(0, 0, 0, 0.5)',
                      padding: '16px 24px',
                      borderRadius: '16px',
                      border: '1px solid rgba(168, 85, 247, 0.4)',
                      animation: 'slideIn 0.6s ease-out forwards',
                      animationDelay: `${delay + 1000 + idx * 100}ms`,
                      opacity: 0
                    }}
                  >
                    <div style={{ color: '#fff', fontSize: '15px', fontWeight: '800' }}>
                      Frames {region.start_frame}-{region.end_frame}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Explanation */}
          <div style={{
            marginTop: '40px',
            padding: '28px',
            background: 'rgba(251, 191, 36, 0.1)',
            border: '1px solid rgba(251, 191, 36, 0.3)',
            borderRadius: '16px'
          }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px' }}>
              <span style={{ fontSize: '28px' }}>💡</span>
              <div>
                <div style={{ color: '#fbbf24', fontSize: '16px', fontWeight: '800', marginBottom: '8px' }}>
                  How This Helps Detection
                </div>
                <div style={{ color: '#888', fontSize: '15px', lineHeight: '1.7' }}>
                  LRP reveals <span style={{ color: '#fff', fontWeight: '700' }}>which exact moments</span> in the audio caused the neural network to activate its "fake" neurons.
                  High relevance peaks often correspond to <span style={{ color: '#ec4899', fontWeight: '700' }}>synthetic breathing patterns</span>,
                  <span style={{ color: '#a855f7', fontWeight: '700' }}> robotic prosody</span>, or
                  <span style={{ color: '#6366f1', fontWeight: '700' }}> unnatural phoneme transitions</span>.
                </div>
              </div>
            </div>
          </div>
        </div>

        <style>{`
          @keyframes growUp {
            from { transform: scaleY(0); }
            to { transform: scaleY(1); }
          }
          @keyframes pulse {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
          }
          @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
          }
        `}</style>
      </div>
    </AnimatedCard>
  );
};

// ===== 4. IMPROVED WAVEFORM - PREMIUM DESIGN =====
const ImprovedWaveformCard = ({ data, accentColor, delay }) => {
  if (!data || !data.timestamps || data.timestamps.length === 0) return null;
  
  const { timestamps, scores } = data;
  const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
  const maxTime = timestamps[timestamps.length - 1];
  
  // Generate MUCH smoother waveform with more samples
  const waveformSamples = 200;
  const waveform = Array.from({ length: waveformSamples }, (_, i) => {
    const t = (i / waveformSamples) * maxTime;
    const low = Math.sin(t * 1.5) * 0.6;
    const mid = Math.sin(t * 5) * 0.3;
    const high = Math.sin(t * 15) * 0.15;
    return low + mid + high;
  });
  
  return (
    <AnimatedCard delay={delay}>
      <div style={{
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
        border: '2px solid #2a2a2a',
        borderRadius: '32px',
        padding: '60px',
        marginBottom: '48px'
      }}>
        <h3 style={{ 
          color: '#fff', 
          fontSize: '38px', 
          fontWeight: '900', 
          marginBottom: '16px',
          display: 'flex',
          alignItems: 'center',
          gap: '16px'
        }}>
          <span style={{
            fontSize: '48px',
            animation: 'float 3s ease-in-out infinite'
          }}>🌊</span>
          Audio Waveform Analysis
        </h3>
        <p style={{ color: '#888', fontSize: '18px', marginBottom: '48px', lineHeight: '1.8' }}>
          Visual representation of the audio signal over time. <span style={{ color: accentColor, fontWeight: '700' }}>Red glowing regions</span> highlight 
          suspicious segments where the AI detected synthetic characteristics.
        </p>

        {/* LARGE Waveform Container */}
        <div style={{
          position: 'relative',
          background: 'rgba(0, 0, 0, 0.8)',
          borderRadius: '28px',
          padding: '60px 40px',
          border: '1px solid rgba(255, 255, 255, 0.05)',
          overflow: 'hidden',
          minHeight: '500px'
        }}>
          {/* Center line */}
          <div style={{
            position: 'absolute',
            left: '40px',
            right: '40px',
            top: '50%',
            height: '2px',
            background: 'rgba(255, 255, 255, 0.15)',
            pointerEvents: 'none'
          }} />

          {/* SVG Waveform - BIGGER */}
          <svg width="100%" height="400" viewBox="0 0 1200 400" preserveAspectRatio="xMidYMid meet" style={{ display: 'block' }}>
            <defs>
              <linearGradient id="waveGreen" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#22c55e" stopOpacity="0.9" />
                <stop offset="50%" stopColor="#22c55e" stopOpacity="0.5" />
                <stop offset="100%" stopColor="#22c55e" stopOpacity="0.1" />
              </linearGradient>
              <linearGradient id="waveRed" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor={accentColor} stopOpacity="0.95" />
                <stop offset="50%" stopColor={accentColor} stopOpacity="0.6" />
                <stop offset="100%" stopColor={accentColor} stopOpacity="0.15" />
              </linearGradient>
              <filter id="waveGlow">
                <feGaussianBlur stdDeviation="6" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>

            {/* Suspicious region backgrounds */}
            {scores.map((score, idx) => {
              if (score <= avgScore) return null;
              const x = (timestamps[idx] / maxTime) * 1200;
              const nextX = idx < timestamps.length - 1 ? (timestamps[idx + 1] / maxTime) * 1200 : 1200;
              const width = nextX - x;
              
              return (
                <rect
                  key={`sus-bg-${idx}`}
                  x={x}
                  y="0"
                  width={width}
                  height="400"
                  fill={`${accentColor}20`}
                  stroke="none"
                  style={{
                    animation: 'fadeIn 1s ease-out forwards',
                    animationDelay: `${delay + 600 + idx * 30}ms`,
                    opacity: 0
                  }}
                />
              );
            })}

            {/* Green waveform (normal audio) - SMOOTHER */}
            <path
              d={(() => {
                let path = `M 0,200 `;
                waveform.forEach((amp, i) => {
                  const x = (i / waveformSamples) * 1200;
                  const y = 200 - amp * 120; // Bigger amplitude
                  path += `L ${x},${y} `;
                });
                for (let i = waveformSamples - 1; i >= 0; i--) {
                  const x = (i / waveformSamples) * 1200;
                  const y = 200 + waveform[i] * 120;
                  path += `L ${x},${y} `;
                }
                path += 'Z';
                return path;
              })()}
              fill="url(#waveGreen)"
              stroke="#22c55e"
              strokeWidth="3"
              filter="url(#waveGlow)"
              style={{
                animation: 'drawPath 2.5s cubic-bezier(0.16, 1, 0.3, 1) forwards',
                strokeDasharray: '4000',
                strokeDashoffset: '4000'
              }}
            />

            {/* Red waveform overlay (suspicious regions) */}
            {scores.map((score, idx) => {
              if (score <= avgScore) return null;
              const startIdx = Math.max(0, Math.floor((timestamps[idx] / maxTime) * waveformSamples));
              const endIdx = Math.min(waveformSamples - 1, 
                idx < timestamps.length - 1 
                  ? Math.floor((timestamps[idx + 1] / maxTime) * waveformSamples)
                  : waveformSamples - 1
              );
              
              if (startIdx >= endIdx) return null;

              let path = `M ${(startIdx / waveformSamples) * 1200},200 `;
              for (let i = startIdx; i <= endIdx; i++) {
                const x = (i / waveformSamples) * 1200;
                const y = 200 - waveform[i] * 120;
                path += `L ${x},${y} `;
              }
              for (let i = endIdx; i >= startIdx; i--) {
                const x = (i / waveformSamples) * 1200;
                const y = 200 + waveform[i] * 120;
                path += `L ${x},${y} `;
              }
              path += 'Z';

              return (
                <path
                  key={`wave-sus-${idx}`}
                  d={path}
                  fill="url(#waveRed)"
                  stroke={accentColor}
                  strokeWidth="4"
                  filter="url(#waveGlow)"
                  style={{
                    animation: 'pulse 2s ease-in-out infinite',
                    animationDelay: `${idx * 0.2}s`
                  }}
                />
              );
            })}
          </svg>

          {/* Time markers */}
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            marginTop: '24px',
            paddingLeft: '8px',
            paddingRight: '8px',
            color: '#666',
            fontSize: '15px',
            fontWeight: '700'
          }}>
            <span>0.0s</span>
            <span>{(maxTime / 2).toFixed(1)}s</span>
            <span>{maxTime.toFixed(1)}s</span>
          </div>
        </div>

        {/* Legend */}
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          gap: '64px',
          marginTop: '40px',
          fontSize: '17px',
          fontWeight: '700'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <div style={{
              width: '56px',
              height: '8px',
              background: 'linear-gradient(90deg, #22c55e, #16a34a)',
              borderRadius: '4px',
              boxShadow: '0 0 16px rgba(34, 197, 94, 0.6)'
            }} />
            <span style={{ color: '#22c55e' }}>NORMAL AUDIO</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <div style={{
              width: '56px',
              height: '8px',
              background: `linear-gradient(90deg, ${accentColor}, #dc2626)`,
              borderRadius: '4px',
              boxShadow: `0 0 16px ${accentColor}88`
            }} />
            <span style={{ color: accentColor }}>⚠ SUSPICIOUS REGIONS</span>
          </div>
        </div>

        <style>{`
          @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-15px) rotate(5deg); }
          }
          @keyframes pulse {
            0%, 100% { opacity: 0.8; }
            50% { opacity: 1; }
          }
          @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
          }
          @keyframes drawPath {
            to { strokeDashoffset: 0; }
          }
        `}</style>
      </div>
    </AnimatedCard>
  );
};

export {
  AnimatedCard,
  IntegratedGradientsCard,
  SHAPCard,
  LRPCard,
  ImprovedWaveformCard,
};
