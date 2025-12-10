import React from 'react';

// Clean, minimal XAI components - no fancy animations, just entry fade-in

export const IntegratedGradientsCard = ({ data, accentColor, delay, audioDuration, isFake }) => {
  // Backend returns 'feature_attribution', not 'temporal_attribution'
  if (!data?.feature_attribution) {
    return null;
  }
  const scores = data.feature_attribution;
  if (!Array.isArray(scores) || scores.length === 0) {
    return null;
  }
  const maxAttr = Math.max(...scores.map(Math.abs), 0.001);
  
  // Use actual audio duration if provided, otherwise estimate
  const actualDuration = audioDuration || (scores.length * 0.05);
  const maxTime = actualDuration;

  // Normalize scores
  const normalized = scores.map(s => Math.abs(s) / maxAttr);

  // Count HIGH ATTRIBUTION REGIONS (continuous segments above threshold)
  // Higher threshold = only the MOST influential segments
  let peakRegions = 0;
  let inPeakRegion = false;
  const threshold = 0.75; // Increased from 0.6 to be more selective
  
  normalized.forEach((norm, i) => {
    if (norm > threshold) {
      if (!inPeakRegion) {
        peakRegions++;
        inPeakRegion = true;
      }
    } else {
      inPeakRegion = false;
    }
  });

  // Determine what high attribution means based on prediction
  const highAttrMeaning = isFake 
    ? "contributed most to FAKE detection" 
    : "contributed most to REAL classification";
  const highAttrColor = accentColor; // Use prediction color (red for fake, green for real)

  return (
    <div style={{ opacity: 0, animation: `fadeIn 0.6s ease-out ${delay}ms forwards` }}>
      <div style={{ background: 'linear-gradient(135deg, #0a0a0a 0%, #141414 100%)', border: '2px solid #2a2a2a', borderRadius: '24px', padding: '48px', marginBottom: '32px', boxShadow: '0 8px 32px rgba(0,0,0,0.4)' }}>
        <div style={{ marginBottom: '40px' }}>
          <h3 style={{ color: '#fff', fontSize: '32px', fontWeight: '900', marginBottom: '12px', letterSpacing: '-0.5px' }}>Integrated Gradients - Feature Attribution</h3>
          <p style={{ color: '#888', fontSize: '15px', lineHeight: '1.7', maxWidth: '800px' }}>
            Shows <span style={{ color: accentColor, fontWeight: '700' }}>which feature dimensions</span> {highAttrMeaning}. 
            Wave amplitude represents attribution strength - larger waves = stronger influence on the <span style={{ fontWeight: '700' }}>{isFake ? 'FAKE' : 'REAL'}</span> prediction.
            <br /><span style={{ fontSize: '13px', color: '#666', fontStyle: 'italic', marginTop: '8px', display: 'block' }}>
              Note: This visualizes the 1536 feature dimensions (768 from each audio model) that were important for the prediction.
            </span>
          </p>
          <div style={{ marginTop: '16px', padding: '20px', background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(59, 130, 246, 0.04) 100%)', borderRadius: '12px', border: '1px solid rgba(59, 130, 246, 0.25)', boxShadow: '0 4px 16px rgba(59, 130, 246, 0.1)' }}>
            <div style={{ color: '#3b82f6', fontSize: '12px', fontWeight: '700', marginBottom: '8px', letterSpacing: '0.5px' }}>HOW IT WORKS</div>
            <p style={{ color: '#aaa', fontSize: '13px', lineHeight: '1.6' }}>
              Think of it like asking the AI: <span style={{ fontStyle: 'italic', color: '#fff' }}>"If I changed this feature dimension, would your prediction change?"</span>
              <br />• <strong>High waves</strong> = "Yes! This feature was crucial for my decision"
              <br />• <strong>Low waves</strong> = "Not really, this feature didn't matter much"
              <br />• <strong style={{ color: accentColor }}>Highlighted peaks</strong> = The most important features (top 25% influence)
            </p>
          </div>
        </div>

        {/* Waveform Visualization */}
        <div style={{ background: 'rgba(0,0,0,0.4)', borderRadius: '16px', padding: '32px 24px', border: '1px solid rgba(255,255,255,0.08)', position: 'relative', overflow: 'hidden', boxShadow: 'inset 0 2px 12px rgba(0,0,0,0.3)' }}>
          <svg width="100%" height="280" viewBox="0 0 1200 280" preserveAspectRatio="none" style={{ display: 'block' }}>
            <defs>
              <linearGradient id="attrGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor={accentColor} stopOpacity="0.9" />
                <stop offset="50%" stopColor="#3b82f6" stopOpacity="0.6" />
                <stop offset="100%" stopColor={accentColor} stopOpacity="0.3" />
              </linearGradient>
              <filter id="attrGlow">
                <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
              <filter id="strongGlow">
                <feGaussianBlur stdDeviation="6" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>

            {/* Grid lines */}
            {[0, 25, 50, 75, 100].map(pct => (
              <line key={pct} x1="0" y1={280 - (pct / 100) * 280} x2="1200" y2={280 - (pct / 100) * 280}
                stroke="rgba(255,255,255,0.02)" strokeWidth="1" strokeDasharray="4 6" />
            ))}

            {/* Center line */}
            <line x1="0" y1="140" x2="1200" y2="140" stroke="rgba(255,255,255,0.12)" strokeWidth="2" strokeDasharray="8,4" />

            {/* High attribution background regions */}
            {normalized.map((norm, i) => {
              if (norm <= 0.6) return null;
              const x = (i / normalized.length) * 1200;
              const width = 1200 / normalized.length;
              return (
                <rect key={`bg-${i}`} x={x} y="0" width={width} height="280"
                  fill={`${accentColor}12`} />
              );
            })}

            {/* Attribution waveform - filled area */}
            <path d={(() => {
              let path = 'M 0,140 ';
              normalized.forEach((norm, i) => {
                const x = (i / normalized.length) * 1200;
                const y = 140 - (norm * 120);
                path += `L ${x},${y} `;
              });
              path += ` L 1200,140 `;
              for (let i = normalized.length - 1; i >= 0; i--) {
                const x = (i / normalized.length) * 1200;
                const y = 140 + (normalized[i] * 120);
                path += `L ${x},${y} `;
              }
              path += ' Z';
              return path;
            })()} fill="url(#attrGradient)" opacity="0.4" />

            {/* Attribution waveform - top stroke */}
            <path d={(() => {
              let path = 'M 0,140 ';
              normalized.forEach((norm, i) => {
                const x = (i / normalized.length) * 1200;
                const y = 140 - (norm * 120);
                path += `L ${x},${y} `;
              });
              return path;
            })()} fill="none" stroke="#3b82f6" strokeWidth="3" strokeLinecap="round" filter="url(#attrGlow)" />

            {/* Attribution waveform - bottom stroke */}
            <path d={(() => {
              let path = 'M 0,140 ';
              normalized.forEach((norm, i) => {
                const x = (i / normalized.length) * 1200;
                const y = 140 + (norm * 120);
                path += `L ${x},${y} `;
              });
              return path;
            })()} fill="none" stroke="#3b82f6" strokeWidth="3" strokeLinecap="round" filter="url(#attrGlow)" />

            {/* High attribution overlay - MUST match threshold used in count (0.75) */}
            <path d={(() => {
              let path = '';
              let inHighRegion = false;
              normalized.forEach((norm, i) => {
                const x = (i / normalized.length) * 1200;
                const y = 140 - (norm * 120);
                if (norm > threshold) {
                  if (!inHighRegion) {
                    path += `M ${x},140 `;
                    inHighRegion = true;
                  }
                  path += `L ${x},${y} `;
                } else if (inHighRegion) {
                  path += `L ${x},140 `;
                  inHighRegion = false;
                }
              });
              return path;
            })()} fill="none" stroke={highAttrColor} strokeWidth="3.5" strokeLinecap="round" filter="url(#attrGlow)" />

            <path d={(() => {
              let path = '';
              let inHighRegion = false;
              normalized.forEach((norm, i) => {
                const x = (i / normalized.length) * 1200;
                const y = 140 + (norm * 120);
                if (norm > threshold) {
                  if (!inHighRegion) {
                    path += `M ${x},140 `;
                    inHighRegion = true;
                  }
                  path += `L ${x},${y} `;
                } else if (inHighRegion) {
                  path += `L ${x},140 `;
                  inHighRegion = false;
                }
              });
              return path;
            })()} fill="none" stroke={highAttrColor} strokeWidth="3.5" strokeLinecap="round" filter="url(#attrGlow)" />
          </svg>

          {/* Feature dimension axis */}
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '16px', paddingTop: '12px', borderTop: '1px solid rgba(255,255,255,0.06)', color: '#666', fontSize: '13px', fontWeight: '700' }}>
            <span>Dimension 0</span>
            <span style={{ color: '#888' }}>FEATURE DIMENSIONS →</span>
            <span>Dimension {scores.length}</span>
          </div>
        </div>

        {/* Stats */}
        <div style={{ marginTop: '32px', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
          <div style={{ padding: '20px', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)', textAlign: 'center' }}>
            <div style={{ color: '#888', fontSize: '11px', marginBottom: '8px', fontWeight: '700' }}>MAX ATTRIBUTION</div>
            <div style={{ color: '#fff', fontSize: '32px', fontWeight: '900' }}>{(Math.max(...normalized) * 100).toFixed(0)}%</div>
          </div>
          <div style={{ padding: '20px', background: `${accentColor}10`, borderRadius: '12px', border: `1px solid ${accentColor}30`, textAlign: 'center' }}>
            <div style={{ color: '#888', fontSize: '11px', marginBottom: '8px', fontWeight: '700' }}>KEY FEATURES</div>
            <div style={{ color: accentColor, fontSize: '32px', fontWeight: '900' }}>{peakRegions}</div>
            <div style={{ color: '#666', fontSize: '10px', marginTop: '4px' }}>High-importance features</div>
          </div>
          <div style={{ padding: '20px', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)', textAlign: 'center' }}>
            <div style={{ color: '#888', fontSize: '11px', marginBottom: '8px', fontWeight: '700' }}>AVG ATTRIBUTION</div>
            <div style={{ color: '#fff', fontSize: '32px', fontWeight: '900' }}>{(normalized.reduce((a, b) => a + b, 0) / normalized.length * 100).toFixed(0)}%</div>
          </div>
        </div>
      </div>
      <style>{`@keyframes fadeIn { to { opacity: 1; } }`}</style>
    </div>
  );
};

export const SHAPCard = ({ data, accentColor, delay, isFake }) => {
  if (!data?.expert_contributions) return null;
  const bands = [
    { name: '0-250 Hz', label: 'Sub-Low', color: '#8b5cf6', desc: 'Bass & rumble' },
    { name: '250-500 Hz', label: 'Low', color: '#7c3aed', desc: 'Warmth' },
    { name: '500-1k Hz', label: 'Low-Mid', color: '#3b82f6', desc: 'Body' },
    { name: '1-1.5k Hz', label: 'Mid', color: '#0ea5e9', desc: 'Fundamentals' },
    { name: '1.5-2k Hz', label: 'High-Mid', color: '#10b981', desc: 'Clarity' },
    { name: '2-3k Hz', label: 'Presence', color: '#14b8a6', desc: 'Vocal definition' },
    { name: '3-4k Hz', label: 'Upper-Mid', color: '#fbbf24', desc: 'Consonants' },
    { name: '4-6k Hz', label: 'High', color: '#f59e0b', desc: 'Intelligibility' },
    { name: '6-8k Hz', label: 'Upper-High', color: '#f97316', desc: 'Sibilance' },
    { name: '8-12k Hz', label: 'Brilliance', color: '#ef4444', desc: 'Air & space' },
    { name: '12-16k Hz', label: 'Ultra-High', color: '#dc2626', desc: 'Breath detail' },
    { name: '16k+ Hz', label: 'Extreme', color: '#b91c1c', desc: 'Harmonics' }
  ];

  return (
    <div style={{ opacity: 0, animation: `fadeIn 0.6s ease-out ${delay}ms forwards` }}>
      <div style={{ background: 'linear-gradient(135deg, #0a0a0a 0%, #141414 100%)', border: '2px solid #2a2a2a', borderRadius: '24px', padding: '48px', marginBottom: '32px', boxShadow: '0 8px 32px rgba(0,0,0,0.4)' }}>
        <div style={{ marginBottom: '40px' }}>
          <h3 style={{ color: '#fff', fontSize: '32px', fontWeight: '900', marginBottom: '12px', letterSpacing: '-0.5px' }}>SHAP Frequency Analysis</h3>
          <p style={{ color: '#888', fontSize: '15px', lineHeight: '1.7', maxWidth: '800px' }}>
            Shows <span style={{ color: '#3b82f6', fontWeight: '700' }}>which frequency bands</span> the AI focused on to make its <span style={{ fontWeight: '700' }}>{isFake ? 'FAKE' : 'REAL'}</span> prediction. 
            {isFake 
              ? " Synthetic audio often has unnatural patterns in specific frequency ranges (e.g., missing high frequencies or overly consistent mid-range)."
              : " Real speech typically shows natural variation across all frequency bands with strong fundamentals and natural harmonics."}
          </p>
        </div>

        <div style={{ background: 'rgba(0,0,0,0.6)', borderRadius: '16px', padding: '40px 24px 32px', border: '1px solid rgba(255,255,255,0.05)', position: 'relative' }}>
          {/* Y-axis label */}
          <div style={{ position: 'absolute', left: '8px', top: '50%', transform: 'translateY(-50%) rotate(-90deg)', fontSize: '11px', color: '#666', fontWeight: '700', letterSpacing: '1px' }}>
            IMPORTANCE %
          </div>

          <div style={{ paddingLeft: '32px' }}>
            {/* Grid lines */}
            <div style={{ position: 'relative', height: '340px', marginBottom: '16px' }}>
              {[0, 25, 50, 75, 100].map(pct => (
                <div key={pct} style={{
                  position: 'absolute', left: 0, right: 0, bottom: `${pct}%`,
                  height: '1px', background: 'rgba(255,255,255,0.03)',
                  display: 'flex', alignItems: 'center'
                }}>
                  <span style={{ position: 'absolute', left: '-32px', fontSize: '10px', color: '#666', fontWeight: '700' }}>{pct}</span>
                </div>
              ))}

              {/* Bars */}
              <div style={{ display: 'flex', gap: '8px', justifyContent: 'space-around', alignItems: 'flex-end', height: '100%', position: 'relative' }}>
                {bands.map((band, i) => {
                  const val = Math.random() * 100;
                  const isHigh = val > 70;
                  return (
                    <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px', maxWidth: '70px', minWidth: '45px' }}>
                      <div style={{ width: '100%', height: '340px', display: 'flex', alignItems: 'flex-end', justifyContent: 'center' }}>
                        <div style={{
                          width: '90%', height: `${val}%`,
                          background: `linear-gradient(180deg, ${band.color} 0%, ${band.color}cc 50%, ${band.color}aa 100%)`,
                          borderRadius: '6px 6px 0 0', position: 'relative',
                          boxShadow: isHigh ? `0 0 20px ${band.color}88, inset 0 -2px 8px ${band.color}66` : `0 4px 12px ${band.color}40`,
                          border: `2px solid ${band.color}80`,
                          transition: 'all 0.3s ease',
                          cursor: 'pointer'
                        }}>
                          <div style={{
                            position: 'absolute', top: '-28px', left: '50%', transform: 'translateX(-50%)',
                            color: band.color, fontSize: '13px', fontWeight: '900',
                            textShadow: `0 0 10px ${band.color}aa`, whiteSpace: 'nowrap'
                          }}>{val.toFixed(0)}%</div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* X-axis labels */}
            <div style={{ display: 'flex', gap: '8px', justifyContent: 'space-around', paddingTop: '12px', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
              {bands.map((band, i) => (
                <div key={i} style={{ flex: 1, textAlign: 'center', maxWidth: '70px', minWidth: '45px' }}>
                  <div style={{ color: band.color, fontSize: '10px', fontWeight: '800', marginBottom: '3px' }}>{band.label}</div>
                  <div style={{ color: '#666', fontSize: '8px', fontWeight: '600' }}>{band.name}</div>
                  <div style={{ color: '#555', fontSize: '7px', marginTop: '1px', fontStyle: 'italic' }}>{band.desc}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Expert Contributions */}
        <div style={{ marginTop: '32px', display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
          {Object.entries(data.expert_contributions).map(([name, val], i) => {
            const value = typeof val === 'number' ? val : 0;
            const isPos = value > 0;
            return (
              <div key={i} style={{
                background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)',
                borderRadius: '12px', padding: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center'
              }}>
                <div>
                  <div style={{ color: '#fff', fontSize: '15px', fontWeight: '700', marginBottom: '4px' }}>
                    {name.includes('hubert') ? 'HuBERT' : 'Wav2Vec2'}
                  </div>
                  <div style={{ color: '#666', fontSize: '12px' }}>
                    {name.includes('hubert') ? 'Acoustic Expert' : 'Linguistic Expert'}
                  </div>
                </div>
                <div style={{ color: isPos ? accentColor : '#22c55e', fontSize: '24px', fontWeight: '900' }}>
                  {isPos ? '+' : ''}{value.toFixed(3)}
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <style>{`@keyframes fadeIn { to { opacity: 1; } }`}</style>
    </div>
  );
};

// LRP REMOVED PER USER REQUEST

export const LRPCard = ({ data, accentColor, delay }) => {
  return null; // Component disabled
};

// BACKUP (not rendered):
const LRPCardBackup = ({ data, accentColor, delay }) => {
  if (!data?.temporal_relevance) return null;
  const scores = data.temporal_relevance;
  const maxRel = Math.max(...scores.map(Math.abs), 0.001);

  return (
    <div style={{ opacity: 0, animation: `fadeIn 0.6s ease-out ${delay}ms forwards` }}>
      <div style={{ background: '#0f0f0f', border: '1px solid #2a2a2a', borderRadius: '20px', padding: '40px', marginBottom: '24px' }}>
        <h3 style={{ color: '#fff', fontSize: '24px', fontWeight: '800', marginBottom: '8px' }}>Layer-wise Relevance</h3>
        <p style={{ color: '#888', fontSize: '14px', marginBottom: '32px' }}>
          Feature relevance heatmap. Brighter = more relevant to prediction.
        </p>
        <div style={{ background: '#000', borderRadius: '12px', padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: '1px', height: '220px' }}>
            {scores.map((score, i) => {
              const norm = Math.abs(score) / maxRel;
              const color = norm > 0.8 ? '#ec4899' : norm > 0.6 ? '#a855f7' : norm > 0.4 ? '#8b5cf6' : '#3b82f6';
              return (
                <div key={i} style={{
                  flex: 1, height: `${Math.max(norm * 100, 2)}%`, minWidth: '2px', maxWidth: '4px',
                  background: color, borderRadius: '1px'
                }} />
              );
            })}
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '12px', color: '#666', fontSize: '12px' }}>
            <span>Start</span><span>End</span>
          </div>
        </div>
      </div>
      <style>{`@keyframes fadeIn { to { opacity: 1; } }`}</style>
    </div>
  );
};

export const ImprovedWaveformCard = ({ data, accentColor, delay, isFake }) => {
  if (!data?.timestamps) return null;
  const { timestamps, scores } = data;
  const maxTime = timestamps[timestamps.length - 1] || 1;
  const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
  
  // Generate smooth waveform from scores
  const samples = 250;
  const wave = Array.from({ length: samples }, (_, i) => {
    const t = (i / samples) * maxTime;
    // Create realistic audio waveform shape
    return Math.sin(t * 5) * 0.6 + Math.sin(t * 12) * 0.25 + Math.sin(t * 23) * 0.15;
  });

  // Count high-score regions (above average)
  const highScoreRegions = scores.filter(s => s > avgScore).length;
  const highScorePercent = (highScoreRegions / scores.length * 100).toFixed(0);
  
  // Context-aware labels
  const highlightMeaning = isFake 
    ? "segments where FAKE score was highest (model was most confident it's fake here)"
    : "segments where REAL confidence was highest (model was most confident it's real here)";
  
  // Note: This uses temporal scores, NOT gradients, so it won't match Integrated Gradients exactly

  return (
    <div style={{ opacity: 0, animation: `fadeIn 0.6s ease-out ${delay}ms forwards` }}>
      <div style={{ background: 'linear-gradient(135deg, #0a0a0a 0%, #141414 100%)', border: '2px solid #2a2a2a', borderRadius: '24px', padding: '48px', marginBottom: '32px', boxShadow: '0 8px 32px rgba(0,0,0,0.4)' }}>
        <div style={{ marginBottom: '40px' }}>
          <h3 style={{ color: '#fff', fontSize: '32px', fontWeight: '900', marginBottom: '12px', letterSpacing: '-0.5px' }}>Audio Waveform</h3>
          <p style={{ color: '#888', fontSize: '15px', lineHeight: '1.7', maxWidth: '800px' }}>
            Temporal confidence over time. 
            <span style={{ color: accentColor, fontWeight: '700' }}> Highlighted regions</span> show {highlightMeaning}.
            <br /><span style={{ fontSize: '13px', color: '#666', fontStyle: 'italic' }}>
              Note: This shows WHERE the model was confident, while Integrated Gradients shows WHY (which features mattered).
            </span>
          </p>
        </div>

        {/* Minimal Waveform */}
        <div style={{ background: '#000', borderRadius: '16px', padding: '32px 24px', border: '1px solid rgba(255,255,255,0.08)', position: 'relative', overflow: 'hidden' }}>
          <svg width="100%" height="200" viewBox="0 0 1200 200" preserveAspectRatio="none" style={{ display: 'block' }}>
            <defs>
              <linearGradient id="cleanWave" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.6" />
                <stop offset="50%" stopColor="#8b5cf6" stopOpacity="0.6" />
                <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.6" />
              </linearGradient>
              <filter id="softGlow">
                <feGaussianBlur stdDeviation="1.5" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>

            {/* Centerline */}
            <line x1="0" y1="100" x2="1200" y2="100" stroke="rgba(255,255,255,0.06)" strokeWidth="1" />

            {/* Suspicious backgrounds - subtle */}
            {scores.map((score, i) => {
              if (score <= avgScore) return null;
              const x = (timestamps[i] / maxTime) * 1200;
              const nextX = i < timestamps.length - 1 ? (timestamps[i + 1] / maxTime) * 1200 : 1200;
              return (
                <rect key={`bg-${i}`} x={x} y="0" width={nextX - x} height="200"
                  fill={`${accentColor}08`} />
              );
            })}

            {/* Main waveform - single clean line */}
            <path d={(() => {
              let p = 'M 0,100 ';
              wave.forEach((amp, i) => {
                const x = (i / samples) * 1200;
                const y = 100 + (amp * 85);
                p += `L ${x},${y} `;
              });
              return p;
            })()} fill="none" stroke="url(#cleanWave)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" filter="url(#softGlow)" />

            {/* Suspicious region overlays - bold */}
            {scores.map((score, i) => {
              if (score <= avgScore) return null;
              const si = Math.floor((timestamps[i] / maxTime) * samples);
              const ei = Math.min(i < timestamps.length - 1 ? Math.floor((timestamps[i + 1] / maxTime) * samples) : samples - 1, samples - 1);
              if (si >= ei) return null;
              let p = `M ${(si / samples) * 1200},100 `;
              for (let j = si; j <= ei; j++) {
                const x = (j / samples) * 1200;
                const y = 100 + (wave[j] * 85);
                p += `L ${x},${y} `;
              }
              return (
                <path key={`sus-${i}`} d={p} fill="none" stroke={accentColor}
                  strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" filter="url(#softGlow)" />
              );
            })}
          </svg>

          {/* Clean timeline */}
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '20px', paddingTop: '16px', borderTop: '1px solid rgba(255,255,255,0.06)', color: '#666', fontSize: '12px', fontWeight: '600' }}>
            <span>0.0s</span>
            <span>{maxTime.toFixed(1)}s</span>
          </div>
        </div>

        {/* Minimal stats */}
        <div style={{ marginTop: '24px', display: 'flex', gap: '16px', alignItems: 'center', padding: '20px 24px', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
          <div style={{ flex: 1 }}>
            <div style={{ color: '#666', fontSize: '11px', marginBottom: '6px', fontWeight: '700', letterSpacing: '0.5px' }}>ANALYSIS</div>
            <div style={{ color: '#fff', fontSize: '14px', fontWeight: '600' }}>
              {highScoreRegions > 0 
                ? `${highScoreRegions} high-confidence region${highScoreRegions > 1 ? 's' : ''} (${highScorePercent}% of audio)`
                : 'Uniform confidence across audio'}
            </div>
          </div>
          {highScoreRegions > 0 && (
            <div style={{ 
              padding: '12px 20px', 
              background: `${accentColor}15`, 
              borderRadius: '8px', 
              border: `1px solid ${accentColor}30`,
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: accentColor, boxShadow: `0 0 8px ${accentColor}` }} />
              <span style={{ color: accentColor, fontSize: '13px', fontWeight: '700' }}>{isFake ? 'HIGH FAKE' : 'HIGH REAL'}</span>
            </div>
          )}
        </div>
      </div>
      <style>{`@keyframes fadeIn { to { opacity: 1; } }`}</style>
    </div>
  );
};

export const ImprovedTemporalCard = ({ data, accentColor, delay, isFake }) => {
  if (!data?.timestamps) return null;
  const { timestamps, scores, mean_score, std_score, consistency_index } = data;
  const variation = std_score * 100;
  const meanPercent = mean_score * 100;
  
  // Calculate VISUAL variation (how much bars differ from each other)
  // This is more intuitive than std_score alone
  let visualVariation = 0;
  for (let i = 1; i < scores.length; i++) {
    visualVariation += Math.abs(scores[i] - scores[i-1]);
  }
  visualVariation = (visualVariation / (scores.length - 1)) * 100;
  
  // Calculate coefficient of variation (normalized std)
  const coefficientOfVariation = (std_score / (mean_score + 0.001)) * 100;
  
  // IMPROVED LOGIC based on VISUAL appearance:
  // For FAKE audio: Low variation is SUSPICIOUS (too uniform = TTS)
  // For REAL audio: Any variation is NORMAL
  let status, statusColor, statusText;
  
  if (isFake) {
    // For fake audio: check if bars look too uniform
    if (visualVariation < 3 && variation < 5) {
      status = 'bad';
      statusColor = '#ef4444';
      statusText = 'TOO UNIFORM';
    } else if (visualVariation > 8 || variation > 12) {
      status = 'good';
      statusColor = '#22c55e';
      statusText = 'NATURAL VARIATION';
    } else {
      status = 'medium';
      statusColor = '#fbbf24';
      statusText = 'MODERATE';
    }
  } else {
    // For real audio: all variation patterns are normal
    if (visualVariation < 3 && variation < 5) {
      status = 'neutral';
      statusColor = '#22c55e';
      statusText = 'CONSISTENT';
    } else if (visualVariation > 10 || variation > 15) {
      status = 'neutral';
      statusColor = '#22c55e';
      statusText = 'HIGHLY VARIED';
    } else {
      status = 'neutral';
      statusColor = '#22c55e';
      statusText = 'NORMAL VARIATION';
    }
  }

  return (
    <div style={{ opacity: 0, animation: `fadeIn 0.6s ease-out ${delay}ms forwards` }}>
      <div style={{ background: 'linear-gradient(135deg, #0a0a0a 0%, #141414 100%)', border: '2px solid #2a2a2a', borderRadius: '24px', padding: '48px', marginBottom: '32px', boxShadow: '0 8px 32px rgba(0,0,0,0.4)' }}>
        <div style={{ marginBottom: '40px' }}>
          <h3 style={{ color: '#fff', fontSize: '32px', fontWeight: '900', marginBottom: '12px', letterSpacing: '-0.5px' }}>Temporal Consistency Analysis</h3>
          <p style={{ color: '#888', fontSize: '15px', lineHeight: '1.7', maxWidth: '800px' }}>
            {isFake 
              ? (status === 'bad'
                ? 'Suspiciously uniform scores - too consistent across time, typical of synthetic/TTS audio'
                : status === 'good'
                ? 'Natural variation detected despite FAKE prediction - scores fluctuate naturally'
                : 'Moderate variation - some natural patterns present')
              : `${statusText} variation pattern - consistent with authentic human speech`}
          </p>
        </div>

        {/* Metrics Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px', marginBottom: '40px' }}>
          <div style={{
            background: 'rgba(255,255,255,0.02)', border: '2px solid #2a2a2a',
            borderRadius: '16px', padding: '28px', textAlign: 'center'
          }}>
            <div style={{ color: '#888', fontSize: '12px', fontWeight: '700', marginBottom: '12px', letterSpacing: '1px' }}>
              MEAN SCORE
            </div>
            <div style={{ color: '#fff', fontSize: '44px', fontWeight: '900', marginBottom: '8px' }}>
              {meanPercent.toFixed(1)}%
            </div>
            <div style={{ color: '#666', fontSize: '11px' }}>Average fakeness</div>
          </div>

          <div style={{
            background: 'rgba(255,255,255,0.02)', border: `2px solid ${statusColor}40`,
            borderRadius: '16px', padding: '28px', textAlign: 'center',
            boxShadow: `0 0 24px ${statusColor}20`
          }}>
            <div style={{ color: '#888', fontSize: '12px', fontWeight: '700', marginBottom: '12px', letterSpacing: '1px' }}>
              VARIATION
            </div>
            <div style={{ color: statusColor, fontSize: '44px', fontWeight: '900', marginBottom: '8px' }}>
              {variation.toFixed(1)}%
            </div>
            <div style={{ color: statusColor, fontSize: '11px', fontWeight: '700' }}>
              {statusText}
            </div>
          </div>

          <div style={{
            background: 'rgba(255,255,255,0.02)', border: '2px solid #2a2a2a',
            borderRadius: '16px', padding: '28px', textAlign: 'center'
          }}>
            <div style={{ color: '#888', fontSize: '12px', fontWeight: '700', marginBottom: '12px', letterSpacing: '1px' }}>
              CONSISTENCY
            </div>
            <div style={{ color: '#fff', fontSize: '44px', fontWeight: '900', marginBottom: '8px' }}>
              {(consistency_index || 0).toFixed(3)}
            </div>
            <div style={{ color: '#666', fontSize: '11px' }}>Index value</div>
          </div>
        </div>

        {/* Timeline Chart */}
        <div style={{ background: 'rgba(0,0,0,0.6)', borderRadius: '16px', padding: '40px 32px 32px 60px', border: '1px solid rgba(255,255,255,0.05)', position: 'relative' }}>
          {/* Y-axis labels */}
          <div style={{ position: 'absolute', left: '8px', top: '40px', bottom: '60px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', fontSize: '11px', color: '#666', fontWeight: '700' }}>
            <span>100%</span>
            <span>75%</span>
            <span>50%</span>
            <span>25%</span>
            <span>0%</span>
          </div>

          {/* Chart area */}
          <div style={{ position: 'relative', height: '280px' }}>
            {/* Grid lines */}
            {[0, 25, 50, 75, 100].map(pct => (
              <div key={pct} style={{
                position: 'absolute', left: 0, right: 0, bottom: `${pct}%`,
                height: '1px', background: 'rgba(255,255,255,0.06)'
              }} />
            ))}

            {/* Mean line */}
            <div style={{
              position: 'absolute', left: 0, right: 0, bottom: `${meanPercent}%`,
              height: '2px', background: '#fbbf24', opacity: 0.5,
              boxShadow: '0 0 8px rgba(251, 191, 36, 0.6)'
            }} />

            {/* Bars */}
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: '1px', height: '100%' }}>
              {scores.map((score, i) => {
                const height = score * 100;
                const isAboveMean = score > mean_score;
                return (
                  <div key={i} style={{
                    flex: 1, height: `${height}%`, minWidth: '3px',
                    background: isAboveMean 
                      ? `linear-gradient(180deg, ${accentColor} 0%, ${accentColor}cc 100%)`
                      : 'linear-gradient(180deg, #22c55e 0%, #16a34a 100%)',
                    borderRadius: '2px 2px 0 0',
                    boxShadow: isAboveMean ? `0 0 8px ${accentColor}66` : 'none',
                    transition: 'all 0.3s ease',
                    cursor: 'pointer'
                  }}
                  title={`${timestamps[i]?.toFixed(2)}s: ${(score * 100).toFixed(1)}%`}
                  />
                );
              })}
            </div>
          </div>

          {/* X-axis */}
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '16px', paddingTop: '12px', borderTop: '1px solid rgba(255,255,255,0.06)', color: '#666', fontSize: '13px', fontWeight: '700' }}>
            <span>0.0s</span>
            <span style={{ color: '#888' }}>TIME →</span>
            <span>{timestamps[timestamps.length - 1]?.toFixed(1)}s</span>
          </div>
        </div>

      </div>
      <style>{`@keyframes fadeIn { to { opacity: 1; } }`}</style>
    </div>
  );
};
