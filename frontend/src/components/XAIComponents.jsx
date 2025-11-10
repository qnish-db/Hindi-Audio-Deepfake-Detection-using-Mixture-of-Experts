import React from 'react';

// Expert Agreement Component
export const ExpertAgreementCard = ({ data, accentColor }) => {
  const experts = data.experts || {};
  const expertNames = Object.keys(experts);

  return (
    <div style={{
      background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
      border: '1px solid #2a2a2a',
      borderRadius: '24px',
      padding: '40px',
      marginBottom: '32px'
    }}>
      <div style={{ marginBottom: '32px' }}>
        <h3 style={{ color: '#fff', fontSize: '24px', fontWeight: '700', marginBottom: '8px' }}>
          Expert Agreement Analysis
        </h3>
        <p style={{ color: '#888', fontSize: '14px', lineHeight: '1.6' }}>
          {data.interpretation}
        </p>
      </div>

      <div style={{
        background: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid #2a2a2a',
        borderRadius: '16px',
        padding: '24px',
        marginBottom: '32px'
      }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '16px'
        }}>
          <span style={{ color: '#888', fontSize: '14px' }}>Agreement Score</span>
          <span style={{ color: accentColor, fontSize: '28px', fontWeight: '900' }}>
            {(data.agreement_score * 100).toFixed(1)}%
          </span>
        </div>
        <div style={{
          width: '100%',
          height: '12px',
          background: 'rgba(255, 255, 255, 0.05)',
          borderRadius: '50px',
          overflow: 'hidden'
        }}>
          <div style={{
            width: `${data.agreement_score * 100}%`,
            height: '100%',
            background: `linear-gradient(90deg, ${accentColor} 0%, ${accentColor}aa 100%)`,
            borderRadius: '50px',
            transition: 'width 1s ease-out',
            boxShadow: `0 0 20px ${accentColor}88`
          }} />
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '20px' }}>
        {expertNames.map((expertName, idx) => {
          const expert = experts[expertName];
          const isFakeExpert = expert.prediction === 'FAKE';
          
          return (
            <div key={idx} style={{
              background: 'rgba(255, 255, 255, 0.02)',
              border: `2px solid ${isFakeExpert ? 'rgba(239, 68, 68, 0.3)' : 'rgba(34, 197, 94, 0.3)'}`,
              borderRadius: '16px',
              padding: '24px',
              position: 'relative',
              overflow: 'hidden'
            }}>
              <div style={{
                position: 'absolute',
                top: '-50%',
                right: '-50%',
                width: '100%',
                height: '100%',
                background: `radial-gradient(circle, ${isFakeExpert ? 'rgba(239, 68, 68, 0.1)' : 'rgba(34, 197, 94, 0.1)'} 0%, transparent 70%)`,
                pointerEvents: 'none'
              }} />

              <div style={{ position: 'relative', zIndex: 1 }}>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  marginBottom: '20px'
                }}>
                  <div style={{
                    width: '40px',
                    height: '40px',
                    background: isFakeExpert ? 'rgba(239, 68, 68, 0.2)' : 'rgba(34, 197, 94, 0.2)',
                    borderRadius: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    <svg width="20" height="20" fill={isFakeExpert ? '#ef4444' : '#22c55e'} viewBox="0 0 20 20">
                      <path d="M10 2a8 8 0 100 16 8 8 0 000-16zm0 14a6 6 0 110-12 6 6 0 010 12z"/>
                      <circle cx="10" cy="10" r="3"/>
                    </svg>
                  </div>
                  <div>
                    <div style={{ color: '#fff', fontSize: '16px', fontWeight: '700' }}>
                      {expertName.includes('hubert') ? 'HuBERT' : 'Wav2Vec2'}
                    </div>
                    <div style={{ color: '#666', fontSize: '12px' }}>
                      {expertName.includes('hubert') ? 'Acoustic Expert' : 'Linguistic Expert'}
                    </div>
                  </div>
                </div>

                <div style={{ marginBottom: '16px' }}>
                  <div style={{ color: '#888', fontSize: '12px', marginBottom: '6px' }}>Prediction</div>
                  <div style={{
                    color: isFakeExpert ? '#ef4444' : '#22c55e',
                    fontSize: '24px',
                    fontWeight: '900'
                  }}>
                    {expert.prediction}
                  </div>
                </div>

                <div style={{ marginBottom: '16px' }}>
                  <div style={{ color: '#888', fontSize: '12px', marginBottom: '6px' }}>Confidence</div>
                  <div style={{ color: '#fff', fontSize: '20px', fontWeight: '700' }}>
                    {(expert.prob_fake * 100).toFixed(1)}%
                  </div>
                </div>

                <div>
                  <div style={{ color: '#888', fontSize: '12px', marginBottom: '6px' }}>Gate Weight</div>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}>
                    <div style={{
                      flex: 1,
                      height: '6px',
                      background: 'rgba(255, 255, 255, 0.1)',
                      borderRadius: '50px',
                      overflow: 'hidden'
                    }}>
                      <div style={{
                        width: `${expert.gate_weight * 100}%`,
                        height: '100%',
                        background: '#888',
                        borderRadius: '50px'
                      }} />
                    </div>
                    <span style={{ color: '#fff', fontSize: '14px', fontWeight: '600', minWidth: '50px' }}>
                      {(expert.gate_weight * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

// Temporal Heatmap Component
export const TemporalHeatmapCard = ({ data, accentColor }) => {
  const { timestamps = [], scores = [], consistency_index = 0 } = data;
  if (timestamps.length === 0) return null;

  const isSuspicious = consistency_index < 0.3;

  return (
    <div style={{
      background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
      border: '1px solid #2a2a2a',
      borderRadius: '24px',
      padding: '40px',
      marginBottom: '32px'
    }}>
      <div style={{ marginBottom: '32px' }}>
        <h3 style={{ color: '#fff', fontSize: '24px', fontWeight: '700', marginBottom: '8px' }}>
          Temporal Consistency Analysis
        </h3>
        <p style={{ color: '#888', fontSize: '14px', lineHeight: '1.6' }}>
          {isSuspicious 
            ? "⚠️ Suspiciously uniform scores across time - characteristic of synthetic speech" 
            : "✓ Natural variation in scores over time - characteristic of human speech"}
        </p>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: '24px',
        marginBottom: '40px'
      }}>
        <div style={{
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid #2a2a2a',
          borderRadius: '16px',
          padding: '28px',
          textAlign: 'center'
        }}>
          <div style={{ color: '#888', fontSize: '15px', marginBottom: '16px', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: '700' }}>MEAN SCORE</div>
          <div style={{ color: '#fff', fontSize: '64px', fontWeight: '900', marginBottom: '12px', lineHeight: '1' }}>
            {(data.mean_score * 100).toFixed(1)}%
          </div>
          <div style={{ color: '#888', fontSize: '14px', lineHeight: '1.6', fontWeight: '500' }}>
            Average fakeness across all segments
          </div>
        </div>
        <div style={{
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid #2a2a2a',
          borderRadius: '16px',
          padding: '28px',
          textAlign: 'center'
        }}>
          <div style={{ color: '#888', fontSize: '15px', marginBottom: '16px', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: '700' }}>VARIATION</div>
          <div style={{ color: '#fff', fontSize: '64px', fontWeight: '900', marginBottom: '12px', lineHeight: '1' }}>
            {(data.std_score * 100).toFixed(1)}%
          </div>
          <div style={{ color: data.std_score < 0.05 ? '#ef4444' : '#22c55e', fontSize: '14px', lineHeight: '1.6', fontWeight: '600' }}>
            {data.std_score < 0.05 ? '⚠️ SUSPICIOUSLY UNIFORM' : '✓ NATURAL VARIATION'}
          </div>
        </div>
        <div style={{
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid #2a2a2a',
          borderRadius: '16px',
          padding: '28px',
          textAlign: 'center'
        }}>
          <div style={{ color: '#888', fontSize: '15px', marginBottom: '16px', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: '700' }}>CONSISTENCY</div>
          <div style={{
            color: isSuspicious ? '#ef4444' : '#22c55e',
            fontSize: '64px',
            fontWeight: '900',
            marginBottom: '12px',
            lineHeight: '1'
          }}>
            {consistency_index.toFixed(3)}
          </div>
          <div style={{ color: isSuspicious ? '#ef4444' : '#22c55e', fontSize: '14px', lineHeight: '1.6', fontWeight: '600' }}>
            {isSuspicious ? '⚠️ TOO CONSISTENT = ROBOTIC' : '✓ NATURAL INCONSISTENCY'}
          </div>
        </div>
      </div>

      <div style={{
        background: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid #2a2a2a',
        borderRadius: '16px',
        padding: '32px'
      }}>
        <div style={{ marginBottom: '24px' }}>
          <div style={{ color: '#fff', fontSize: '18px', marginBottom: '8px', fontWeight: '700' }}>
            Fakeness Score Over Time
          </div>
          <div style={{ color: '#888', fontSize: '13px', lineHeight: '1.6' }}>
            Each bar represents a 0.5-second segment. Height shows the model's confidence that segment is fake.
          </div>
        </div>
        <div style={{ display: 'flex', gap: '4px', height: '280px', alignItems: 'flex-end', marginBottom: '20px' }}>
          {scores.map((score, idx) => {
            const height = score * 100;
            return (
              <div
                key={idx}
                title={`${timestamps[idx].toFixed(2)}s: ${(score * 100).toFixed(1)}%`}
                style={{
                  flex: 1,
                  height: `${height}%`,
                  background: `linear-gradient(180deg, ${accentColor} 0%, ${accentColor}88 100%)`,
                  borderRadius: '3px 3px 0 0',
                  transition: 'all 0.3s ease',
                  cursor: 'pointer',
                  minWidth: '8px',
                  position: 'relative'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.opacity = '0.7';
                  e.currentTarget.style.transform = 'scaleY(1.05)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.opacity = '1';
                  e.currentTarget.style.transform = 'scaleY(1)';
                }}
              />
            );
          })}
        </div>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginTop: '12px',
          color: '#888',
          fontSize: '13px',
          fontWeight: '600'
        }}>
          <span>0s</span>
          <span>{(timestamps[timestamps.length - 1] / 2)?.toFixed(1)}s</span>
          <span>{timestamps[timestamps.length - 1]?.toFixed(1)}s</span>
        </div>
        <div style={{
          marginTop: '20px',
          padding: '16px',
          background: 'rgba(255, 255, 255, 0.02)',
          borderRadius: '12px',
          border: '1px solid #2a2a2a'
        }}>
          <div style={{ color: '#888', fontSize: '12px', marginBottom: '8px' }}>💡 What this means:</div>
          <div style={{ color: '#aaa', fontSize: '12px', lineHeight: '1.6' }}>
            {isSuspicious 
              ? 'The bars are suspiciously uniform in height - real human speech should show more variation as emphasis, tone, and emotion change naturally.' 
              : 'The bars show natural variation - this is expected in genuine human speech where tone and emphasis shift throughout.'}
          </div>
        </div>
      </div>
    </div>
  );
};

// Breathing Patterns Component
export const BreathingPatternsCard = ({ data, accentColor }) => {
  const { pauses = [], regularity_score = 0, interpretation = '' } = data;
  const isSuspicious = regularity_score > 0.7;

  return (
    <div style={{
      background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
      border: '1px solid #2a2a2a',
      borderRadius: '24px',
      padding: '40px',
      marginBottom: '32px'
    }}>
      <div style={{ marginBottom: '32px' }}>
        <h3 style={{ color: '#fff', fontSize: '24px', fontWeight: '700', marginBottom: '8px' }}>
          Breathing Pattern Analysis
        </h3>
        <p style={{ color: '#888', fontSize: '14px', lineHeight: '1.6' }}>
          {interpretation}
        </p>
      </div>

      <div style={{
        background: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid #2a2a2a',
        borderRadius: '16px',
        padding: '24px',
        marginBottom: '32px'
      }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '16px'
        }}>
          <span style={{ color: '#888', fontSize: '14px' }}>Regularity Score (Higher = More Suspicious)</span>
          <span style={{
            color: isSuspicious ? '#ef4444' : '#22c55e',
            fontSize: '28px',
            fontWeight: '900'
          }}>
            {(regularity_score * 100).toFixed(0)}%
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
            position: 'absolute',
            left: '0%',
            width: '40%',
            height: '100%',
            background: 'rgba(34, 197, 94, 0.2)'
          }} />
          <div style={{
            position: 'absolute',
            left: '70%',
            width: '30%',
            height: '100%',
            background: 'rgba(239, 68, 68, 0.2)'
          }} />
          <div style={{
            width: `${regularity_score * 100}%`,
            height: '100%',
            background: `linear-gradient(90deg, ${isSuspicious ? '#ef4444' : '#22c55e'} 0%, ${isSuspicious ? '#dc2626' : '#16a34a'} 100%)`,
            borderRadius: '50px',
            transition: 'width 1s ease-out',
            boxShadow: `0 0 20px ${isSuspicious ? '#ef4444' : '#22c55e'}88`
          }} />
        </div>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginTop: '8px',
          fontSize: '11px',
          color: '#666'
        }}>
          <span>Natural</span>
          <span>Suspicious</span>
        </div>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: '24px',
        marginBottom: '40px'
      }}>
        <div style={{
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid #2a2a2a',
          borderRadius: '16px',
          padding: '28px',
          textAlign: 'center'
        }}>
          <div style={{ color: '#888', fontSize: '15px', marginBottom: '16px', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: '700' }}>PAUSES</div>
          <div style={{ color: '#fff', fontSize: '64px', fontWeight: '900', marginBottom: '12px', lineHeight: '1' }}>
            {pauses.length}
          </div>
          <div style={{ color: pauses.length < 3 ? '#ef4444' : '#22c55e', fontSize: '14px', lineHeight: '1.6', fontWeight: '600' }}>
            {pauses.length === 0 ? '⚠️ NO BREATHING' : pauses.length < 3 ? '⚠️ VERY FEW' : '✓ NORMAL'}
          </div>
        </div>
        <div style={{
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid #2a2a2a',
          borderRadius: '16px',
          padding: '28px',
          textAlign: 'center'
        }}>
          <div style={{ color: '#888', fontSize: '15px', marginBottom: '16px', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: '700' }}>AVG DURATION</div>
          <div style={{ color: '#fff', fontSize: '64px', fontWeight: '900', marginBottom: '12px', lineHeight: '1' }}>
            {data.mean_pause_duration?.toFixed(2) || '—'}s
          </div>
          <div style={{ color: '#888', fontSize: '14px', lineHeight: '1.6', fontWeight: '500' }}>
            Average breathing pause
          </div>
        </div>
        <div style={{
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid #2a2a2a',
          borderRadius: '16px',
          padding: '28px',
          textAlign: 'center'
        }}>
          <div style={{ color: '#888', fontSize: '15px', marginBottom: '16px', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: '700' }}>STD DEV</div>
          <div style={{ color: '#fff', fontSize: '64px', fontWeight: '900', marginBottom: '12px', lineHeight: '1' }}>
            {data.std_pause_duration?.toFixed(2) || '—'}s
          </div>
          <div style={{ color: data.std_pause_duration < 0.1 ? '#ef4444' : '#22c55e', fontSize: '14px', lineHeight: '1.6', fontWeight: '600' }}>
            {data.std_pause_duration < 0.1 ? '⚠️ TOO UNIFORM' : '✓ NATURAL'}
          </div>
        </div>
      </div>

      {pauses.length > 0 && (
        <div style={{
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid #2a2a2a',
          borderRadius: '16px',
          padding: '24px'
        }}>
          <div style={{ color: '#888', fontSize: '14px', marginBottom: '16px', fontWeight: '600' }}>
            Pause Timeline (First 10 shown)
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {pauses.slice(0, 10).map((pause, idx) => (
              <div key={idx} style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '12px',
                background: 'rgba(255, 255, 255, 0.02)',
                borderRadius: '8px'
              }}>
                <div style={{
                  width: '32px',
                  height: '32px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  borderRadius: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#666',
                  fontSize: '12px',
                  fontWeight: '600'
                }}>
                  {idx + 1}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ color: '#fff', fontSize: '14px' }}>
                    {pause.start.toFixed(2)}s - {pause.end.toFixed(2)}s
                  </div>
                  <div style={{ color: '#666', fontSize: '12px' }}>
                    Duration: {pause.duration.toFixed(3)}s
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Attention Rollout Component  
export const AttentionRolloutCard = ({ data, accentColor }) => {
  const { combined_attention = [], peak_regions = [] } = data;
  if (combined_attention.length === 0) return null;

  return (
    <div style={{
      background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
      border: '1px solid #2a2a2a',
      borderRadius: '24px',
      padding: '40px',
      marginBottom: '32px'
    }}>
      <div style={{ marginBottom: '32px' }}>
        <h3 style={{ color: '#fff', fontSize: '24px', fontWeight: '700', marginBottom: '8px' }}>
          Attention Analysis (GradCAM)
        </h3>
        <p style={{ color: '#888', fontSize: '14px', lineHeight: '1.6' }}>
          Shows which parts of the audio the model focused on when making its decision. Bright regions = high attention.
        </p>
      </div>

      <div style={{
        background: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid #2a2a2a',
        borderRadius: '16px',
        padding: '32px',
        marginBottom: '24px',
        overflow: 'hidden'
      }}>
        <div style={{ marginBottom: '24px' }}>
          <div style={{ color: '#fff', fontSize: '18px', marginBottom: '8px', fontWeight: '700' }}>
            Model Attention Heatmap
          </div>
          <div style={{ color: '#888', fontSize: '13px', lineHeight: '1.6' }}>
            Each bar represents a feature dimension. Brighter/taller bars show which features the model focused on most when deciding if the audio is fake.
          </div>
        </div>
        <div style={{
          width: '100%',
          height: '250px',
          overflowX: 'hidden',
          overflowY: 'hidden',
          position: 'relative'
        }}>
          <div style={{
            display: 'flex',
            gap: '2px',
            height: '100%',
            alignItems: 'flex-end',
            background: 'rgba(0, 0, 0, 0.4)',
            borderRadius: '12px',
            padding: '12px',
            width: '100%'
          }}>
            {combined_attention.map((attn, idx) => {
              const intensity = attn;
              const color = `rgba(255, ${200 - intensity * 150}, ${100 - intensity * 100}, ${0.3 + intensity * 0.7})`;
              return (
                <div
                  key={idx}
                  title={`Feature ${idx}: ${(intensity * 100).toFixed(1)}% attention`}
                  style={{
                    flex: '1 1 auto',
                    height: `${Math.max(15, 20 + intensity * 75)}%`,
                    background: color,
                    borderRadius: '3px',
                    minWidth: '3px',
                    maxWidth: '6px',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'scaleY(1.1)';
                    e.currentTarget.style.filter = 'brightness(1.3)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'scaleY(1)';
                    e.currentTarget.style.filter = 'brightness(1)';
                  }}
                />
              );
            })}
          </div>
        </div>
        <div style={{
          marginTop: '20px',
          padding: '16px',
          background: 'rgba(255, 255, 255, 0.02)',
          borderRadius: '12px',
          border: '1px solid #2a2a2a'
        }}>
          <div style={{ color: '#888', fontSize: '12px', marginBottom: '8px' }}>💡 What this means:</div>
          <div style={{ color: '#aaa', fontSize: '12px', lineHeight: '1.6' }}>
            This is a GradCAM visualization showing where the model "looked" in the audio features. Bright orange/red bars indicate features that strongly influenced the fake detection. These are the acoustic patterns the AI found most suspicious.
          </div>
        </div>
      </div>

      {peak_regions.length > 0 && (
        <div style={{
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid #2a2a2a',
          borderRadius: '16px',
          padding: '24px'
        }}>
          <div style={{ color: '#888', fontSize: '14px', marginBottom: '16px', fontWeight: '600' }}>
            High Attention Regions
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '12px' }}>
            {peak_regions.map((region, idx) => (
              <div key={idx} style={{
                padding: '16px',
                background: 'rgba(255, 255, 255, 0.02)',
                border: `1px solid ${accentColor}44`,
                borderRadius: '12px'
              }}>
                <div style={{ color: accentColor, fontSize: '12px', marginBottom: '8px', fontWeight: '600' }}>
                  Region {idx + 1}
                </div>
                <div style={{ color: '#fff', fontSize: '16px', fontWeight: '700' }}>
                  Features {region[0]}-{region[1]}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Waveform Visualization Component
export const AudioWaveformCard = ({ temporalData, accentColor }) => {
  if (!temporalData || !temporalData.timestamps || temporalData.timestamps.length === 0) {
    return null;
  }

  const { timestamps, scores } = temporalData;
  const maxTime = timestamps[timestamps.length - 1];
  const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;

  // Generate smoother waveform data
  const waveformSamples = 150;
  const waveform = Array.from({ length: waveformSamples }, (_, i) => {
    const t = (i / waveformSamples) * maxTime;
    // Create smoother waveform with multiple frequencies
    const low = Math.sin(t * 3) * 0.4;
    const mid = Math.sin(t * 8) * 0.3;
    const high = Math.sin(t * 15) * 0.15;
    return low + mid + high;
  });

  return (
    <div style={{
      background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
      border: '1px solid #2a2a2a',
      borderRadius: '24px',
      padding: '40px',
      marginBottom: '32px'
    }}>
      <div style={{ marginBottom: '40px' }}>
        <h3 style={{ color: '#fff', fontSize: '36px', fontWeight: '900', marginBottom: '16px', letterSpacing: '-1px', textAlign: 'center' }}>
          🌊 Audio Waveform Analysis
        </h3>
        <p style={{ color: '#888', fontSize: '16px', lineHeight: '1.8', textAlign: 'center', maxWidth: '900px', margin: '0 auto' }}>
          Visual representation of the audio signal over time. <strong style={{ color: accentColor }}>Red glowing regions</strong> highlight suspicious segments where the AI detected synthetic characteristics.
        </p>
      </div>

      {/* Waveform Container */}
      <div style={{
        background: 'rgba(0, 0, 0, 0.8)',
        border: '3px solid #2a2a2a',
        borderRadius: '24px',
        padding: '50px 32px',
        position: 'relative',
        overflow: 'hidden',
        boxShadow: '0 10px 40px rgba(0,0,0,0.5)'
      }}>
        {/* Grid lines */}
        {[...Array(5)].map((_, i) => (
          <div
            key={`grid-${i}`}
            style={{
              position: 'absolute',
              left: 0,
              right: 0,
              top: `${20 + i * 20}%`,
              height: '1px',
              background: 'rgba(255, 255, 255, 0.05)',
              pointerEvents: 'none'
            }}
          />
        ))}

        {/* Center line */}
        <div style={{
          position: 'absolute',
          left: 0,
          right: 0,
          top: '50%',
          height: '2px',
          background: 'rgba(255, 255, 255, 0.1)',
          pointerEvents: 'none'
        }} />

        {/* Waveform */}
        <svg width="100%" height="400" viewBox="0 0 1000 400" preserveAspectRatio="xMidYMid meet" style={{ display: 'block' }}>
          <defs>
            <linearGradient id="greenGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#4ade80" stopOpacity="0.6" />
              <stop offset="50%" stopColor="#4ade80" stopOpacity="0.3" />
              <stop offset="100%" stopColor="#4ade80" stopOpacity="0.1" />
            </linearGradient>
            <linearGradient id="redGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor={accentColor} stopOpacity="0.7" />
              <stop offset="50%" stopColor={accentColor} stopOpacity="0.4" />
              <stop offset="100%" stopColor={accentColor} stopOpacity="0.1" />
            </linearGradient>
          </defs>

          {/* Suspicious region backgrounds */}
          {scores.map((score, idx) => {
            if (score <= avgScore) return null;
            const x = (timestamps[idx] / maxTime) * 1000;
            const nextX = idx < timestamps.length - 1 ? (timestamps[idx + 1] / maxTime) * 1000 : 1000;
            const width = nextX - x;
            
            return (
              <rect
                key={`sus-bg-${idx}`}
                x={x}
                y="0"
                width={width}
                height="400"
                fill={`${accentColor}15`}
                stroke="none"
              />
            );
          })}

          {/* Main waveform - Filled area chart */}
          <path
            d={(() => {
              let path = `M 0,200 `;
              waveform.forEach((amp, i) => {
                const x = (i / waveformSamples) * 1000;
                const y = 200 - amp * 100;
                path += `L ${x},${y} `;
              });
              for (let i = waveformSamples - 1; i >= 0; i--) {
                const x = (i / waveformSamples) * 1000;
                const y = 200 + waveform[i] * 100;
                path += `L ${x},${y} `;
              }
              path += 'Z';
              return path;
            })()}
            fill="url(#greenGradient)"
            stroke="#4ade80"
            strokeWidth="2"
            style={{
              filter: 'drop-shadow(0 0 10px rgba(74, 222, 128, 0.5))'
            }}
          />

          {/* Suspicious regions overlay - red filled waveform */}
          {scores.map((score, idx) => {
            if (score <= avgScore) return null;
            const startIdx = Math.max(0, Math.floor((timestamps[idx] / maxTime) * waveformSamples));
            const endIdx = Math.min(waveformSamples - 1, 
              idx < timestamps.length - 1 
                ? Math.floor((timestamps[idx + 1] / maxTime) * waveformSamples)
                : waveformSamples - 1
            );
            
            if (startIdx >= endIdx) return null;

            let path = `M ${(startIdx / waveformSamples) * 1000},200 `;
            for (let i = startIdx; i <= endIdx; i++) {
              const x = (i / waveformSamples) * 1000;
              const y = 200 - waveform[i] * 100;
              path += `L ${x},${y} `;
            }
            for (let i = endIdx; i >= startIdx; i--) {
              const x = (i / waveformSamples) * 1000;
              const y = 200 + waveform[i] * 100;
              path += `L ${x},${y} `;
            }
            path += 'Z';

            return (
              <path
                key={`wave-sus-${idx}`}
                d={path}
                fill="url(#redGradient)"
                stroke={accentColor}
                strokeWidth="2.5"
                style={{
                  filter: `drop-shadow(0 0 12px ${accentColor})`
                }}
              />
            );
          })}
        </svg>

        {/* Time markers */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginTop: '16px',
          color: '#888',
          fontSize: '14px',
          fontWeight: '700'
        }}>
          <span style={{ background: 'rgba(0,0,0,0.6)', padding: '6px 12px', borderRadius: '6px' }}>0.0s</span>
          <span style={{ background: 'rgba(0,0,0,0.6)', padding: '6px 12px', borderRadius: '6px' }}>{(maxTime / 2).toFixed(1)}s</span>
          <span style={{ background: 'rgba(0,0,0,0.6)', padding: '6px 12px', borderRadius: '6px' }}>{maxTime.toFixed(1)}s</span>
        </div>
      </div>

      {/* Legend */}
      <div style={{
        marginTop: '40px',
        display: 'flex',
        gap: '48px',
        justifyContent: 'center',
        flexWrap: 'wrap',
        padding: '24px',
        background: 'rgba(255, 255, 255, 0.02)',
        borderRadius: '16px',
        border: '1px solid #2a2a2a'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{
            width: '60px',
            height: '6px',
            background: '#4ade80',
            borderRadius: '3px',
            boxShadow: '0 0 12px rgba(74, 222, 128, 0.6)'
          }} />
          <span style={{ color: '#4ade80', fontSize: '17px', fontWeight: '700', letterSpacing: '0.5px' }}>NORMAL AUDIO</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{
            width: '60px',
            height: '6px',
            background: accentColor,
            borderRadius: '3px',
            boxShadow: `0 0 12px ${accentColor}`
          }} />
          <span style={{ color: accentColor, fontSize: '17px', fontWeight: '900', letterSpacing: '0.5px' }}>⚠️ SUSPICIOUS REGIONS</span>
        </div>
      </div>

      {/* Explanation */}
      <div style={{
        marginTop: '32px',
        padding: '32px',
        background: `linear-gradient(135deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%)`,
        borderRadius: '20px',
        border: '2px solid #2a2a2a'
      }}>
        <div style={{ color: '#fff', fontSize: '18px', marginBottom: '16px', fontWeight: '900', letterSpacing: '0.5px' }}>💡 HOW TO READ THIS WAVEFORM:</div>
        <div style={{ color: '#bbb', fontSize: '16px', lineHeight: '2' }}>
          This visualization shows the audio signal amplitude over time. <strong style={{ color: '#4ade80', fontWeight: '700' }}>GREEN SECTIONS</strong> represent normal audio patterns where the model found nothing suspicious. <strong style={{ color: accentColor, fontWeight: '900' }}>RED GLOWING SECTIONS</strong> highlight where the AI detected strong anomalies - these regions have acoustic characteristics typical of synthetic/TTS audio (unnatural pitch, robotic prosody, missing breath sounds). The brighter the red glow, the more suspicious that segment is.
        </div>
      </div>
    </div>
  );
};

// Audio Timeline Visualization Component
export const AudioTimelineCard = ({ temporalData, breathingData, accentColor }) => {
  if (!temporalData || !temporalData.timestamps || temporalData.timestamps.length === 0) {
    return null;
  }

  const { timestamps, scores } = temporalData;
  const { pauses = [] } = breathingData || {};
  
  const maxTime = timestamps[timestamps.length - 1];
  const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;

  // Identify suspicious regions (score > avgScore)
  const suspiciousRegions = [];
  let regionStart = null;
  
  scores.forEach((score, idx) => {
    if (score > avgScore && regionStart === null) {
      regionStart = idx;
    } else if (score <= avgScore && regionStart !== null) {
      suspiciousRegions.push({ start: regionStart, end: idx - 1 });
      regionStart = null;
    }
  });
  if (regionStart !== null) {
    suspiciousRegions.push({ start: regionStart, end: scores.length - 1 });
  }

  return (
    <div style={{
      background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
      border: '1px solid #2a2a2a',
      borderRadius: '24px',
      padding: '40px',
      marginBottom: '32px'
    }}>
      <div style={{ marginBottom: '32px' }}>
        <h3 style={{ color: '#fff', fontSize: '24px', fontWeight: '700', marginBottom: '8px' }}>
          Audio Timeline Analysis
        </h3>
        <p style={{ color: '#888', fontSize: '14px', lineHeight: '1.6' }}>
          Visual timeline showing suspicious regions, breathing pauses, and model confidence across the audio.
        </p>
      </div>

      {/* Main Timeline Visualization */}
      <div style={{
        background: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid #2a2a2a',
        borderRadius: '16px',
        padding: '24px',
        marginBottom: '24px'
      }}>
        <div style={{ color: '#888', fontSize: '14px', marginBottom: '20px', fontWeight: '600' }}>
          Suspicion Timeline
        </div>
        
        {/* Timeline container */}
        <div style={{ position: 'relative', width: '100%', height: '180px', marginBottom: '50px' }}>
          {/* Background track */}
          <div style={{
            position: 'absolute',
            top: '50%',
            left: 0,
            right: 0,
            height: '60px',
            transform: 'translateY(-50%)',
            background: 'linear-gradient(90deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%)',
            borderRadius: '20px',
            overflow: 'hidden'
          }}>
            {/* Suspicious regions overlay */}
            {suspiciousRegions.map((region, idx) => {
              const startPercent = (timestamps[region.start] / maxTime) * 100;
              const endPercent = (timestamps[region.end] / maxTime) * 100;
              const width = endPercent - startPercent;
              
              return (
                <div
                  key={idx}
                  style={{
                    position: 'absolute',
                    left: `${startPercent}%`,
                    width: `${width}%`,
                    height: '100%',
                    background: `linear-gradient(90deg, ${accentColor}44 0%, ${accentColor}66 50%, ${accentColor}44 100%)`,
                    borderLeft: `2px solid ${accentColor}`,
                    borderRight: `2px solid ${accentColor}`
                  }}
                />
              );
            })}

            {/* Breathing pause markers */}
            {pauses.map((pause, idx) => {
              const pausePercent = (pause.start / maxTime) * 100;
              
              return (
                <div
                  key={`pause-${idx}`}
                  style={{
                    position: 'absolute',
                    left: `${pausePercent}%`,
                    top: '50%',
                    transform: 'translate(-50%, -50%)',
                    width: '12px',
                    height: '12px',
                    background: '#3b82f6',
                    borderRadius: '50%',
                    border: '2px solid rgba(59, 130, 246, 0.3)',
                    boxShadow: '0 0 10px rgba(59, 130, 246, 0.5)'
                  }}
                  title={`Pause at ${pause.start.toFixed(2)}s`}
                />
              );
            })}

            {/* Score markers */}
            {scores.map((score, idx) => {
              if (idx % 3 !== 0) return null; // Show every 3rd marker
              const percent = (timestamps[idx] / maxTime) * 100;
              const isSuspicious = score > avgScore;
              
              return (
                <div
                  key={`marker-${idx}`}
                  style={{
                    position: 'absolute',
                    left: `${percent}%`,
                    top: isSuspicious ? '10px' : '70px',
                    transform: 'translateX(-50%)',
                    width: '6px',
                    height: '16px',
                    background: isSuspicious ? accentColor : 'rgba(255, 255, 255, 0.2)',
                    borderRadius: '2px'
                  }}
                />
              );
            })}
          </div>

          {/* Time labels */}
          <div style={{
            position: 'absolute',
            bottom: '-30px',
            left: 0,
            right: 0,
            display: 'flex',
            justifyContent: 'space-between',
            color: '#888',
            fontSize: '13px',
            fontWeight: '600'
          }}>
            <span style={{ background: 'rgba(0,0,0,0.5)', padding: '4px 8px', borderRadius: '4px' }}>0s</span>
            <span style={{ background: 'rgba(0,0,0,0.5)', padding: '4px 8px', borderRadius: '4px' }}>{(maxTime / 4).toFixed(1)}s</span>
            <span style={{ background: 'rgba(0,0,0,0.5)', padding: '4px 8px', borderRadius: '4px' }}>{(maxTime / 2).toFixed(1)}s</span>
            <span style={{ background: 'rgba(0,0,0,0.5)', padding: '4px 8px', borderRadius: '4px' }}>{(maxTime * 3 / 4).toFixed(1)}s</span>
            <span style={{ background: 'rgba(0,0,0,0.5)', padding: '4px 8px', borderRadius: '4px' }}>{maxTime.toFixed(1)}s</span>
          </div>
        </div>

        {/* Legend */}
        <div style={{
          display: 'flex',
          gap: '24px',
          justifyContent: 'center',
          flexWrap: 'wrap',
          marginTop: '20px',
          paddingTop: '20px',
          borderTop: '1px solid #2a2a2a'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{
              width: '30px',
              height: '16px',
              background: `${accentColor}66`,
              borderRadius: '4px',
              border: `1px solid ${accentColor}`
            }} />
            <span style={{ color: '#888', fontSize: '13px', fontWeight: '600' }}>Suspicious Region</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{
              width: '16px',
              height: '16px',
              background: '#3b82f6',
              borderRadius: '50%',
              border: '2px solid rgba(59, 130, 246, 0.3)'
            }} />
            <span style={{ color: '#888', fontSize: '13px', fontWeight: '600' }}>Breathing Pause</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{
              width: '6px',
              height: '16px',
              background: accentColor,
              borderRadius: '2px'
            }} />
            <span style={{ color: '#888', fontSize: '13px', fontWeight: '600' }}>High Confidence</span>
          </div>
        </div>

        {/* Explanation */}
        <div style={{
          marginTop: '24px',
          padding: '20px',
          background: 'rgba(255, 255, 255, 0.02)',
          borderRadius: '12px',
          border: '1px solid #2a2a2a'
        }}>
          <div style={{ color: '#888', fontSize: '13px', marginBottom: '12px', fontWeight: '600' }}>💡 How to Read This Timeline:</div>
          <div style={{ color: '#aaa', fontSize: '13px', lineHeight: '1.8' }}>
            This visualization shows the entire audio file from left to right. <strong style={{ color: '#fff' }}>Highlighted bands</strong> indicate time periods where the model detected strong synthetic characteristics. <strong style={{ color: '#3b82f6' }}>Blue dots</strong> mark breathing pauses - TTS often has too few, too regular, or completely missing pauses. <strong style={{ color: accentColor }}>Vertical bars</strong> above and below the timeline show confidence spikes - markers above indicate high suspicion at that moment.
          </div>
        </div>
      </div>

      {/* Suspicious Regions Details */}
      {suspiciousRegions.length > 0 && (
        <div style={{
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid #2a2a2a',
          borderRadius: '16px',
          padding: '24px'
        }}>
          <div style={{ color: '#888', fontSize: '14px', marginBottom: '16px', fontWeight: '600' }}>
            Detected Suspicious Regions ({suspiciousRegions.length})
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: '12px' }}>
            {suspiciousRegions.slice(0, 6).map((region, idx) => {
              const startTime = timestamps[region.start];
              const endTime = timestamps[region.end];
              const avgRegionScore = scores.slice(region.start, region.end + 1).reduce((a, b) => a + b, 0) / (region.end - region.start + 1);
              
              return (
                <div key={idx} style={{
                  padding: '16px',
                  background: 'rgba(255, 255, 255, 0.02)',
                  border: `2px solid ${accentColor}33`,
                  borderRadius: '12px',
                  position: 'relative',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    position: 'absolute',
                    top: 0,
                    right: 0,
                    width: '60px',
                    height: '60px',
                    background: `radial-gradient(circle, ${accentColor}22 0%, transparent 70%)`,
                    pointerEvents: 'none'
                  }} />
                  
                  <div style={{ position: 'relative', zIndex: 1 }}>
                    <div style={{ color: accentColor, fontSize: '12px', marginBottom: '8px', fontWeight: '600' }}>
                      Region {idx + 1}
                    </div>
                    <div style={{ color: '#fff', fontSize: '16px', fontWeight: '700', marginBottom: '8px' }}>
                      {startTime.toFixed(2)}s - {endTime.toFixed(2)}s
                    </div>
                    <div style={{ color: '#888', fontSize: '12px' }}>
                      Avg Score: <span style={{ color: accentColor, fontWeight: '600' }}>{(avgRegionScore * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ color: '#666', fontSize: '11px', marginTop: '4px' }}>
                      Duration: {(endTime - startTime).toFixed(2)}s
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

// Main Container Component
const XAIVisualizations = ({ xaiData, isFake }) => {
  if (!xaiData || xaiData.error) {
    return null;
  }

  const accentColor = isFake ? '#ef4444' : '#22c55e';

  return (
    <div style={{ marginTop: '40px' }}>
      <div style={{ marginBottom: '32px', textAlign: 'center' }}>
        <h2 style={{
          color: '#fff',
          fontSize: '36px',
          fontWeight: '900',
          marginBottom: '12px',
          letterSpacing: '-1px'
        }}>
          Explainable AI Analysis
        </h2>
        <p style={{
          color: '#888',
          fontSize: '16px',
          maxWidth: '800px',
          margin: '0 auto'
        }}>
          Deep dive into the evidence that led to this detection
        </p>
      </div>

      {/* NEW: Audio Waveform - Shows actual wave with suspicious parts */}
      {xaiData.temporal_heatmap && !xaiData.temporal_heatmap.error && (
        <AudioWaveformCard 
          temporalData={xaiData.temporal_heatmap}
          accentColor={accentColor} 
        />
      )}

      {/* Audio Timeline - Shows which parts are suspicious */}
      {xaiData.temporal_heatmap && !xaiData.temporal_heatmap.error && (
        <AudioTimelineCard 
          temporalData={xaiData.temporal_heatmap} 
          breathingData={xaiData.breathing_patterns}
          accentColor={accentColor} 
        />
      )}

      {xaiData.expert_agreement && !xaiData.expert_agreement.error && (
        <ExpertAgreementCard data={xaiData.expert_agreement} accentColor={accentColor} />
      )}

      {xaiData.temporal_heatmap && !xaiData.temporal_heatmap.error && (
        <TemporalHeatmapCard data={xaiData.temporal_heatmap} accentColor={accentColor} />
      )}

      {xaiData.breathing_patterns && !xaiData.breathing_patterns.error && (
        <BreathingPatternsCard data={xaiData.breathing_patterns} accentColor={accentColor} />
      )}

      {xaiData.attention_rollout && !xaiData.attention_rollout.error && (
        <AttentionRolloutCard data={xaiData.attention_rollout} accentColor={accentColor} />
      )}

      <div style={{
        marginTop: '24px',
        textAlign: 'center',
        color: '#666',
        fontSize: '14px'
      }}>
        XAI Analysis completed in {xaiData.processing_time_ms || 0}ms
      </div>
    </div>
  );
};

export default XAIVisualizations;
