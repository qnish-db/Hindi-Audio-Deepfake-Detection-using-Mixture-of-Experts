// Part 2 - Temporal, Breathing, Timeline, Expert Agreement, Main Component

// ===== 5. TEMPORAL CONSISTENCY - ENHANCED =====
export const ImprovedTemporalCard = ({ data, accentColor, delay }) => {
  if (!data || !data.timestamps) return null;
  
  const { timestamps, scores, mean_score, std_score, consistency_index } = data;
  const variation = std_score * 100;
  const meanPercent = mean_score * 100;
  
  // Determine status colors
  const variationStatus = variation < 5 ? 'bad' : variation > 10 ? 'good' : 'medium';
  const consistencyStatus = consistency_index < 0.1 ? 'bad' : consistency_index > 0.3 ? 'good' : 'medium';
  
  const statusColor = (status) => {
    if (status === 'good') return '#22c55e';
    if (status === 'bad') return '#ef4444';
    return '#fbbf24';
  };
  
  return (
    <div style={{
      opacity: 0,
      animation: 'fadeInUp 1.2s cubic-bezier(0.16, 1, 0.3, 1) forwards',
      animationDelay: `${delay}ms`
    }}>
      <div style={{
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
        border: '2px solid #2a2a2a',
        borderRadius: '32px',
        padding: '60px',
        marginBottom: '48px'
      }}>
        <h3 style={{ color: '#fff', fontSize: '38px', fontWeight: '900', marginBottom: '12px' }}>
          📊 Temporal Consistency Analysis
        </h3>
        <p style={{ color: '#888', fontSize: '18px', marginBottom: '48px' }}>
          {variationStatus === 'good' ? '✓ Natural variation in scores over time - characteristic of human speech' : 
           variationStatus === 'bad' ? '✗ Suspiciously uniform scores - typical of synthetic audio' :
           '◐ Moderate variation detected'}
        </p>

        {/* Metrics Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '28px', marginBottom: '48px' }}>
          <div style={{
            background: 'rgba(255, 255, 255, 0.02)',
            border: '2px solid #2a2a2a',
            borderRadius: '20px',
            padding: '36px',
            textAlign: 'center'
          }}>
            <div style={{ color: '#888', fontSize: '15px', fontWeight: '700', marginBottom: '16px', letterSpacing: '1px' }}>
              MEAN SCORE
            </div>
            <div style={{ color: '#fff', fontSize: '56px', fontWeight: '900', marginBottom: '12px' }}>
              {meanPercent.toFixed(1)}%
            </div>
            <div style={{ color: '#666', fontSize: '14px' }}>
              Average fakeness across all segments
            </div>
          </div>

          <div style={{
            background: 'rgba(255, 255, 255, 0.02)',
            border: `2px solid ${statusColor(variationStatus)}40`,
            borderRadius: '20px',
            padding: '36px',
            textAlign: 'center',
            boxShadow: `0 0 32px ${statusColor(variationStatus)}20`
          }}>
            <div style={{ color: '#888', fontSize: '15px', fontWeight: '700', marginBottom: '16px', letterSpacing: '1px' }}>
              VARIATION
            </div>
            <div style={{ color: statusColor(variationStatus), fontSize: '56px', fontWeight: '900', marginBottom: '12px' }}>
              {variation.toFixed(1)}%
            </div>
            <div style={{ color: statusColor(variationStatus), fontSize: '14px', fontWeight: '700' }}>
              {variationStatus === 'good' ? '✓ NATURAL VARIATION' : 
               variationStatus === 'bad' ? '✗ TOO UNIFORM' : '◐ MODERATE'}
            </div>
          </div>

          <div style={{
            background: 'rgba(255, 255, 255, 0.02)',
            border: `2px solid ${statusColor(consistencyStatus)}40`,
            borderRadius: '20px',
            padding: '36px',
            textAlign: 'center',
            boxShadow: `0 0 32px ${statusColor(consistencyStatus)}20`
          }}>
            <div style={{ color: '#888', fontSize: '15px', fontWeight: '700', marginBottom: '16px', letterSpacing: '1px' }}>
              CONSISTENCY
            </div>
            <div style={{ color: statusColor(consistencyStatus), fontSize: '56px', fontWeight: '900', marginBottom: '12px' }}>
              {consistency_index.toFixed(3)}
            </div>
            <div style={{ color: statusColor(consistencyStatus), fontSize: '14px', fontWeight: '700' }}>
              {consistencyStatus === 'good' ? '✓ NATURAL INCONSISTENCY' : 
               consistencyStatus === 'bad' ? '✗ TOO CONSISTENT' : '◐ MODERATE'}
            </div>
          </div>
        </div>

        {/* Bar Chart - LARGER */}
        <div style={{
          background: 'rgba(0, 0, 0, 0.6)',
          borderRadius: '28px',
          padding: '48px',
          border: '1px solid rgba(255, 255, 255, 0.05)',
          minHeight: '400px'
        }}>
          <h4 style={{ color: '#888', fontSize: '15px', fontWeight: '700', marginBottom: '28px', letterSpacing: '1.5px' }}>
            FAKENESS SCORE OVER TIME
          </h4>
          <p style={{ color: '#666', fontSize: '14px', marginBottom: '28px' }}>
            Each bar represents a 0.5-second segment. Height shows the model's confidence that segment is fake.
          </p>
          
          <div style={{
            display: 'flex',
            alignItems: 'flex-end',
            gap: '3px',
            height: '300px',
            background: 'rgba(255, 255, 255, 0.02)',
            borderRadius: '16px',
            padding: '20px',
            position: 'relative'
          }}>
            {/* Grid lines */}
            {[0, 25, 50, 75, 100].map(percent => (
              <div
                key={percent}
                style={{
                  position: 'absolute',
                  left: '20px',
                  right: '20px',
                  bottom: `${20 + (percent / 100) * 260}px`,
                  height: '1px',
                  background: 'rgba(255, 255, 255, 0.06)',
                  pointerEvents: 'none'
                }}
              />
            ))}
            
            {scores.map((score, idx) => {
              const height = Math.max(score * 100, 2);
              return (
                <div
                  key={idx}
                  style={{
                    flex: 1,
                    minWidth: '4px',
                    maxWidth: '16px',
                    height: `${height}%`,
                    background: score > mean_score 
                      ? `linear-gradient(180deg, ${accentColor} 0%, #dc2626 100%)`
                      : 'linear-gradient(180deg, #4ade80 0%, #22c55e 100%)',
                    borderRadius: '6px 6px 0 0',
                    transition: 'all 0.4s ease',
                    boxShadow: score > mean_score ? `0 0 12px ${accentColor}66` : '0 0 6px rgba(34, 197, 94, 0.4)',
                    animation: 'growUp 1.2s cubic-bezier(0.16, 1, 0.3, 1) forwards',
                    animationDelay: `${delay + 400 + idx * 20}ms`,
                    transformOrigin: 'bottom',
                    transform: 'scaleY(0)'
                  }}
                  title={`${timestamps[idx].toFixed(2)}s: ${(score * 100).toFixed(1)}%`}
                />
              );
            })}
          </div>

          {/* Timeline */}
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            marginTop: '16px',
            color: '#666',
            fontSize: '14px',
            fontWeight: '700'
          }}>
            <span>0s</span>
            <span>{timestamps[Math.floor(timestamps.length / 2)]?.toFixed(1)}s</span>
            <span>{timestamps[timestamps.length - 1]?.toFixed(1)}s</span>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(40px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes growUp {
          from { transform: scaleY(0); }
          to { transform: scaleY(1); }
        }
      `}</style>
    </div>
  );
};

// ===== 6. BREATHING PATTERN ANALYSIS =====
export const BreathingAnalysisCard = ({ data, accentColor, delay }) => {
  if (!data || !data.silence_segments) return null;
  
  const { silence_segments, analysis } = data;
  const { total_silence_duration, avg_silence_duration, silence_ratio, is_suspicious } = analysis;
  
  return (
    <div style={{
      opacity: 0,
      animation: 'fadeInUp 1.2s cubic-bezier(0.16, 1, 0.3, 1) forwards',
      animationDelay: `${delay}ms`
    }}>
      <div style={{
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
        border: '2px solid #2a2a2a',
        borderRadius: '32px',
        padding: '60px',
        marginBottom: '48px'
      }}>
        <h3 style={{ color: '#fff', fontSize: '38px', fontWeight: '900', marginBottom: '12px' }}>
          🫁 Breathing Pattern Analysis
        </h3>
        <p style={{ color: '#888', fontSize: '18px', marginBottom: '48px' }}>
          {is_suspicious 
            ? '⚠️ Unnatural breathing patterns detected - TTS systems often have robotic silence intervals'
            : '✓ Natural breathing patterns - consistent with human speech'}
        </p>

        {/* Metrics */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '28px', marginBottom: '48px' }}>
          <div style={{
            background: 'rgba(255, 255, 255, 0.02)',
            border: '2px solid #2a2a2a',
            borderRadius: '20px',
            padding: '36px',
            textAlign: 'center'
          }}>
            <div style={{ color: '#888', fontSize: '15px', fontWeight: '700', marginBottom: '16px' }}>
              TOTAL SILENCE
            </div>
            <div style={{ color: '#fff', fontSize: '48px', fontWeight: '900', marginBottom: '12px' }}>
              {total_silence_duration.toFixed(2)}s
            </div>
          </div>

          <div style={{
            background: 'rgba(255, 255, 255, 0.02)',
            border: '2px solid #2a2a2a',
            borderRadius: '20px',
            padding: '36px',
            textAlign: 'center'
          }}>
            <div style={{ color: '#888', fontSize: '15px', fontWeight: '700', marginBottom: '16px' }}>
              AVG PAUSE
            </div>
            <div style={{ color: '#fff', fontSize: '48px', fontWeight: '900', marginBottom: '12px' }}>
              {avg_silence_duration.toFixed(2)}s
            </div>
          </div>

          <div style={{
            background: 'rgba(255, 255, 255, 0.02)',
            border: `2px solid ${is_suspicious ? accentColor + '40' : '#22c55e40'}`,
            borderRadius: '20px',
            padding: '36px',
            textAlign: 'center',
            boxShadow: is_suspicious ? `0 0 32px ${accentColor}20` : '0 0 32px rgba(34, 197, 94, 0.2)'
          }}>
            <div style={{ color: '#888', fontSize: '15px', fontWeight: '700', marginBottom: '16px' }}>
              SILENCE RATIO
            </div>
            <div style={{
              color: is_suspicious ? accentColor : '#22c55e',
              fontSize: '48px',
              fontWeight: '900',
              marginBottom: '12px'
            }}>
              {(silence_ratio * 100).toFixed(1)}%
            </div>
          </div>
        </div>

        {/* Silence Timeline */}
        <div style={{
          background: 'rgba(0, 0, 0, 0.6)',
          borderRadius: '28px',
          padding: '48px',
          border: '1px solid rgba(255, 255, 255, 0.05)'
        }}>
          <h4 style={{ color: '#888', fontSize: '15px', fontWeight: '700', marginBottom: '32px', letterSpacing: '1.5px' }}>
            BREATHING / PAUSE TIMELINE
          </h4>
          
          <div style={{ position: 'relative', height: '120px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '16px', padding: '20px' }}>
            {/* Audio regions (green) and silence regions (red) */}
            {silence_segments.map((seg, idx) => {
              const x = (seg.start / (silence_segments[silence_segments.length - 1]?.end || 1)) * 100;
              const width = ((seg.end - seg.start) / (silence_segments[silence_segments.length - 1]?.end || 1)) * 100;
              
              return (
                <div
                  key={idx}
                  style={{
                    position: 'absolute',
                    left: `${x}%`,
                    width: `${width}%`,
                    height: '80px',
                    background: is_suspicious ? `linear-gradient(180deg, ${accentColor}, #dc2626)` : 'linear-gradient(180deg, #fbbf24, #f59e0b)',
                    borderRadius: '8px',
                    animation: 'slideDown 0.8s ease-out forwards',
                    animationDelay: `${delay + 600 + idx * 100}ms`,
                    opacity: 0,
                    boxShadow: is_suspicious ? `0 4px 20px ${accentColor}66` : '0 4px 20px rgba(251, 191, 36, 0.4)'
                  }}
                />
              );
            })}
          </div>

          <div style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '48px',
            marginTop: '32px',
            fontSize: '15px',
            fontWeight: '700'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{
                width: '40px',
                height: '8px',
                background: 'linear-gradient(90deg, #22c55e, #16a34a)',
                borderRadius: '4px'
              }} />
              <span style={{ color: '#22c55e' }}>SPEECH</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{
                width: '40px',
                height: '8px',
                background: is_suspicious ? `linear-gradient(90deg, ${accentColor}, #dc2626)` : 'linear-gradient(90deg, #fbbf24, #f59e0b)',
                borderRadius: '4px'
              }} />
              <span style={{ color: is_suspicious ? accentColor : '#fbbf24' }}>PAUSES</span>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(40px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideDown {
          from { opacity: 0; transform: translateY(-20px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
};

// ===== 7. EXPERT AGREEMENT - ENHANCED =====
export const ExpertAgreementCard = ({ data, accentColor, delay }) => {
  const experts = data.experts || {};
  const expertNames = Object.keys(experts);

  return (
    <div style={{
      opacity: 0,
      animation: 'fadeInUp 1.2s cubic-bezier(0.16, 1, 0.3, 1) forwards',
      animationDelay: `${delay}ms`
    }}>
      <div style={{
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
        border: '2px solid #2a2a2a',
        borderRadius: '32px',
        padding: '60px',
        marginBottom: '48px'
      }}>
        <h3 style={{ color: '#fff', fontSize: '38px', fontWeight: '900', marginBottom: '12px' }}>
          🤝 Expert Agreement Analysis
        </h3>
        <p style={{ color: '#888', fontSize: '18px', lineHeight: '1.7', marginBottom: '48px' }}>
          {data.interpretation}
        </p>

        {/* Agreement Score - LARGE */}
        <div style={{
          background: 'rgba(255, 255, 255, 0.02)',
          border: '2px solid #2a2a2a',
          borderRadius: '24px',
          padding: '40px',
          marginBottom: '48px',
          textAlign: 'center'
        }}>
          <div style={{ color: '#888', fontSize: '16px', fontWeight: '700', marginBottom: '20px', letterSpacing: '2px' }}>
            AGREEMENT SCORE
          </div>
          <div style={{ color: accentColor, fontSize: '72px', fontWeight: '900', marginBottom: '24px' }}>
            {(data.agreement_score * 100).toFixed(1)}%
          </div>
          <div style={{
            width: '100%',
            height: '16px',
            background: 'rgba(255, 255, 255, 0.05)',
            borderRadius: '50px',
            overflow: 'hidden'
          }}>
            <div style={{
              width: `${data.agreement_score * 100}%`,
              height: '100%',
              background: `linear-gradient(90deg, ${accentColor} 0%, ${accentColor}aa 100%)`,
              borderRadius: '50px',
              transition: 'width 1.5s cubic-bezier(0.16, 1, 0.3, 1)',
              transitionDelay: `${delay}ms`,
              boxShadow: `0 0 24px ${accentColor}88`
            }} />
          </div>
        </div>

        {/* Expert Cards - ENHANCED */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '28px' }}>
          {expertNames.map((expertName, idx) => {
            const expert = experts[expertName];
            const isFakeExpert = expert.prediction === 'FAKE';
            
            return (
              <div key={idx} style={{
                background: 'rgba(255, 255, 255, 0.02)',
                border: `3px solid ${isFakeExpert ? 'rgba(239, 68, 68, 0.4)' : 'rgba(34, 197, 94, 0.4)'}`,
                borderRadius: '24px',
                padding: '36px',
                position: 'relative',
                overflow: 'hidden',
                animation: 'slideIn 0.8s ease-out forwards',
                animationDelay: `${delay + 400 + idx * 200}ms`,
                opacity: 0
              }}>
                <div style={{
                  position: 'absolute',
                  top: '-50%',
                  right: '-50%',
                  width: '150%',
                  height: '150%',
                  background: `radial-gradient(circle, ${isFakeExpert ? 'rgba(239, 68, 68, 0.15)' : 'rgba(34, 197, 94, 0.15)'} 0%, transparent 70%)`,
                  pointerEvents: 'none'
                }} />

                <div style={{ position: 'relative', zIndex: 1 }}>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '16px',
                    marginBottom: '28px'
                  }}>
                    <div style={{
                      width: '56px',
                      height: '56px',
                      background: isFakeExpert ? 'rgba(239, 68, 68, 0.2)' : 'rgba(34, 197, 94, 0.2)',
                      borderRadius: '16px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '28px'
                    }}>
                      {expertName.includes('hubert') ? '🎵' : '🔊'}
                    </div>
                    <div>
                      <div style={{ color: '#fff', fontSize: '22px', fontWeight: '800' }}>
                        {expertName.includes('hubert') ? 'HuBERT' : 'Wav2Vec2'}
                      </div>
                      <div style={{ color: '#666', fontSize: '14px' }}>
                        {expertName.includes('hubert') ? 'Acoustic Expert' : 'Linguistic Expert'}
                      </div>
                    </div>
                  </div>

                  <div style={{ marginBottom: '24px' }}>
                    <div style={{ color: '#888', fontSize: '13px', marginBottom: '8px', fontWeight: '700' }}>Prediction</div>
                    <div style={{
                      color: isFakeExpert ? '#ef4444' : '#22c55e',
                      fontSize: '32px',
                      fontWeight: '900'
                    }}>
                      {expert.prediction}
                    </div>
                  </div>

                  <div style={{ marginBottom: '24px' }}>
                    <div style={{ color: '#888', fontSize: '13px', marginBottom: '8px', fontWeight: '700' }}>Confidence</div>
                    <div style={{ color: '#fff', fontSize: '26px', fontWeight: '800' }}>
                      {(expert.prob_fake * 100).toFixed(1)}%
                    </div>
                  </div>

                  <div>
                    <div style={{ color: '#888', fontSize: '13px', marginBottom: '8px', fontWeight: '700' }}>Gate Weight</div>
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '12px'
                    }}>
                      <div style={{
                        flex: 1,
                        height: '10px',
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
                      <span style={{ color: '#fff', fontSize: '18px', fontWeight: '700', minWidth: '60px' }}>
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

      <style>{`
        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(40px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideIn {
          from { opacity: 0; transform: translateX(-20px); }
          to { opacity: 1; transform: translateX(0); }
        }
      `}</style>
    </div>
  );
};
