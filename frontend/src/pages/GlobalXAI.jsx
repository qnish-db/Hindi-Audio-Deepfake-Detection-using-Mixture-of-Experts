import React, { useState, useEffect } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, ScatterChart, Scatter, Cell } from 'recharts';

const GlobalXAI = ({ onClose }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Load JSON data from global_xai_results_FINAL (NEW CORRECTED VERSION)
    const loadData = async () => {
      try {
        // Try new version first (with cross-reference fixes)
        const res = await fetch('/global_xai_results_FINAL/global_xai_results.json');
        const jsonData = await res.json();
        setData(jsonData);
        setLoading(false);
      } catch (err) {
        setError('Failed to load global XAI data. Please run global_xai_CORRECTED.py first.');
        setLoading(false);
      }
    };

    loadData();
  }, []);

  if (loading) {
    return (
      <div style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <div style={{
          width: '60px',
          height: '60px',
          border: '4px solid #333',
          borderTop: '4px solid #fff',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite'
        }} />
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '40px'
      }}>
        <div style={{
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid #ef4444',
          borderRadius: '12px',
          padding: '32px',
          maxWidth: '500px',
          color: '#fff'
        }}>
          <p style={{ fontSize: '18px', marginBottom: '12px' }}>{error}</p>
          <p style={{ color: '#999', fontSize: '14px' }}>Run the global_xai_analysis.py script first</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return <div>Loading...</div>;
  }

  // Prepare temporal patterns data
  const temporalData = data.temporal_patterns.real.avg_scores_by_position.map((realScore, idx) => ({
    position: `${idx * 2}%`,
    Real: realScore,
    Fake: data.temporal_patterns.fake.avg_scores_by_position[idx]
  }));

  // Prepare frequency band data
  const freqBands = Object.keys(data.frequency_analysis.band_statistics.real);
  const frequencyData = freqBands.map(band => ({
    band: band.replace('Hz', ''),
    'Real Variance': data.frequency_analysis.band_statistics.real[band].mean_variance,
    'Fake Variance': data.frequency_analysis.band_statistics.fake[band].mean_variance
  }));

  // Prepare cross-referenced insights (only entries with temporal peaks)
  const crossInsights = (data.cross_referenced_insights || [])
    .filter(it => it && it.temporal_peak)
    .slice(0, 5);

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
      padding: '40px 20px',
      fontFamily: "'Montserrat', sans-serif"
    }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        
        {/* Back Button */}
        <button
          onClick={onClose}
          style={{
            marginBottom: '32px',
            padding: '12px 24px',
            background: 'rgba(255, 255, 255, 0.05)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '8px',
            color: '#fff',
            fontSize: '14px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            transition: 'all 0.3s ease'
          }}
          onMouseOver={(e) => e.target.style.background = 'rgba(255, 255, 255, 0.1)'}
          onMouseOut={(e) => e.target.style.background = 'rgba(255, 255, 255, 0.05)'}
        >
          ← Back to Home
        </button>
        
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '48px' }}>
          <div style={{ marginBottom: '12px' }}>
            <div style={{ 
              display: 'inline-block',
              padding: '6px 16px',
              background: 'rgba(167, 139, 250, 0.1)',
              border: '1px solid rgba(167, 139, 250, 0.3)',
              borderRadius: '20px',
              fontSize: '12px',
              fontWeight: '600',
              letterSpacing: '0.5px',
              color: '#a78bfa',
              marginBottom: '20px'
            }}>
              EXPLAINABLE AI ANALYSIS
            </div>
          </div>
          <h1 style={{
            color: '#fff',
            fontSize: '52px',
            fontWeight: '700',
            marginBottom: '16px',
            letterSpacing: '-1px',
            lineHeight: '1.1'
          }}>
            Global Interpretability Report
          </h1>
          <p style={{ color: '#999', fontSize: '16px', marginBottom: '8px', lineHeight: '1.6' }}>
            Comprehensive analysis of model decision patterns across temporal, spectral, and linguistic dimensions.
          </p>
          <p style={{ color: '#666', fontSize: '14px' }}>
            Dataset: {data.summary.n_real_samples.toLocaleString()} authentic samples | {data.summary.n_fake_samples.toLocaleString()} synthetic samples
          </p>
        </div>

        {/* Summary Cards */}
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', 
          gap: '20px', 
          marginBottom: '48px' 
        }}>
          <div style={{
            background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: '16px',
            padding: '28px',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.2)'
          }}>
            <div style={{ color: '#666', fontSize: '11px', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: '600' }}>Authentic Samples</div>
            <div style={{ color: '#fff', fontSize: '36px', fontWeight: '600', marginBottom: '8px', letterSpacing: '-1px' }}>
              {data.summary.n_real_samples.toLocaleString()}
            </div>
            <div style={{ color: '#22c55e', fontSize: '13px', fontWeight: '500' }}>
              Baseline reference set
            </div>
          </div>

          <div style={{
            background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: '16px',
            padding: '28px',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.2)'
          }}>
            <div style={{ color: '#666', fontSize: '11px', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: '600' }}>Synthetic Samples</div>
            <div style={{ color: '#fff', fontSize: '36px', fontWeight: '600', marginBottom: '8px', letterSpacing: '-1px' }}>
              {data.summary.n_fake_samples.toLocaleString()}
            </div>
            <div style={{ color: '#ef4444', fontSize: '13px', fontWeight: '500' }}>
              Deepfake test set
            </div>
          </div>

          <div style={{
            background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: '16px',
            padding: '28px',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.2)'
          }}>
            <div style={{ color: '#666', fontSize: '11px', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: '600' }}>Spectral Bands</div>
            <div style={{ color: '#fff', fontSize: '36px', fontWeight: '600', marginBottom: '8px', letterSpacing: '-1px' }}>
              {freqBands.length}
            </div>
            <div style={{ color: '#a78bfa', fontSize: '13px', fontWeight: '500' }}>
              0-8 kHz coverage
            </div>
          </div>

          <div style={{
            background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: '16px',
            padding: '28px',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.2)'
          }}>
            <div style={{ color: '#666', fontSize: '11px', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: '600' }}>Linguistic Markers</div>
            <div style={{ color: '#fff', fontSize: '36px', fontWeight: '600', marginBottom: '8px', letterSpacing: '-1px' }}>
              {data.linguistic_patterns.high_risk_words.length}
            </div>
            <div style={{ color: '#fbbf24', fontSize: '13px', fontWeight: '500' }}>
              High-correlation lexemes
            </div>
          </div>
        </div>

        {/* Chart 1: Temporal Patterns */}
        <div style={{
          background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          borderRadius: '16px',
          padding: '36px',
          marginBottom: '32px',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.25)'
        }}>
          <div style={{ marginBottom: '24px' }}>
            <h2 style={{ color: '#fff', fontSize: '26px', fontWeight: '600', marginBottom: '8px', letterSpacing: '-0.5px' }}>
              Temporal Feature Importance
            </h2>
            <p style={{ color: '#888', fontSize: '14px', lineHeight: '1.6', marginBottom: '4px' }}>
              Distribution of model attention across temporal feature dimensions. Perturbation-based analysis reveals which feature positions drive classification decisions.
            </p>
            <p style={{ color: '#666', fontSize: '13px' }}>
              Detected {data.temporal_patterns.real.hotspot_regions.length + data.temporal_patterns.fake.hotspot_regions.length} high-importance regions across both classes
            </p>
          </div>
          <ResponsiveContainer width="100%" height={500}>
            <AreaChart data={temporalData} margin={{ top: 20, right: 40, left: 10, bottom: 10 }}>
              <defs>
                <linearGradient id="colorReal" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#22c55e" stopOpacity={0.5}/>
                  <stop offset="50%" stopColor="#22c55e" stopOpacity={0.2}/>
                  <stop offset="100%" stopColor="#22c55e" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="colorFake" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity={0.5}/>
                  <stop offset="50%" stopColor="#ef4444" stopOpacity={0.2}/>
                  <stop offset="100%" stopColor="#ef4444" stopOpacity={0}/>
                </linearGradient>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                  <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                  </feMerge>
                </filter>
              </defs>
              <CartesianGrid 
                strokeDasharray="1 3" 
                stroke="rgba(255,255,255,0.02)" 
                vertical={false}
                strokeWidth={1}
              />
              <XAxis 
                dataKey="position" 
                stroke="#444" 
                style={{ fontSize: '12px', fontWeight: '500', fontFamily: 'Inter, system-ui, sans-serif' }}
                tick={{ fill: '#999' }}
                axisLine={{ stroke: '#2a2a2a', strokeWidth: 1.5 }}
                tickLine={{ stroke: '#444' }}
              />
              <YAxis 
                stroke="#444" 
                style={{ fontSize: '12px', fontWeight: '500', fontFamily: 'Inter, system-ui, sans-serif' }}
                tick={{ fill: '#999' }}
                axisLine={{ stroke: '#2a2a2a', strokeWidth: 1.5 }}
                tickLine={{ stroke: '#444' }}
                label={{ 
                  value: 'Importance Score', 
                  angle: -90, 
                  position: 'insideLeft', 
                  style: { fill: '#888', fontSize: '13px', fontWeight: '600', fontFamily: 'Inter, system-ui, sans-serif' } 
                }}
              />
              <Tooltip 
                contentStyle={{ 
                  background: 'rgba(5, 5, 5, 0.97)', 
                  backdropFilter: 'blur(12px)',
                  border: '1px solid rgba(167, 139, 250, 0.2)', 
                  borderRadius: '12px', 
                  color: '#fff',
                  boxShadow: '0 8px 32px rgba(0,0,0,0.7), 0 0 0 1px rgba(255,255,255,0.05)',
                  fontSize: '13px',
                  padding: '12px 16px'
                }}
                labelStyle={{ 
                  color: '#a78bfa', 
                  fontWeight: '700', 
                  marginBottom: '8px',
                  fontSize: '14px',
                  fontFamily: 'Inter, system-ui, sans-serif'
                }}
                itemStyle={{
                  padding: '4px 0',
                  fontFamily: 'Inter, system-ui, sans-serif'
                }}
              />
              <Legend 
                wrapperStyle={{ paddingTop: '24px' }}
                iconType="line"
                iconSize={20}
                formatter={(value) => 
                  <span style={{ 
                    color: '#ddd', 
                    fontSize: '14px', 
                    fontWeight: '600',
                    fontFamily: 'Inter, system-ui, sans-serif'
                  }}>
                    {value === 'Real' ? 'Authentic Samples' : 'Synthetic Samples'}
                  </span>
                }
              />
              <Area 
                type="monotone" 
                dataKey="Real" 
                stroke="#22c55e" 
                strokeWidth={3} 
                fill="url(#colorReal)"
                dot={false}
                activeDot={{ r: 6, fill: '#22c55e', stroke: '#fff', strokeWidth: 2 }}
              />
              <Area 
                type="monotone" 
                dataKey="Fake" 
                stroke="#ef4444" 
                strokeWidth={3} 
                fill="url(#colorFake)"
                dot={false}
                activeDot={{ r: 6, fill: '#ef4444', stroke: '#fff', strokeWidth: 2 }}
              />
            </AreaChart>
          </ResponsiveContainer>
          
          {/* Hotspot Regions */}
          {(data.temporal_patterns.real.hotspot_regions.length > 0 || data.temporal_patterns.fake.hotspot_regions.length > 0) && (
            <div style={{ marginTop: '24px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div style={{ padding: '18px', background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.08) 0%, rgba(34, 197, 94, 0.04) 100%)', border: '1px solid rgba(34, 197, 94, 0.25)', borderRadius: '10px', boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)' }}>
                <div style={{ color: '#22c55e', fontSize: '11px', fontWeight: '600', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Authentic - Critical Regions</div>
                {data.temporal_patterns.real.hotspot_regions.slice(0, 3).map((h, i) => (
                  <div key={i} style={{ color: '#ccc', fontSize: '13px', marginBottom: '4px' }}>
                    {h.start_bin}-{h.end_bin} ({typeof h.avg_score === 'number' ? h.avg_score.toFixed(3) : 'N/A'})
                  </div>
                ))}
              </div>
              <div style={{ padding: '18px', background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(239, 68, 68, 0.04) 100%)', border: '1px solid rgba(239, 68, 68, 0.25)', borderRadius: '10px', boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)' }}>
                <div style={{ color: '#ef4444', fontSize: '11px', fontWeight: '600', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Synthetic - Critical Regions</div>
                {data.temporal_patterns.fake.hotspot_regions.slice(0, 3).map((h, i) => (
                  <div key={i} style={{ color: '#ccc', fontSize: '13px', marginBottom: '4px' }}>
                    {h.start_bin}-{h.end_bin} ({typeof h.avg_score === 'number' ? h.avg_score.toFixed(3) : 'N/A'})
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Chart 2: Frequency Band Analysis */}
        <div style={{
          background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          borderRadius: '16px',
          padding: '36px',
          marginBottom: '32px',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.25)'
        }}>
          <div style={{ marginBottom: '24px' }}>
            <h2 style={{ color: '#fff', fontSize: '26px', fontWeight: '600', marginBottom: '8px', letterSpacing: '-0.5px' }}>
              Spectral Variance Profiling
            </h2>
            <p style={{ color: '#888', fontSize: '14px', lineHeight: '1.6', marginBottom: '4px' }}>
              Statistical variance of mel-spectrogram energy distribution across 16 frequency bands (0-8 kHz). Text-to-speech systems exhibit characteristically reduced variance in mid-to-high frequency ranges.
            </p>
            <p style={{ color: '#666', fontSize: '13px' }}>
              Analysis identifies spectral artifacts indicative of neural vocoder synthesis
            </p>
          </div>
          <ResponsiveContainer width="100%" height={520}>
            <BarChart data={frequencyData} margin={{ top: 20, right: 40, left: 10, bottom: 90 }}>
              <defs>
                <linearGradient id="barReal" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#22c55e" stopOpacity={1}/>
                  <stop offset="100%" stopColor="#16a34a" stopOpacity={0.85}/>
                </linearGradient>
                <linearGradient id="barFake" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity={1}/>
                  <stop offset="100%" stopColor="#dc2626" stopOpacity={0.85}/>
                </linearGradient>
              </defs>
              <CartesianGrid 
                strokeDasharray="1 3" 
                stroke="rgba(255,255,255,0.02)" 
                vertical={false}
                strokeWidth={1}
              />
              <XAxis 
                dataKey="band" 
                stroke="#444" 
                angle={-45} 
                textAnchor="end" 
                height={100}
                style={{ fontSize: '11px', fontWeight: '600', fontFamily: 'Inter, system-ui, sans-serif' }}
                tick={{ fill: '#999' }}
                axisLine={{ stroke: '#2a2a2a', strokeWidth: 1.5 }}
                tickLine={{ stroke: '#444' }}
                interval={0}
              />
              <YAxis 
                stroke="#444" 
                style={{ fontSize: '12px', fontWeight: '500', fontFamily: 'Inter, system-ui, sans-serif' }}
                tick={{ fill: '#999' }}
                axisLine={{ stroke: '#2a2a2a', strokeWidth: 1.5 }}
                tickLine={{ stroke: '#444' }}
                label={{ 
                  value: 'Variance (dB)', 
                  angle: -90, 
                  position: 'insideLeft', 
                  style: { fill: '#888', fontSize: '13px', fontWeight: '600', fontFamily: 'Inter, system-ui, sans-serif' } 
                }}
              />
              <Tooltip 
                contentStyle={{ 
                  background: 'rgba(5, 5, 5, 0.97)', 
                  backdropFilter: 'blur(12px)',
                  border: '1px solid rgba(167, 139, 250, 0.2)', 
                  borderRadius: '12px', 
                  color: '#fff',
                  boxShadow: '0 8px 32px rgba(0,0,0,0.7), 0 0 0 1px rgba(255,255,255,0.05)',
                  fontSize: '13px',
                  padding: '12px 16px'
                }}
                labelStyle={{ 
                  color: '#a78bfa', 
                  fontWeight: '700', 
                  marginBottom: '8px',
                  fontSize: '14px',
                  fontFamily: 'Inter, system-ui, sans-serif'
                }}
                formatter={(value, name) => [
                  typeof value === 'number' ? value.toFixed(2) + ' dB' : value, 
                  name === 'Real Variance' ? 'Authentic Samples' : 'Synthetic Samples'
                ]}
                itemStyle={{
                  padding: '4px 0',
                  fontFamily: 'Inter, system-ui, sans-serif',
                  fontWeight: '600'
                }}
                cursor={{ fill: 'rgba(167, 139, 250, 0.05)' }}
              />
              <Legend 
                wrapperStyle={{ paddingTop: '24px' }}
                iconSize={14}
                iconType="square"
                formatter={(value) => 
                  <span style={{ 
                    color: '#ddd', 
                    fontSize: '14px', 
                    fontWeight: '600',
                    fontFamily: 'Inter, system-ui, sans-serif'
                  }}>
                    {value === 'Real Variance' ? 'Authentic Variance' : 'Synthetic Variance'}
                  </span>
                }
              />
              <Bar 
                dataKey="Real Variance" 
                fill="url(#barReal)" 
                radius={[6, 6, 0, 0]}
                maxBarSize={50}
              />
              <Bar 
                dataKey="Fake Variance" 
                fill="url(#barFake)" 
                radius={[6, 6, 0, 0]}
                maxBarSize={50}
              />
            </BarChart>
          </ResponsiveContainer>
          <div style={{ marginTop: '20px', padding: '20px', background: 'linear-gradient(135deg, rgba(167, 139, 250, 0.05) 0%, rgba(167, 139, 250, 0.02) 100%)', border: '1px solid rgba(167, 139, 250, 0.15)', borderRadius: '8px' }}>
            <div style={{ color: '#a78bfa', fontSize: '12px', fontWeight: '600', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Key Observation</div>
            <p style={{ color: '#ccc', fontSize: '14px', lineHeight: '1.7' }}>
              Synthetic samples demonstrate significantly reduced variance in the 4-6 kHz spectral region compared to authentic recordings. This pattern is characteristic of neural vocoder architectures, which tend to produce overly smooth spectral envelopes in mid-high frequency bands.
            </p>
          </div>
        </div>

        {/* High-Risk Words */}
        <div style={{
          background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          borderRadius: '16px',
          padding: '36px',
          marginBottom: '32px',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.25)'
        }}>
          <div style={{ marginBottom: '32px' }}>
            <h2 style={{ color: '#fff', fontSize: '26px', fontWeight: '600', marginBottom: '8px', letterSpacing: '-0.5px' }}>
              Linguistic Correlation Analysis
            </h2>
            <p style={{ color: '#888', fontSize: '14px', lineHeight: '1.6', marginBottom: '4px' }}>
              Lexical items exhibiting statistically significant correlation with elevated model confidence scores for synthetic classification. These patterns may reflect phonetic complexity or pronunciation artifacts in TTS systems.
            </p>
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginTop: '12px' }}>
              <div style={{ 
                padding: '6px 14px', 
                background: 'rgba(251, 191, 36, 0.1)', 
                border: '1px solid rgba(251, 191, 36, 0.3)',
                borderRadius: '20px',
                fontSize: '12px',
                fontWeight: '600',
                color: '#fbbf24'
              }}>
                {data.linguistic_patterns.high_risk_words.length} Markers Detected
              </div>
            </div>
          </div>
          
          {/* Table Header */}
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: '60px 1fr 140px 120px 120px',
            gap: '16px',
            padding: '16px 20px',
            background: 'rgba(255, 255, 255, 0.02)',
            borderRadius: '10px 10px 0 0',
            borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
            marginBottom: '0'
          }}>
            <div style={{ color: '#888', fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Rank</div>
            <div style={{ color: '#888', fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Lexeme</div>
            <div style={{ color: '#888', fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Confidence</div>
            <div style={{ color: '#888', fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Frequency</div>
            <div style={{ color: '#888', fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Impact</div>
          </div>
          
          {/* Table Rows */}
          <div style={{ 
            background: 'rgba(255, 255, 255, 0.01)',
            borderRadius: '0 0 10px 10px',
            overflow: 'hidden'
          }}>
            {data.linguistic_patterns.high_risk_words.slice(0, 10).map((word, idx) => {
              const score = typeof word.avg_score_when_present === 'number' ? word.avg_score_when_present : 0;
              const confidence = score * 100;
              const impactLevel = score > 0.8 ? 'Critical' : score > 0.6 ? 'High' : 'Moderate';
              const impactColor = score > 0.8 ? '#ef4444' : score > 0.6 ? '#f59e0b' : '#fbbf24';
              
              return (
                <div 
                  key={idx} 
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '60px 1fr 140px 120px 120px',
                    gap: '16px',
                    padding: '20px',
                    borderBottom: idx < 9 ? '1px solid rgba(255, 255, 255, 0.03)' : 'none',
                    transition: 'all 0.2s ease',
                    cursor: 'default'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(251, 191, 36, 0.03)'}
                  onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                >
                  {/* Rank Badge */}
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <div style={{
                      width: '32px',
                      height: '32px',
                      borderRadius: '8px',
                      background: idx < 3 ? `linear-gradient(135deg, ${impactColor}40, ${impactColor}20)` : 'rgba(255, 255, 255, 0.05)',
                      border: `1px solid ${idx < 3 ? impactColor + '40' : 'rgba(255, 255, 255, 0.08)'}`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '14px',
                      fontWeight: '700',
                      color: idx < 3 ? impactColor : '#999'
                    }}>
                      {idx + 1}
                    </div>
                  </div>
                  
                  {/* Word */}
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <span style={{ 
                      fontSize: '18px', 
                      fontWeight: '600', 
                      color: '#fff',
                      fontFamily: 'Noto Sans Devanagari, sans-serif'
                    }}>
                      {word.word}
                    </span>
                  </div>
                  
                  {/* Confidence Bar */}
                  <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    <div style={{ 
                      height: '6px', 
                      background: 'rgba(255, 255, 255, 0.05)', 
                      borderRadius: '3px',
                      overflow: 'hidden',
                      marginBottom: '6px'
                    }}>
                      <div style={{
                        height: '100%',
                        width: `${confidence}%`,
                        background: `linear-gradient(90deg, ${impactColor}, ${impactColor}cc)`,
                        borderRadius: '3px',
                        transition: 'width 0.3s ease'
                      }} />
                    </div>
                    <span style={{ fontSize: '12px', fontWeight: '600', color: '#ccc' }}>
                      {score.toFixed(3)}
                    </span>
                  </div>
                  
                  {/* Frequency */}
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <span style={{ fontSize: '15px', fontWeight: '600', color: '#aaa' }}>
                      {word.occurrences}×
                    </span>
                  </div>
                  
                  {/* Impact Badge */}
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <div style={{
                      padding: '6px 12px',
                      background: `${impactColor}15`,
                      border: `1px solid ${impactColor}40`,
                      borderRadius: '6px',
                      fontSize: '11px',
                      fontWeight: '700',
                      color: impactColor,
                      textTransform: 'uppercase',
                      letterSpacing: '0.5px'
                    }}>
                      {impactLevel}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Cross-Referenced Insights */}
        {data.cross_referenced_insights && data.cross_referenced_insights.length > 0 && (
          <div style={{
            background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: '16px',
            padding: '36px',
            marginBottom: '32px',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.25)'
          }}>
            <div style={{ marginBottom: '24px' }}>
              <h2 style={{ color: '#fff', fontSize: '26px', fontWeight: '600', marginBottom: '8px', letterSpacing: '-0.5px' }}>
                Multi-Modal Alignment Analysis
              </h2>
              <p style={{ color: '#888', fontSize: '14px', lineHeight: '1.6', marginBottom: '4px' }}>
                Cross-referenced analysis of temporal importance peaks with spectral anomalies and linguistic content. Reveals systematic patterns where model attention, frequency artifacts, and specific lexical items co-occur.
              </p>
              <p style={{ color: '#666', fontSize: '13px' }}>
                Provides interpretable insights into model decision-making across multiple modalities
              </p>
            </div>
            <div style={{ display: 'grid', gap: '16px' }}>
              {crossInsights.map((insight, idx) => (
                <div key={idx} style={{
                  padding: '24px',
                  background: 'linear-gradient(135deg, rgba(167, 139, 250, 0.08) 0%, rgba(167, 139, 250, 0.04) 100%)',
                  border: '1px solid rgba(167, 139, 250, 0.25)',
                  borderRadius: '12px',
                  boxShadow: '0 2px 12px rgba(0, 0, 0, 0.2)'
                }}>
                  <div style={{ color: '#a78bfa', fontSize: '14px', fontWeight: '600', marginBottom: '12px' }}>
                    Sample: {insight.sample_id || 'Unknown'}
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px' }}>
                    {insight.temporal_peak && (
                      <div>
                        <div style={{ color: '#666', fontSize: '10px', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.5px', fontWeight: '600' }}>Temporal Peak</div>
                        <div style={{ color: '#fff', fontSize: '13px' }}>{insight.temporal_peak.time_range || 'N/A'}</div>
                        <div style={{ color: '#888', fontSize: '12px' }}>Score: {insight.temporal_peak.score?.toFixed(3) || 'N/A'}</div>
                      </div>
                    )}
                    {insight.frequency_anomaly && (
                      <div>
                        <div style={{ color: '#666', fontSize: '10px', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.5px', fontWeight: '600' }}>Spectral Anomaly</div>
                        <div style={{ color: '#fff', fontSize: '13px' }}>{insight.frequency_anomaly.freq_range || 'N/A'}</div>
                        <div style={{ color: '#888', fontSize: '12px' }}>{insight.frequency_anomaly.interpretation || 'N/A'}</div>
                      </div>
                    )}
                    {insight.linguistic_content && (
                      <div>
                        <div style={{ color: '#666', fontSize: '10px', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.5px', fontWeight: '600' }}>Linguistic Content</div>
                        <div style={{ color: '#fff', fontSize: '13px' }}>"{insight.linguistic_content.word || 'N/A'}"</div>
                        <div style={{ color: '#888', fontSize: '12px' }}>Est. time: {insight.linguistic_content.estimated_time || 'N/A'}</div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Summary Stats */}
        <div style={{
          background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          borderRadius: '16px',
          padding: '36px',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.25)'
        }}>
          <h2 style={{ color: '#fff', fontSize: '26px', fontWeight: '600', marginBottom: '24px', letterSpacing: '-0.5px' }}>
            Executive Summary
          </h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px' }}>
            <div>
              <h3 style={{ color: '#22c55e', fontSize: '15px', fontWeight: '600', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Temporal Analysis
              </h3>
              <p style={{ color: '#ccc', fontSize: '13px', lineHeight: '1.8' }}>
                <span style={{ color: '#888' }}>KL Divergence:</span> <strong style={{ color: '#fff', fontWeight: '600' }}>
                {data.temporal_patterns.comparison?.divergence_metric === null ? 'Very High' : data.temporal_patterns.comparison?.divergence_metric?.toFixed(3) || 'N/A'}
                </strong><br/>
                <span style={{ color: '#888' }}>Authentic hotspots:</span> <strong style={{ color: '#fff', fontWeight: '600' }}>{data.temporal_patterns.real.hotspot_regions.length}</strong><br/>
                <span style={{ color: '#888' }}>Synthetic hotspots:</span> <strong style={{ color: '#fff', fontWeight: '600' }}>{data.temporal_patterns.fake.hotspot_regions.length}</strong>
              </p>
            </div>

            <div>
              <h3 style={{ color: '#a78bfa', fontSize: '15px', fontWeight: '600', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Spectral Analysis
              </h3>
              <p style={{ color: '#ccc', fontSize: '13px', lineHeight: '1.8' }}>
                <span style={{ color: '#888' }}>Frequency bands:</span> <strong style={{ color: '#fff', fontWeight: '600' }}>{freqBands.length}</strong><br/>
                <span style={{ color: '#888' }}>Anomalous bands:</span> <strong style={{ color: '#fff', fontWeight: '600' }}>{data.frequency_analysis.suspicious_bands?.length || 0}</strong><br/>
                <span style={{ color: '#888' }}>Coverage:</span> <strong style={{ color: '#fff', fontWeight: '600' }}>0-8 kHz</strong>
              </p>
            </div>

            <div>
              <h3 style={{ color: '#fbbf24', fontSize: '15px', fontWeight: '600', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Linguistic Analysis
              </h3>
              <p style={{ color: '#ccc', fontSize: '13px', lineHeight: '1.8' }}>
                <span style={{ color: '#888' }}>Correlation markers:</span> <strong style={{ color: '#fff', fontWeight: '600' }}>{data.linguistic_patterns.high_risk_words.length}</strong><br/>
                <span style={{ color: '#888' }}>Authentic complexity:</span> <strong style={{ color: '#fff', fontWeight: '600' }}>{typeof data.linguistic_patterns.syllable_complexity.real?.avg_syllables_per_word === 'number' ? data.linguistic_patterns.syllable_complexity.real.avg_syllables_per_word.toFixed(2) : 'N/A'}</strong> syll/word<br/>
                <span style={{ color: '#888' }}>Synthetic complexity:</span> <strong style={{ color: '#fff', fontWeight: '600' }}>{typeof data.linguistic_patterns.syllable_complexity.fake?.avg_syllables_per_word === 'number' ? data.linguistic_patterns.syllable_complexity.fake.avg_syllables_per_word.toFixed(2) : 'N/A'}</strong> syll/word
              </p>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default GlobalXAI;
