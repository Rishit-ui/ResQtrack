import React from 'react';
import { Link } from 'react-router-dom';
import { Button, Card, Badge } from '../components/ui';
import {
  ShieldAlert,
  Activity,
  Radio,
  Zap,
  Eye,
  Clock,
  ArrowRight,
  Video,
  Hospital,
  Compass,
  CheckCircle,
  Cpu,
  Server,
  Lock,
} from 'lucide-react';

export const LandingPage: React.FC = () => {
  return (
    <div style={{ backgroundColor: '#090B0A', minHeight: '100vh', color: '#F2F0E9' }}>
      {/* Hero Section */}
      <section
        style={{
          position: 'relative',
          padding: '80px 24px 60px',
          overflow: 'hidden',
          borderBottom: '1px solid #303832',
          background: 'radial-gradient(ellipse at 50% 20%, rgba(242, 162, 58, 0.08) 0%, rgba(9, 11, 10, 0) 70%)',
        }}
      >
        <div style={{ maxWidth: '1200px', margin: '0 auto', textAlign: 'center' }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', marginBottom: '20px' }}>
            <Badge variant="amber">AI-POWERED ACCIDENT DETECTION & EMERGENCY RESPONSE</Badge>
          </div>

          <h1
            style={{
              fontSize: 'clamp(36px, 6vw, 64px)',
              fontWeight: 800,
              lineHeight: 1.15,
              letterSpacing: '-1.5px',
              margin: '0 auto 24px',
              maxWidth: '900px',
              color: '#F2F0E9',
            }}
          >
            Every Second Matters.{' '}
            <span style={{ color: '#F2A23A', display: 'inline-block' }}>Every Response Counts.</span>
          </h1>

          <p
            style={{
              fontSize: 'clamp(16px, 2vw, 20px)',
              color: '#A3ACA5',
              maxWidth: '750px',
              margin: '0 auto 36px',
              lineHeight: 1.6,
            }}
          >
            From the millisecond of impact, ResQTrack detects collisions through physical vision evidence,
            alerts the nearest emergency responders over real-time WebSockets, and routes ambulances directly to
            specialized trauma facilities.
          </p>

          <div
            style={{
              display: 'flex',
              gap: '16px',
              justifyContent: 'center',
              flexWrap: 'wrap',
              marginBottom: '50px',
            }}
          >
            <Link to="/register">
              <Button variant="primary" style={{ padding: '12px 28px', fontSize: '15px' }}>
                Citizen Portal <ArrowRight size={18} />
              </Button>
            </Link>
            <Link to="/login">
              <Button variant="secondary" style={{ padding: '12px 28px', fontSize: '15px' }}>
                Control Room Login
              </Button>
            </Link>
          </div>

          {/* Live Preview Card */}
          <div
            style={{
              maxWidth: '1000px',
              margin: '0 auto',
              backgroundColor: '#151917',
              border: '1px solid #303832',
              borderRadius: '16px',
              padding: '20px',
              boxShadow: '0 25px 60px rgba(0, 0, 0, 0.7)',
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                borderBottom: '1px solid #303832',
                paddingBottom: '12px',
                marginBottom: '16px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: '#D95757' }} />
                <div style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: '#E7AE4B' }} />
                <div style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: '#55A77B' }} />
                <span style={{ fontSize: '12px', color: '#A3ACA5', marginLeft: '12px', fontFamily: 'monospace' }}>
                  ResQTrack Control Room v3.0 • Live Telemetry Active
                </span>
              </div>
              <Badge variant="amber">AI DETECTOR CONNECTED</Badge>
            </div>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '16px',
                textAlign: 'left',
              }}
            >
              <div style={{ backgroundColor: '#111512', padding: '14px', borderRadius: '10px', border: '1px solid #303832' }}>
                <div style={{ fontSize: '11px', color: '#A3ACA5', textTransform: 'uppercase' }}>Vision Tracking</div>
                <div style={{ fontSize: '20px', fontWeight: 700, color: '#F2A23A', margin: '4px 0' }}>YOLO11 + ByteTrack</div>
                <div style={{ fontSize: '12px', color: '#55A77B' }}>30+ FPS Real-time HUD</div>
              </div>
              <div style={{ backgroundColor: '#111512', padding: '14px', borderRadius: '10px', border: '1px solid #303832' }}>
                <div style={{ fontSize: '11px', color: '#A3ACA5', textTransform: 'uppercase' }}>Classification</div>
                <div style={{ fontSize: '20px', fontWeight: 700, color: '#F2F0E9', margin: '4px 0' }}>Temporal ML</div>
                <div style={{ fontSize: '12px', color: '#A3ACA5' }}>Multi-frame kinematics</div>
              </div>
              <div style={{ backgroundColor: '#111512', padding: '14px', borderRadius: '10px', border: '1px solid #303832' }}>
                <div style={{ fontSize: '11px', color: '#A3ACA5', textTransform: 'uppercase' }}>Dispatch Speed</div>
                <div style={{ fontSize: '20px', fontWeight: 700, color: '#55A77B', margin: '4px 0' }}>&lt; 500 ms</div>
                <div style={{ fontSize: '12px', color: '#A3ACA5' }}>Instant crew push</div>
              </div>
              <div style={{ backgroundColor: '#111512', padding: '14px', borderRadius: '10px', border: '1px solid #303832' }}>
                <div style={{ fontSize: '11px', color: '#A3ACA5', textTransform: 'uppercase' }}>Hospital Routing</div>
                <div style={{ fontSize: '20px', fontWeight: 700, color: '#F2F0E9', margin: '4px 0' }}>OSRM + Traffic</div>
                <div style={{ fontSize: '12px', color: '#55A77B' }}>Least road travel time</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Workflow Section */}
      <section id="how-it-works" style={{ padding: '80px 24px', borderBottom: '1px solid #303832' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <div style={{ textAlign: 'center', marginBottom: '60px' }}>
            <Badge variant="amber">END-TO-END PIPELINE</Badge>
            <h2 style={{ fontSize: '32px', fontWeight: 700, margin: '14px 0 10px', color: '#F2F0E9' }}>
              How ResQTrack Responds to Emergencies
            </h2>
            <p style={{ color: '#A3ACA5', maxWidth: '600px', margin: '0 auto' }}>
              Autonomous detection replaces unreliable bystander calling with immediate, evidence-backed emergency dispatch.
            </p>
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
              gap: '16px',
              position: 'relative',
            }}
          >
            {[
              { step: '01', title: 'Camera Stream', icon: <Video size={22} color="#F2A23A" />, desc: 'Live CCTV / traffic feed ingested frame by frame.' },
              { step: '02', title: 'AI Detection', icon: <Eye size={22} color="#F2A23A" />, desc: 'YOLO11 identifies vehicles, pedestrians, and trajectories.' },
              { step: '03', title: 'Evidence Engine', icon: <Cpu size={22} color="#F2A23A" />, desc: 'Multi-frame kinematics verify approach, impact, and immobility.' },
              { step: '04', title: 'Control Room', icon: <Activity size={22} color="#F2A23A" />, desc: 'Incident pushed instantly with annotated scene snapshot.' },
              { step: '05', title: 'Crew Dispatch', icon: <Radio size={22} color="#F2A23A" />, desc: 'Nearest ambulance receives alert, GPS route, and ETA.' },
              { step: '06', title: 'Hospital Routing', icon: <Hospital size={22} color="#F2A23A" />, desc: 'Facilities ranked by road time; least-traffic route plotted.' },
            ].map((item, i) => (
              <Card key={i} style={{ backgroundColor: '#111512', border: '1px solid #303832', padding: '20px' }}>
                <div style={{ fontSize: '11px', fontFamily: 'monospace', color: '#F2A23A', fontWeight: 700, marginBottom: '12px' }}>
                  STEP {item.step}
                </div>
                <div style={{ marginBottom: '12px' }}>{item.icon}</div>
                <h3 style={{ fontSize: '16px', fontWeight: 600, margin: '0 0 8px', color: '#F2F0E9' }}>
                  {item.title}
                </h3>
                <p style={{ fontSize: '12px', color: '#A3ACA5', margin: 0, lineHeight: 1.5 }}>
                  {item.desc}
                </p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section id="features" style={{ padding: '80px 24px', borderBottom: '1px solid #303832', backgroundColor: '#0C0F0D' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <div style={{ textAlign: 'center', marginBottom: '50px' }}>
            <Badge variant="amber">KEY CAPABILITIES</Badge>
            <h2 style={{ fontSize: '32px', fontWeight: 700, margin: '14px 0 10px', color: '#F2F0E9' }}>
              Built for High-Stakes Operations
            </h2>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '24px' }}>
            <Card style={{ padding: '24px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                <div style={{ width: '36px', height: '36px', borderRadius: '8px', backgroundColor: 'rgba(242, 162, 58, 0.1)', display: 'grid', placeItems: 'center' }}>
                  <Zap size={20} color="#F2A23A" />
                </div>
                <h3 style={{ fontSize: '17px', fontWeight: 600, margin: 0 }}>Zero-Latency Alerting</h3>
              </div>
              <p style={{ fontSize: '13px', color: '#A3ACA5', lineHeight: 1.6, margin: 0 }}>
                Persistent WebSockets bypass slow HTTP polling. When a crash is verified, alerts reach the control room and responder devices within 500 milliseconds.
              </p>
            </Card>

            <Card style={{ padding: '24px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                <div style={{ width: '36px', height: '36px', borderRadius: '8px', backgroundColor: 'rgba(85, 167, 123, 0.1)', display: 'grid', placeItems: 'center' }}>
                  <Compass size={20} color="#55A77B" />
                </div>
                <h3 style={{ fontSize: '17px', fontWeight: 600, margin: 0 }}>Real Road-Time Ranking</h3>
              </div>
              <p style={{ fontSize: '13px', color: '#A3ACA5', lineHeight: 1.6, margin: 0 }}>
                Hospitals are ranked by actual driving time computed via OSRM and live traffic matrices rather than misleading Euclidean straight-line distance.
              </p>
            </Card>

            <Card style={{ padding: '24px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                <div style={{ width: '36px', height: '36px', borderRadius: '8px', backgroundColor: 'rgba(217, 87, 87, 0.1)', display: 'grid', placeItems: 'center' }}>
                  <CheckCircle size={20} color="#D95757" />
                </div>
                <h3 style={{ fontSize: '17px', fontWeight: 600, margin: 0 }}>Geofence Arrival Verification</h3>
              </div>
              <p style={{ fontSize: '13px', color: '#A3ACA5', lineHeight: 1.6, margin: 0 }}>
                Responders stream coordinates during transit; automated 75-metre geofencing verifies arrival at the accident scene without requiring manual button taps.
              </p>
            </Card>

            <Card style={{ padding: '24px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                <div style={{ width: '36px', height: '36px', borderRadius: '8px', backgroundColor: 'rgba(242, 162, 58, 0.1)', display: 'grid', placeItems: 'center' }}>
                  <Lock size={20} color="#F2A23A" />
                </div>
                <h3 style={{ fontSize: '17px', fontWeight: 600, margin: 0 }}>Role-Based Access & Security</h3>
              </div>
              <p style={{ fontSize: '13px', color: '#A3ACA5', lineHeight: 1.6, margin: 0 }}>
                Strict boundaries between Citizens, Emergency Dispatchers, and Field Responders with JWT sessions, bcrypt hashing, and audit trails.
              </p>
            </Card>

            <Card style={{ padding: '24px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                <div style={{ width: '36px', height: '36px', borderRadius: '8px', backgroundColor: 'rgba(85, 167, 123, 0.1)', display: 'grid', placeItems: 'center' }}>
                  <Server size={20} color="#55A77B" />
                </div>
                <h3 style={{ fontSize: '17px', fontWeight: 600, margin: 0 }}>Offline Fallback Facilities</h3>
              </div>
              <p style={{ fontSize: '13px', color: '#A3ACA5', lineHeight: 1.6, margin: 0 }}>
                Even when internet access to OpenStreetMap or Overpass drops, pre-seeded local SQLite hospital datasets guarantee recommendations are never blank.
              </p>
            </Card>

            <Card style={{ padding: '24px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                <div style={{ width: '36px', height: '36px', borderRadius: '8px', backgroundColor: 'rgba(242, 162, 58, 0.1)', display: 'grid', placeItems: 'center' }}>
                  <Activity size={20} color="#F2A23A" />
                </div>
                <h3 style={{ fontSize: '17px', fontWeight: 600, margin: 0 }}>Citizen Manual Reporting</h3>
              </div>
              <p style={{ fontSize: '13px', color: '#A3ACA5', lineHeight: 1.6, margin: 0 }}>
                Bystanders can submit manual incident reports with GPS fixes and photos into an unverified review queue for control room approval or escalation.
              </p>
            </Card>
          </div>
        </div>
      </section>

      {/* Tech Stack */}
      <section id="tech" style={{ padding: '60px 24px', borderBottom: '1px solid #303832' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', textAlign: 'center' }}>
          <Badge variant="amber">TECHNOLOGY STACK</Badge>
          <h2 style={{ fontSize: '28px', fontWeight: 700, margin: '14px 0 30px', color: '#F2F0E9' }}>
            Industrial-Grade Open Source Infrastructure
          </h2>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', justifyContent: 'center' }}>
            {['FastAPI', 'YOLO11', 'ByteTrack', 'PyTorch', 'OpenCV', 'Scikit-Learn', 'SQLite / PostgreSQL', 'React 18', 'TypeScript', 'Leaflet', 'OSRM Routing', 'Overpass API'].map((tech, i) => (
              <span
                key={i}
                style={{
                  backgroundColor: '#151917',
                  border: '1px solid #303832',
                  padding: '8px 16px',
                  borderRadius: '20px',
                  fontSize: '13px',
                  fontWeight: 600,
                  color: '#F2F0E9',
                }}
              >
                {tech}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer style={{ padding: '40px 24px', backgroundColor: '#090B0A', color: '#A3ACA5', fontSize: '12px' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <b style={{ color: '#F2F0E9' }}>ResQTrack Emergency Response</b> — AI Accident Detection & Autonomous Dispatch
          </div>
          <div>
            Smart India Hackathon 2026 Prototype • Confidential Emergency Services Use Only
          </div>
        </div>
      </footer>
    </div>
  );
};
