import React, { useState } from 'react';
import { Incident } from '../types';
import { Badge, Button, Card } from './ui';
import { apiCloseIncident } from '../api';
import { X, Camera, AlertTriangle, Clock, MapPin, Truck, CheckCircle2, ShieldAlert } from 'lucide-react';

interface IncidentDetailModalProps {
  incident: Incident | null;
  onClose: () => void;
  onRefresh?: () => void;
}

export const IncidentDetailModal: React.FC<IncidentDetailModalProps> = ({
  incident,
  onClose,
  onRefresh,
}) => {
  const [closing, setClosing] = useState(false);
  const [outcome, setOutcome] = useState('RESOLVED');
  const [notes, setNotes] = useState('');

  if (!incident) return null;

  const handleCloseCase = async () => {
    try {
      setClosing(true);
      await apiCloseIncident(incident.incident_id, outcome, notes);
      if (onRefresh) onRefresh();
      onClose();
    } catch (err: any) {
      alert(`Error closing incident: ${err.message}`);
    } finally {
      setClosing(false);
    }
  };

  const evidenceList = Array.isArray(incident.evidence_types)
    ? incident.evidence_types
    : typeof incident.evidence_types === 'string'
    ? JSON.parse(incident.evidence_types || '[]')
    : [];

  const timelineList = incident.timeline || [];
  const dispatchesList = incident.dispatches || [];

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.75)',
        backdropFilter: 'blur(4px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 100,
        padding: '20px',
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: '100%',
          maxWidth: '850px',
          maxHeight: '90vh',
          backgroundColor: '#151917',
          border: '1px solid #303832',
          borderRadius: '16px',
          overflowY: 'auto',
          boxShadow: '0 20px 40px rgba(0, 0, 0, 0.8)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          style={{
            padding: '18px 24px',
            borderBottom: '1px solid #303832',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            backgroundColor: '#111512',
            position: 'sticky',
            top: 0,
            zIndex: 10,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span
              style={{
                fontFamily: 'monospace',
                fontSize: '18px',
                fontWeight: 700,
                color: '#F2A23A',
              }}
            >
              {incident.incident_id}
            </span>
            <Badge
              variant={
                incident.severity === 'CRITICAL'
                  ? 'danger'
                  : incident.severity === 'HIGH'
                  ? 'amber'
                  : 'warning'
              }
            >
              {incident.severity}
            </Badge>
            <Badge
              variant={
                incident.status === 'CLOSED'
                  ? 'success'
                  : incident.status === 'DETECTED'
                  ? 'danger'
                  : 'amber'
              }
            >
              {incident.status}
            </Badge>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: '#A3ACA5',
              cursor: 'pointer',
              padding: '6px',
            }}
          >
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Top Banner Details */}
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: 700, margin: '0 0 6px', color: '#F2F0E9' }}>
              {incident.label}
            </h2>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', fontSize: '12px', color: '#A3ACA5' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <MapPin size={14} color="#F2A23A" />
                {incident.location_name || `${incident.latitude.toFixed(4)}, ${incident.longitude.toFixed(4)}`}
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Camera size={14} color="#55A77B" />
                Camera: <b>{incident.camera_id}</b>
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Clock size={14} color="#A3ACA5" />
                {incident.detected_at || incident.created_at}
              </span>
            </div>
          </div>

          {/* Grid Layout: Snapshot + AI Verdict */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }} className="modal-grid">
            {/* Snapshot */}
            <Card style={{ backgroundColor: '#111512' }}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: '#A3ACA5', marginBottom: '10px' }}>
                INCIDENT SCENE SNAPSHOT
              </div>
              <div
                style={{
                  height: '200px',
                  backgroundColor: '#090B0A',
                  borderRadius: '8px',
                  border: '1px solid #303832',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  overflow: 'hidden',
                }}
              >
                {incident.snapshot_path ? (
                  <img
                    src={incident.snapshot_path}
                    alt="Scene Snapshot"
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                    onError={(e) => {
                      (e.target as any).style.display = 'none';
                    }}
                  />
                ) : (
                  <div style={{ textAlign: 'center', color: '#A3ACA5' }}>
                    <Camera size={32} style={{ opacity: 0.4, marginBottom: '6px' }} />
                    <div style={{ fontSize: '12px' }}>Annotated scene frame pending</div>
                  </div>
                )}
              </div>
            </Card>

            {/* AI Physical Evidence & Detection */}
            <Card style={{ backgroundColor: '#111512' }}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: '#A3ACA5', marginBottom: '10px' }}>
                AI PHYSICAL EVIDENCE
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div>
                  <div style={{ fontSize: '11px', color: '#A3ACA5', marginBottom: '4px' }}>
                    DETECTION CONFIDENCE ({Math.round(incident.confidence * 100)}%)
                  </div>
                  <div
                    style={{
                      height: '8px',
                      backgroundColor: '#1D2420',
                      borderRadius: '4px',
                      overflow: 'hidden',
                    }}
                  >
                    <div
                      style={{
                        width: `${incident.confidence * 100}%`,
                        height: '100%',
                        backgroundColor: '#F2A23A',
                      }}
                    />
                  </div>
                </div>

                <div>
                  <div style={{ fontSize: '11px', color: '#A3ACA5', marginBottom: '6px' }}>
                    PHYSICAL EVIDENCE TAGS
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                    {evidenceList.map((tag: string, i: number) => (
                      <Badge key={i} variant="amber">
                        {tag}
                      </Badge>
                    ))}
                    {evidenceList.length === 0 && (
                      <span style={{ fontSize: '11px', color: '#A3ACA5' }}>No explicit evidence tags</span>
                    )}
                  </div>
                </div>

                {incident.reason && (
                  <div
                    style={{
                      backgroundColor: '#1D2420',
                      border: '1px solid #303832',
                      borderRadius: '6px',
                      padding: '8px 12px',
                      fontSize: '12px',
                      color: '#F2F0E9',
                    }}
                  >
                    <b>Reason:</b> {incident.reason}
                  </div>
                )}
              </div>
            </Card>
          </div>

          {/* Dispatch & Responder Status */}
          <Card style={{ backgroundColor: '#111512' }}>
            <div
              style={{
                fontSize: '12px',
                fontWeight: 600,
                color: '#A3ACA5',
                marginBottom: '10px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}
            >
              <Truck size={14} color="#55A77B" />
              EMERGENCY RESPONDER DISPATCH ({dispatchesList.length})
            </div>
            {dispatchesList.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {dispatchesList.map((d, idx) => (
                  <div
                    key={idx}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '10px 14px',
                      backgroundColor: '#151917',
                      border: '1px solid #303832',
                      borderRadius: '8px',
                    }}
                  >
                    <div>
                      <div style={{ fontSize: '13px', fontWeight: 600, color: '#F2F0E9' }}>
                        {d.name || d.responder_id} ({d.responder_type || 'AMBULANCE'})
                      </div>
                      <div style={{ fontSize: '11px', color: '#A3ACA5' }}>
                        Distance: <b>{d.distance_km ? `${d.distance_km} km` : 'Near'}</b> • ETA:{' '}
                        <b>{d.eta_minutes ? `${d.eta_minutes} min` : '5 min'}</b>
                      </div>
                    </div>
                    <Badge variant={d.status === 'ACCEPTED' ? 'success' : d.status === 'SENT' ? 'amber' : 'neutral'}>
                      {d.status}
                    </Badge>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ fontSize: '12px', color: '#A3ACA5' }}>
                No responders currently alerted or assigned.
              </div>
            )}
          </Card>

          {/* Timeline of Events */}
          <Card style={{ backgroundColor: '#111512' }}>
            <div style={{ fontSize: '12px', fontWeight: 600, color: '#A3ACA5', marginBottom: '12px' }}>
              INCIDENT RESPONSE TIMELINE
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {timelineList.map((evt, idx) => (
                <div
                  key={idx}
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '12px',
                    padding: '8px 0',
                    borderBottom: idx < timelineList.length - 1 ? '1px solid #202722' : 'none',
                  }}
                >
                  <div style={{ marginTop: '2px' }}>
                    <CheckCircle2 size={14} color="#55A77B" />
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: '13px', fontWeight: 500, color: '#F2F0E9' }}>{evt.message}</div>
                    <div style={{ fontSize: '11px', color: '#A3ACA5' }}>
                      {evt.created_at} • Actor: {evt.actor || 'System'}
                    </div>
                  </div>
                </div>
              ))}
              {timelineList.length === 0 && (
                <div style={{ fontSize: '12px', color: '#A3ACA5' }}>No events recorded yet.</div>
              )}
            </div>
          </Card>

          {/* Admin Closure Control */}
          {incident.status !== 'CLOSED' && (
            <Card style={{ backgroundColor: '#1D2420', border: '1px solid #F2A23A' }}>
              <div style={{ fontSize: '13px', fontWeight: 700, color: '#F2A23A', marginBottom: '10px' }}>
                OPERATIONAL CLOSE / STAND DOWN
              </div>
              <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
                <select
                  value={outcome}
                  onChange={(e) => setOutcome(e.target.value)}
                  style={{
                    backgroundColor: '#111512',
                    border: '1px solid #303832',
                    color: '#F2F0E9',
                    borderRadius: '8px',
                    padding: '8px 12px',
                    fontSize: '13px',
                  }}
                >
                  <option value="RESOLVED">Resolved (Scene Cleared)</option>
                  <option value="PATIENT_DELIVERED">Patient Delivered to Hospital</option>
                  <option value="FALSE_ALARM">False Alarm (Dismiss)</option>
                  <option value="CANCELLED">Cancelled</option>
                </select>

                <input
                  type="text"
                  placeholder="Closure notes / handover log..."
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  style={{
                    flex: 1,
                    minWidth: '200px',
                    backgroundColor: '#111512',
                    border: '1px solid #303832',
                    color: '#F2F0E9',
                    borderRadius: '8px',
                    padding: '8px 12px',
                    fontSize: '13px',
                  }}
                />

                <Button variant="danger" disabled={closing} onClick={handleCloseCase}>
                  {closing ? 'Closing...' : 'Close Incident'}
                </Button>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};
