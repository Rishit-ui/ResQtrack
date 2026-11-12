import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { useRealtime } from '../context/RealtimeContext';
import { Card, Button, Badge, Input, ElevatedCard } from '../components/ui';
import { MapView } from '../components/MapView';
import { IncidentDetailModal } from '../components/IncidentDetailModal';
import {
  apiListIncidents,
  apiListResponders,
  apiGetStats,
  apiGetSystemHealth,
  apiAdminListUsers,
  apiAdminProvisionUser,
  apiAdminUpdateUserStatus,
  apiAdminListReports,
  apiAdminReviewReport,
  apiAdminAuditLogs,
} from '../api';
import { Incident, Responder, ManualReport, User, AuditLog, SystemStats, Facility } from '../types';
import {
  Activity,
  AlertTriangle,
  Camera,
  Map,
  Truck,
  Users,
  FileCheck,
  Hospital,
  BarChart3,
  ScrollText,
  HeartPulse,
  Settings,
  Eye,
  CheckCircle,
  XCircle,
  ArrowUpRight,
  RefreshCw,
  Search,
} from 'lucide-react';

export const AdminDashboard: React.FC = () => {
  const { user } = useAuth();
  const { lastEvent, status: wsStatus } = useRealtime();

  const [activeTab, setActiveTab] = useState<
    'overview' | 'monitoring' | 'incidents' | 'history' | 'map' | 'responders' | 'users' | 'reports' | 'hospitals' | 'analytics' | 'audit' | 'health' | 'settings'
  >('overview');

  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [responders, setResponders] = useState<Responder[]>([]);
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [reports, setReports] = useState<ManualReport[]>([]);
  const [usersList, setUsersList] = useState<User[]>([]);
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([]);
  const [healthData, setHealthData] = useState<any>(null);

  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState<string>('');

  // Provisioning form
  const [newEmail, setNewEmail] = useState('');
  const [newPass, setNewPass] = useState('');
  const [newName, setNewName] = useState('');
  const [newRole, setNewRole] = useState('RESPONDER');
  const [newPhone, setNewPhone] = useState('');
  const [newRespId, setNewRespId] = useState('');
  const [provisionMsg, setProvisionMsg] = useState<string | null>(null);

  const loadAllData = async () => {
    try {
      const [incs, resps, st, rep, usrs, logs, hlth] = await Promise.all([
        apiListIncidents(undefined, false, 100),
        apiListResponders(),
        apiGetStats(),
        apiAdminListReports(),
        apiAdminListUsers(),
        apiAdminAuditLogs(50),
        apiGetSystemHealth(),
      ]);
      setIncidents(incs);
      setResponders(resps);
      setStats(st);
      setReports(rep);
      setUsersList(usrs);
      setAuditLogs(logs);
      setHealthData(hlth);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    loadAllData();
  }, []);

  // Real-time event handler
  useEffect(() => {
    if (!lastEvent) return;
    if (
      lastEvent.type.startsWith('incident.') ||
      lastEvent.type.startsWith('responder.') ||
      lastEvent.type.startsWith('report.')
    ) {
      loadAllData();
    }
  }, [lastEvent]);

  const handleReviewReport = async (reportId: string, action: 'APPROVED' | 'REJECTED' | 'ESCALATED') => {
    const notes = prompt(`Enter review notes for report ${reportId}:`) || 'Control room review';
    try {
      await apiAdminReviewReport(reportId, action, notes);
      await loadAllData();
      alert(`Report ${reportId} marked as ${action}`);
    } catch (err: any) {
      alert(`Error reviewing report: ${err.message}`);
    }
  };

  const handleToggleUserStatus = async (targetUser: User) => {
    const nextStatus = targetUser.status === 'ACTIVE' ? 'SUSPENDED' : 'ACTIVE';
    if (confirm(`Change status of ${targetUser.email} to ${nextStatus}?`)) {
      try {
        await apiAdminUpdateUserStatus(targetUser.user_id, nextStatus);
        await loadAllData();
      } catch (err: any) {
        alert(`Error: ${err.message}`);
      }
    }
  };

  const handleProvisionUser = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await apiAdminProvisionUser({
        email: newEmail,
        password: newPass,
        full_name: newName,
        role: newRole,
        phone: newPhone,
        responder_id: newRole === 'RESPONDER' ? newRespId || undefined : undefined,
      });
      setProvisionMsg(`Successfully provisioned ${newRole} account for ${newEmail}`);
      setNewEmail('');
      setNewPass('');
      setNewName('');
      setNewPhone('');
      setNewRespId('');
      await loadAllData();
    } catch (err: any) {
      alert(`Error provisioning user: ${err.message}`);
    }
  };

  const activeIncidents = incidents.filter(
    (i) => i.status !== 'CLOSED' && i.status !== 'CANCELLED' && i.status !== 'FALSE_ALARM'
  );
  const closedIncidents = incidents.filter(
    (i) => i.status === 'CLOSED' || i.status === 'CANCELLED' || i.status === 'FALSE_ALARM'
  );
  const availableResponders = responders.filter((r) => r.status === 'AVAILABLE');

  const filteredIncidents = incidents.filter((i) => {
    const matchesStatus = !statusFilter || i.status === statusFilter;
    const matchesSearch =
      !searchQuery ||
      i.incident_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      i.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (i.location_name && i.location_name.toLowerCase().includes(searchQuery.toLowerCase()));
    return matchesStatus && matchesSearch;
  });

  return (
    <div style={{ display: 'flex', minHeight: 'calc(100vh - 70px)', backgroundColor: '#090B0A' }}>
      {/* Sidebar Navigation */}
      <aside
        style={{
          width: '240px',
          backgroundColor: '#111512',
          borderRight: '1px solid #303832',
          padding: '20px 10px',
          display: 'flex',
          flexDirection: 'column',
          gap: '4px',
        }}
        className="admin-sidebar"
      >
        <div style={{ padding: '0 12px 14px', fontSize: '11px', color: '#A3ACA5', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.8px' }}>
          COMMAND CONSOLE
        </div>

        {[
          { id: 'overview', label: 'Overview', icon: <Activity size={16} /> },
          { id: 'monitoring', label: 'Live Monitoring', icon: <Camera size={16} /> },
          { id: 'incidents', label: `Active Incidents (${activeIncidents.length})`, icon: <AlertTriangle size={16} /> },
          { id: 'history', label: 'Incident History', icon: <ScrollText size={16} /> },
          { id: 'map', label: 'Tactical Map', icon: <Map size={16} /> },
          { id: 'responders', label: `Responders (${responders.length})`, icon: <Truck size={16} /> },
          { id: 'reports', label: `Citizen Reports (${reports.filter((r) => r.status === 'SUBMITTED').length})`, icon: <FileCheck size={16} /> },
          { id: 'users', label: 'User Accounts', icon: <Users size={16} /> },
          { id: 'analytics', label: 'Analytics', icon: <BarChart3 size={16} /> },
          { id: 'audit', label: 'Audit Trail', icon: <ScrollText size={16} /> },
          { id: 'health', label: 'System Health', icon: <HeartPulse size={16} /> },
        ].map((item) => (
          <button
            key={item.id}
            onClick={() => setActiveTab(item.id as any)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              width: '100%',
              padding: '9px 12px',
              borderRadius: '8px',
              border: 'none',
              backgroundColor: activeTab === item.id ? '#1D2420' : 'transparent',
              color: activeTab === item.id ? '#F2A23A' : '#A3ACA5',
              fontSize: '13px',
              fontWeight: 600,
              cursor: 'pointer',
              textAlign: 'left',
              transition: 'all 0.15s ease',
            }}
          >
            {item.icon}
            <span>{item.label}</span>
          </button>
        ))}

        <div style={{ marginTop: 'auto', padding: '12px', borderTop: '1px solid #303832' }}>
          <div style={{ fontSize: '11px', color: '#A3ACA5' }}>OPERATOR:</div>
          <div style={{ fontSize: '12px', fontWeight: 600, color: '#F2F0E9', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {user?.full_name}
          </div>
          <div style={{ fontSize: '10px', color: '#55A77B', marginTop: '2px' }}>● AUTHORIZED ADMIN</div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main style={{ flex: 1, padding: '24px', overflowY: 'auto', maxWidth: 'calc(100% - 240px)' }}>
        {/* TAB: OVERVIEW */}
        {activeTab === 'overview' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <div>
                <h1 style={{ fontSize: '24px', fontWeight: 800, margin: '0 0 4px', color: '#F2F0E9' }}>
                  Emergency Operations Command
                </h1>
                <div style={{ fontSize: '12px', color: '#A3ACA5' }}>
                  Real-time database metrics • Telemetry active • WebSocket {wsStatus.toUpperCase()}
                </div>
              </div>
              <Button variant="outline" onClick={loadAllData}>
                <RefreshCw size={14} /> Refresh Data
              </Button>
            </div>

            {/* Metrics Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '24px' }}>
              <Card>
                <div style={{ fontSize: '11px', color: '#A3ACA5', textTransform: 'uppercase', fontWeight: 600 }}>Total Incidents</div>
                <div style={{ fontSize: '32px', fontWeight: 800, color: '#F2F0E9', margin: '4px 0' }}>{incidents.length}</div>
                <div style={{ fontSize: '11px', color: '#A3ACA5' }}>Recorded cases</div>
              </Card>

              <Card>
                <div style={{ fontSize: '11px', color: '#A3ACA5', textTransform: 'uppercase', fontWeight: 600 }}>Active / In Progress</div>
                <div style={{ fontSize: '32px', fontWeight: 800, color: activeIncidents.length > 0 ? '#D95757' : '#55A77B', margin: '4px 0' }}>
                  {activeIncidents.length}
                </div>
                <div style={{ fontSize: '11px', color: '#A3ACA5' }}>Live emergency scenes</div>
              </Card>

              <Card>
                <div style={{ fontSize: '11px', color: '#A3ACA5', textTransform: 'uppercase', fontWeight: 600 }}>Available Responders</div>
                <div style={{ fontSize: '32px', fontWeight: 800, color: '#55A77B', margin: '4px 0' }}>
                  {availableResponders.length} / {responders.length}
                </div>
                <div style={{ fontSize: '11px', color: '#A3ACA5' }}>Ambulance & rescue units</div>
              </Card>

              <Card>
                <div style={{ fontSize: '11px', color: '#A3ACA5', textTransform: 'uppercase', fontWeight: 600 }}>Citizen Reports Awaiting</div>
                <div style={{ fontSize: '32px', fontWeight: 800, color: '#F2A23A', margin: '4px 0' }}>
                  {reports.filter((r) => r.status === 'SUBMITTED').length}
                </div>
                <div style={{ fontSize: '11px', color: '#F2A23A' }}>Unverified queue</div>
              </Card>
            </div>

            {/* Map & Live Feed Layout */}
            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px', marginBottom: '24px' }} className="overview-grid">
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                  <div style={{ fontSize: '13px', fontWeight: 700, color: '#F2F0E9', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    TACTICAL INCIDENT & UNIT LOCATIONS
                  </div>
                  <Button variant="outline" style={{ padding: '4px 8px', fontSize: '11px' }} onClick={() => setActiveTab('map')}>
                    Full Map View
                  </Button>
                </div>
                <MapView incidents={activeIncidents} responders={responders} height="360px" onSelectIncident={(inc) => setSelectedIncident(inc)} />
              </div>

              {/* Active alerts panel */}
              <div>
                <div style={{ fontSize: '13px', fontWeight: 700, color: '#F2F0E9', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '10px' }}>
                  ACTIVE EMERGENCY ROSTER
                </div>
                <Card style={{ height: '360px', overflowY: 'auto', padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {activeIncidents.map((inc) => (
                    <div
                      key={inc.incident_id}
                      onClick={() => setSelectedIncident(inc)}
                      style={{
                        padding: '10px',
                        borderRadius: '8px',
                        backgroundColor: '#111512',
                        borderLeft: `4px solid ${inc.severity === 'CRITICAL' ? '#D95757' : '#F2A23A'}`,
                        border: '1px solid #303832',
                        cursor: 'pointer',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                        <span style={{ fontFamily: 'monospace', fontWeight: 700, color: '#F2A23A', fontSize: '12px' }}>
                          {inc.incident_id}
                        </span>
                        <Badge variant={inc.severity === 'CRITICAL' ? 'danger' : 'amber'}>{inc.severity}</Badge>
                      </div>
                      <div style={{ fontSize: '13px', fontWeight: 600, color: '#F2F0E9' }}>{inc.label}</div>
                      <div style={{ fontSize: '11px', color: '#A3ACA5', marginTop: '4px' }}>
                        Camera: {inc.camera_id} • Status: <b>{inc.status}</b>
                      </div>
                    </div>
                  ))}
                  {activeIncidents.length === 0 && (
                    <div style={{ textAlign: 'center', padding: '40px 10px', color: '#A3ACA5', fontSize: '13px' }}>
                      No active emergency calls at this time.
                    </div>
                  )}
                </Card>
              </div>
            </div>
          </div>
        )}

        {/* TAB: LIVE MONITORING */}
        {activeTab === 'monitoring' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <div>
                <h1 style={{ fontSize: '24px', fontWeight: 800, margin: '0 0 4px', color: '#F2F0E9' }}>
                  Live Camera Monitoring Stream
                </h1>
                <div style={{ fontSize: '12px', color: '#A3ACA5' }}>
                  Continuous video ingestion • YOLO11 vehicle tracking & anomaly overlay
                </div>
              </div>
              <Badge variant="amber">DETECTOR STREAM PORT: 8001</Badge>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px' }} className="overview-grid">
              <Card style={{ padding: '16px' }}>
                <div
                  style={{
                    width: '100%',
                    height: '480px',
                    backgroundColor: '#090B0A',
                    border: '1px solid #303832',
                    borderRadius: '10px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    overflow: 'hidden',
                    position: 'relative',
                  }}
                >
                  <img
                    src="http://localhost:8001/stream"
                    alt="Live Video Stream"
                    style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                    onError={(e) => {
                      (e.target as any).style.display = 'none';
                      const fallback = document.getElementById('camera-fallback');
                      if (fallback) fallback.style.display = 'flex';
                    }}
                  />
                  <div
                    id="camera-fallback"
                    style={{
                      display: 'none',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: '#A3ACA5',
                      textAlign: 'center',
                      padding: '20px',
                    }}
                  >
                    <Camera size={48} color="#F2A23A" style={{ opacity: 0.6, marginBottom: '12px' }} />
                    <div style={{ fontSize: '16px', fontWeight: 600, color: '#F2F0E9' }}>
                      Direct Video Stream Server (Port 8001) Standby
                    </div>
                    <p style={{ fontSize: '13px', maxWidth: '400px', margin: '8px 0 16px' }}>
                      To stream live video: launch the AI detector process with{' '}
                      <code style={{ color: '#F2A23A' }}>python main.py --source data/test.mp4 --loop</code>.
                    </p>
                    <Badge variant="neutral">HUD HUD OVERLAY ENGINE READY</Badge>
                  </div>
                </div>
              </Card>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <Card>
                  <div style={{ fontSize: '13px', fontWeight: 700, color: '#F2F0E9', marginBottom: '12px' }}>
                    DETECTOR HUD TELEMETRY
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: '#A3ACA5' }}>Detector Architecture:</span>
                      <b style={{ color: '#F2A23A' }}>YOLO11 Nano + ByteTrack</b>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: '#A3ACA5' }}>Temporal Classifier:</span>
                      <b>v5 Multi-Window (PKL)</b>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: '#A3ACA5' }}>Kinematics Scale:</span>
                      <b>Auto m/px (Vehicle-norm)</b>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: '#A3ACA5' }}>Rehearsal Shortcut:</span>
                      <kbd style={{ backgroundColor: '#1D2420', padding: '2px 6px', borderRadius: '4px' }}>T key in window</kbd>
                    </div>
                  </div>
                </Card>

                <Card>
                  <div style={{ fontSize: '13px', fontWeight: 700, color: '#F2F0E9', marginBottom: '10px' }}>
                    RECENT DETECTIONS
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '240px', overflowY: 'auto' }}>
                    {incidents.slice(0, 5).map((inc) => (
                      <div
                        key={inc.incident_id}
                        onClick={() => setSelectedIncident(inc)}
                        style={{
                          padding: '8px 10px',
                          borderRadius: '6px',
                          backgroundColor: '#111512',
                          cursor: 'pointer',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                        }}
                      >
                        <div>
                          <div style={{ fontSize: '12px', fontWeight: 600, color: '#F2F0E9' }}>{inc.incident_id}</div>
                          <div style={{ fontSize: '11px', color: '#A3ACA5' }}>{inc.label}</div>
                        </div>
                        <Badge variant={inc.severity === 'CRITICAL' ? 'danger' : 'amber'}>{inc.severity}</Badge>
                      </div>
                    ))}
                  </div>
                </Card>
              </div>
            </div>
          </div>
        )}

        {/* TAB: ACTIVE INCIDENTS */}
        {activeTab === 'incidents' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <div>
                <h1 style={{ fontSize: '24px', fontWeight: 800, margin: '0 0 4px', color: '#F2F0E9' }}>
                  Live Active Emergency Incidents
                </h1>
                <div style={{ fontSize: '12px', color: '#A3ACA5' }}>
                  All confirmed incidents requiring dispatch or in transport
                </div>
              </div>
              <div style={{ display: 'flex', gap: '10px' }}>
                <Input
                  type="text"
                  placeholder="Search by ID or location..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  style={{ width: '220px' }}
                />
              </div>
            </div>

            <Card style={{ padding: 0, overflow: 'hidden' }}>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
                  <thead>
                    <tr style={{ backgroundColor: '#111512', borderBottom: '1px solid #303832', color: '#A3ACA5', fontSize: '11px', textTransform: 'uppercase' }}>
                      <th style={{ padding: '12px 16px' }}>Incident ID</th>
                      <th style={{ padding: '12px 16px' }}>Severity</th>
                      <th style={{ padding: '12px 16px' }}>Description / Label</th>
                      <th style={{ padding: '12px 16px' }}>Location</th>
                      <th style={{ padding: '12px 16px' }}>Camera</th>
                      <th style={{ padding: '12px 16px' }}>Status</th>
                      <th style={{ padding: '12px 16px' }}>Assigned Unit</th>
                      <th style={{ padding: '12px 16px' }}>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {activeIncidents.map((inc) => (
                      <tr key={inc.incident_id} style={{ borderBottom: '1px solid #202722' }}>
                        <td style={{ padding: '12px 16px', fontFamily: 'monospace', fontWeight: 700, color: '#F2A23A' }}>
                          {inc.incident_id}
                        </td>
                        <td style={{ padding: '12px 16px' }}>
                          <Badge variant={inc.severity === 'CRITICAL' ? 'danger' : 'amber'}>{inc.severity}</Badge>
                        </td>
                        <td style={{ padding: '12px 16px', fontWeight: 600, color: '#F2F0E9' }}>{inc.label}</td>
                        <td style={{ padding: '12px 16px', color: '#A3ACA5' }}>{inc.location_name || `${inc.latitude.toFixed(4)}, ${inc.longitude.toFixed(4)}`}</td>
                        <td style={{ padding: '12px 16px', color: '#A3ACA5' }}>{inc.camera_id}</td>
                        <td style={{ padding: '12px 16px' }}>
                          <Badge variant={inc.status === 'DETECTED' ? 'danger' : 'amber'}>{inc.status}</Badge>
                        </td>
                        <td style={{ padding: '12px 16px', color: inc.assigned_responder ? '#55A77B' : '#A3ACA5', fontWeight: 600 }}>
                          {inc.assigned_responder || 'Dispatching...'}
                        </td>
                        <td style={{ padding: '12px 16px' }}>
                          <Button variant="outline" style={{ padding: '5px 10px', fontSize: '12px' }} onClick={() => setSelectedIncident(inc)}>
                            Inspect
                          </Button>
                        </td>
                      </tr>
                    ))}
                    {activeIncidents.length === 0 && (
                      <tr>
                        <td colSpan={8} style={{ textAlign: 'center', padding: '30px', color: '#A3ACA5' }}>
                          No active emergency incidents currently open.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        )}

        {/* TAB: INCIDENT HISTORY */}
        {activeTab === 'history' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <div>
                <h1 style={{ fontSize: '24px', fontWeight: 800, margin: '0 0 4px', color: '#F2F0E9' }}>
                  Closed Incident Case Archives
                </h1>
                <div style={{ fontSize: '12px', color: '#A3ACA5' }}>
                  Full response timing, outcomes, and case completion audits
                </div>
              </div>
            </div>

            <Card style={{ padding: 0, overflow: 'hidden' }}>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
                  <thead>
                    <tr style={{ backgroundColor: '#111512', borderBottom: '1px solid #303832', color: '#A3ACA5', fontSize: '11px', textTransform: 'uppercase' }}>
                      <th style={{ padding: '12px 16px' }}>Incident ID</th>
                      <th style={{ padding: '12px 16px' }}>Outcome</th>
                      <th style={{ padding: '12px 16px' }}>Label</th>
                      <th style={{ padding: '12px 16px' }}>Location</th>
                      <th style={{ padding: '12px 16px' }}>Assigned Unit</th>
                      <th style={{ padding: '12px 16px' }}>Hospital Handover</th>
                      <th style={{ padding: '12px 16px' }}>Closed At</th>
                      <th style={{ padding: '12px 16px' }}>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {closedIncidents.map((inc) => (
                      <tr key={inc.incident_id} style={{ borderBottom: '1px solid #202722' }}>
                        <td style={{ padding: '12px 16px', fontFamily: 'monospace', fontWeight: 700, color: '#F2A23A' }}>
                          {inc.incident_id}
                        </td>
                        <td style={{ padding: '12px 16px' }}>
                          <Badge variant={inc.outcome === 'RESOLVED' ? 'success' : 'neutral'}>
                            {inc.outcome || inc.status}
                          </Badge>
                        </td>
                        <td style={{ padding: '12px 16px', color: '#F2F0E9' }}>{inc.label}</td>
                        <td style={{ padding: '12px 16px', color: '#A3ACA5' }}>{inc.location_name || 'Scene'}</td>
                        <td style={{ padding: '12px 16px', color: '#A3ACA5' }}>{inc.assigned_responder || 'N/A'}</td>
                        <td style={{ padding: '12px 16px', color: '#55A77B' }}>{inc.hospital_name || 'None'}</td>
                        <td style={{ padding: '12px 16px', color: '#A3ACA5' }}>{inc.closed_at || 'Archived'}</td>
                        <td style={{ padding: '12px 16px' }}>
                          <Button variant="outline" style={{ padding: '5px 10px', fontSize: '12px' }} onClick={() => setSelectedIncident(inc)}>
                            Review
                          </Button>
                        </td>
                      </tr>
                    ))}
                    {closedIncidents.length === 0 && (
                      <tr>
                        <td colSpan={8} style={{ textAlign: 'center', padding: '30px', color: '#A3ACA5' }}>
                          No closed incidents found in archives.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        )}

        {/* TAB: TACTICAL MAP */}
        {activeTab === 'map' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h1 style={{ fontSize: '24px', fontWeight: 800, margin: 0, color: '#F2F0E9' }}>
                Full-Screen Tactical Interactive Map
              </h1>
              <div style={{ display: 'flex', gap: '10px' }}>
                <Badge variant="amber">● Live Incident</Badge>
                <Badge variant="success">🚑 Responder Unit</Badge>
                <Badge variant="neutral">🏥 Hospital Facility</Badge>
              </div>
            </div>
            <MapView
              incidents={activeIncidents}
              responders={responders}
              height="680px"
              onSelectIncident={(inc) => setSelectedIncident(inc)}
            />
          </div>
        )}

        {/* TAB: RESPONDERS */}
        {activeTab === 'responders' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <div>
                <h1 style={{ fontSize: '24px', fontWeight: 800, margin: '0 0 4px', color: '#F2F0E9' }}>
                  Emergency Fleet Responders
                </h1>
                <div style={{ fontSize: '12px', color: '#A3ACA5' }}>
                  Registered ambulance, paramedic bike, and rescue vehicles
                </div>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
              {responders.map((resp) => (
                <Card key={resp.responder_id}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <span style={{ fontFamily: 'monospace', fontWeight: 700, color: '#F2A23A', fontSize: '13px' }}>
                      {resp.responder_id}
                    </span>
                    <Badge variant={resp.status === 'AVAILABLE' ? 'success' : 'amber'}>{resp.status}</Badge>
                  </div>
                  <div style={{ fontSize: '16px', fontWeight: 700, color: '#F2F0E9', marginBottom: '6px' }}>
                    {resp.name}
                  </div>
                  <div style={{ fontSize: '12px', color: '#A3ACA5', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <div>Vehicle: <b>{resp.vehicle_number || 'N/A'}</b> • Type: <b>{resp.responder_type}</b></div>
                    <div>Phone: <b>{resp.phone || 'N/A'}</b></div>
                    <div>Location: {resp.latitude.toFixed(4)}, {resp.longitude.toFixed(4)}</div>
                    {resp.active_incident && (
                      <div style={{ color: '#D95757', fontWeight: 600 }}>Active Call: {resp.active_incident}</div>
                    )}
                  </div>
                </Card>
              ))}
            </div>
          </div>
        )}

        {/* TAB: MANUAL CITIZEN REPORTS REVIEW */}
        {activeTab === 'reports' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <div>
                <h1 style={{ fontSize: '24px', fontWeight: 800, margin: '0 0 4px', color: '#F2F0E9' }}>
                  Citizen Incident Reports Review Queue
                </h1>
                <div style={{ fontSize: '12px', color: '#A3ACA5' }}>
                  Review, approve, dismiss, or escalate reports to active emergency incidents
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {reports.map((rep) => (
                <Card key={rep.report_id} style={{ padding: '20px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px' }}>
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px' }}>
                        <span style={{ fontFamily: 'monospace', fontWeight: 700, color: '#F2A23A', fontSize: '14px' }}>
                          {rep.report_id}
                        </span>
                        <span style={{ fontSize: '15px', fontWeight: 700, color: '#F2F0E9' }}>{rep.category}</span>
                        <Badge
                          variant={
                            rep.status === 'ESCALATED' ? 'success' : rep.status === 'SUBMITTED' ? 'amber' : 'neutral'
                          }
                        >
                          {rep.status}
                        </Badge>
                      </div>

                      <p style={{ fontSize: '13px', color: '#F2F0E9', margin: '6px 0 10px', maxWidth: '700px' }}>
                        {rep.description}
                      </p>

                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', fontSize: '12px', color: '#A3ACA5' }}>
                        <span>Reporter: <b>{rep.reporter_name || rep.user_id}</b></span>
                        <span>Location: <b>{rep.location_name || 'N/A'}</b></span>
                        <span>Contact: <b>{rep.contact_phone || 'None'}</b></span>
                        <span>Submitted: {rep.created_at}</span>
                      </div>

                      {rep.linked_incident_id && (
                        <div style={{ marginTop: '10px', color: '#55A77B', fontWeight: 600, fontSize: '13px' }}>
                          Escalated to Confirmed Emergency: Incident <b>{rep.linked_incident_id}</b>
                        </div>
                      )}
                    </div>

                    {rep.status === 'SUBMITTED' && (
                      <div style={{ display: 'flex', gap: '8px' }}>
                        <Button
                          variant="primary"
                          onClick={() => handleReviewReport(rep.report_id, 'ESCALATED')}
                        >
                          <ArrowUpRight size={14} /> Escalate & Dispatch
                        </Button>
                        <Button
                          variant="secondary"
                          onClick={() => handleReviewReport(rep.report_id, 'APPROVED')}
                        >
                          <CheckCircle size={14} /> Approve
                        </Button>
                        <Button
                          variant="danger"
                          onClick={() => handleReviewReport(rep.report_id, 'REJECTED')}
                        >
                          <XCircle size={14} /> Reject
                        </Button>
                      </div>
                    )}
                  </div>
                </Card>
              ))}
              {reports.length === 0 && (
                <Card style={{ textAlign: 'center', padding: '40px', color: '#A3ACA5' }}>
                  No citizen reports in queue.
                </Card>
              )}
            </div>
          </div>
        )}

        {/* TAB: USER MANAGEMENT */}
        {activeTab === 'users' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <div>
                <h1 style={{ fontSize: '24px', fontWeight: 800, margin: '0 0 4px', color: '#F2F0E9' }}>
                  User Management & Provisioning
                </h1>
                <div style={{ fontSize: '12px', color: '#A3ACA5' }}>
                  Administrative bootstrap, role management, and status controls
                </div>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '20px' }} className="overview-grid">
              {/* Provision Form */}
              <Card style={{ padding: '20px' }}>
                <h2 style={{ fontSize: '15px', fontWeight: 700, margin: '0 0 14px', color: '#F2A23A' }}>
                  Provision New Account
                </h2>
                {provisionMsg && (
                  <div style={{ padding: '8px 12px', borderRadius: '6px', backgroundColor: 'rgba(85, 167, 123, 0.15)', color: '#55A77B', fontSize: '12px', marginBottom: '14px' }}>
                    {provisionMsg}
                  </div>
                )}
                <form onSubmit={handleProvisionUser} style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '11px', fontWeight: 600, color: '#A3ACA5', marginBottom: '4px' }}>
                      FULL NAME
                    </label>
                    <Input required value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="Paramedic Jane" />
                  </div>

                  <div>
                    <label style={{ display: 'block', fontSize: '11px', fontWeight: 600, color: '#A3ACA5', marginBottom: '4px' }}>
                      EMAIL
                    </label>
                    <Input type="email" required value={newEmail} onChange={(e) => setNewEmail(e.target.value)} placeholder="crew@resqtrack.gov" />
                  </div>

                  <div>
                    <label style={{ display: 'block', fontSize: '11px', fontWeight: 600, color: '#A3ACA5', marginBottom: '4px' }}>
                      ROLE
                    </label>
                    <select
                      value={newRole}
                      onChange={(e) => setNewRole(e.target.value)}
                      style={{ width: '100%', backgroundColor: '#111512', border: '1px solid #303832', borderRadius: '8px', padding: '10px', color: '#F2F0E9', fontSize: '13px' }}
                    >
                      <option value="RESPONDER">RESPONDER (Ambulance Crew)</option>
                      <option value="ADMIN">ADMIN (Control Room)</option>
                      <option value="CITIZEN">CITIZEN</option>
                    </select>
                  </div>

                  {newRole === 'RESPONDER' && (
                    <div>
                      <label style={{ display: 'block', fontSize: '11px', fontWeight: 600, color: '#A3ACA5', marginBottom: '4px' }}>
                        LINKED RESPONDER ID
                      </label>
                      <Input value={newRespId} onChange={(e) => setNewRespId(e.target.value)} placeholder="RESP-0001" />
                    </div>
                  )}

                  <div>
                    <label style={{ display: 'block', fontSize: '11px', fontWeight: 600, color: '#A3ACA5', marginBottom: '4px' }}>
                      PASSWORD
                    </label>
                    <Input type="password" required value={newPass} onChange={(e) => setNewPass(e.target.value)} placeholder="••••••••••••" />
                  </div>

                  <Button type="submit" variant="primary" style={{ marginTop: '6px' }}>
                    Provision User
                  </Button>
                </form>
              </Card>

              {/* Users Table */}
              <Card style={{ padding: 0, overflow: 'hidden' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
                  <thead>
                    <tr style={{ backgroundColor: '#111512', borderBottom: '1px solid #303832', color: '#A3ACA5', fontSize: '11px', textTransform: 'uppercase' }}>
                      <th style={{ padding: '10px 14px' }}>User ID</th>
                      <th style={{ padding: '10px 14px' }}>Name / Email</th>
                      <th style={{ padding: '10px 14px' }}>Role</th>
                      <th style={{ padding: '10px 14px' }}>Status</th>
                      <th style={{ padding: '10px 14px' }}>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {usersList.map((u) => (
                      <tr key={u.user_id} style={{ borderBottom: '1px solid #202722' }}>
                        <td style={{ padding: '10px 14px', fontFamily: 'monospace', color: '#F2A23A', fontSize: '12px' }}>
                          {u.user_id}
                        </td>
                        <td style={{ padding: '10px 14px' }}>
                          <div style={{ fontWeight: 600, color: '#F2F0E9' }}>{u.full_name}</div>
                          <div style={{ fontSize: '11px', color: '#A3ACA5' }}>{u.email}</div>
                        </td>
                        <td style={{ padding: '10px 14px' }}>
                          <Badge variant={u.role === 'ADMIN' ? 'amber' : u.role === 'RESPONDER' ? 'success' : 'neutral'}>
                            {u.role}
                          </Badge>
                        </td>
                        <td style={{ padding: '10px 14px' }}>
                          <Badge variant={u.status === 'ACTIVE' ? 'success' : 'danger'}>{u.status}</Badge>
                        </td>
                        <td style={{ padding: '10px 14px' }}>
                          <Button
                            variant="outline"
                            style={{ padding: '4px 8px', fontSize: '11px' }}
                            onClick={() => handleToggleUserStatus(u)}
                          >
                            {u.status === 'ACTIVE' ? 'Suspend' : 'Activate'}
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            </div>
          </div>
        )}

        {/* TAB: AUDIT LOGS */}
        {activeTab === 'audit' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <div>
                <h1 style={{ fontSize: '24px', fontWeight: 800, margin: '0 0 4px', color: '#F2F0E9' }}>
                  Security & Operational Audit Trail
                </h1>
                <div style={{ fontSize: '12px', color: '#A3ACA5' }}>
                  Immutable log of administrative, dispatch, and authentication events
                </div>
              </div>
            </div>

            <Card style={{ padding: 0, overflow: 'hidden' }}>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
                  <thead>
                    <tr style={{ backgroundColor: '#111512', borderBottom: '1px solid #303832', color: '#A3ACA5', fontSize: '11px', textTransform: 'uppercase' }}>
                      <th style={{ padding: '10px 14px' }}>Log ID</th>
                      <th style={{ padding: '10px 14px' }}>Timestamp</th>
                      <th style={{ padding: '10px 14px' }}>Action</th>
                      <th style={{ padding: '10px 14px' }}>User / Actor</th>
                      <th style={{ padding: '10px 14px' }}>Target</th>
                      <th style={{ padding: '10px 14px' }}>IP</th>
                    </tr>
                  </thead>
                  <tbody>
                    {auditLogs.map((log) => (
                      <tr key={log.log_id} style={{ borderBottom: '1px solid #202722' }}>
                        <td style={{ padding: '10px 14px', fontFamily: 'monospace', color: '#F2A23A', fontSize: '12px' }}>
                          {log.log_id}
                        </td>
                        <td style={{ padding: '10px 14px', color: '#A3ACA5', fontSize: '12px' }}>{log.created_at}</td>
                        <td style={{ padding: '10px 14px', fontWeight: 600, color: '#F2F0E9' }}>{log.action}</td>
                        <td style={{ padding: '10px 14px', color: '#A3ACA5' }}>
                          {log.user_email || log.user_id || 'System'}
                        </td>
                        <td style={{ padding: '10px 14px', color: '#A3ACA5' }}>
                          {log.target_type ? `${log.target_type}: ${log.target_id || ''}` : 'N/A'}
                        </td>
                        <td style={{ padding: '10px 14px', color: '#7E8880', fontFamily: 'monospace' }}>
                          {log.ip_address || 'local'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        )}

        {/* TAB: SYSTEM HEALTH */}
        {activeTab === 'health' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <div>
                <h1 style={{ fontSize: '24px', fontWeight: 800, margin: '0 0 4px', color: '#F2F0E9' }}>
                  Infrastructure Health & Diagnostics
                </h1>
                <div style={{ fontSize: '12px', color: '#A3ACA5' }}>
                  Backend status, database size, WebSocket fan-out metrics
                </div>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
              <Card>
                <div style={{ fontSize: '12px', fontWeight: 700, color: '#55A77B', marginBottom: '10px' }}>
                  CORE ENGINE STATUS
                </div>
                <div style={{ fontSize: '22px', fontWeight: 800, color: '#F2F0E9', margin: '4px 0' }}>
                  {healthData?.status || 'ONLINE'}
                </div>
                <div style={{ fontSize: '12px', color: '#A3ACA5' }}>
                  Service: {healthData?.service || 'ResQTrack'} • v{healthData?.version || '3.0'}
                </div>
              </Card>

              <Card>
                <div style={{ fontSize: '12px', fontWeight: 700, color: '#F2A23A', marginBottom: '10px' }}>
                  DATABASE STORAGE
                </div>
                <div style={{ fontSize: '22px', fontWeight: 800, color: '#F2F0E9', margin: '4px 0' }}>
                  {healthData?.database?.size_bytes ? `${(healthData.database.size_bytes / 1024).toFixed(1)} KB` : 'Active'}
                </div>
                <div style={{ fontSize: '12px', color: '#A3ACA5' }}>
                  Engine: {healthData?.database?.engine || 'SQLite WAL mode'}
                </div>
              </Card>

              <Card>
                <div style={{ fontSize: '12px', fontWeight: 700, color: '#55A77B', marginBottom: '10px' }}>
                  REALTIME FAN-OUT
                </div>
                <div style={{ fontSize: '22px', fontWeight: 800, color: '#F2F0E9', margin: '4px 0' }}>
                  {healthData?.realtime?.live_websocket_connections ?? 0} Sockets
                </div>
                <div style={{ fontSize: '12px', color: '#A3ACA5' }}>
                  Broadcast ring buffer: {healthData?.realtime?.cached_events ?? 0} events
                </div>
              </Card>
            </div>
          </div>
        )}

        {/* TAB: ANALYTICS */}
        {activeTab === 'analytics' && stats && (
          <div>
            <h1 style={{ fontSize: '24px', fontWeight: 800, margin: '0 0 16px', color: '#F2F0E9' }}>
              Incident Analytics & Performance
            </h1>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '16px', marginBottom: '24px' }}>
              <Card>
                <div style={{ fontSize: '11px', color: '#A3ACA5', textTransform: 'uppercase', fontWeight: 600 }}>Avg Time to Accept</div>
                <div style={{ fontSize: '28px', fontWeight: 800, color: '#F2A23A', margin: '4px 0' }}>
                  {stats.avg_acceptance_seconds > 0 ? `${stats.avg_acceptance_seconds}s` : 'Data pending'}
                </div>
                <div style={{ fontSize: '11px', color: '#A3ACA5' }}>From confirmation to crew accept</div>
              </Card>

              <Card>
                <div style={{ fontSize: '11px', color: '#A3ACA5', textTransform: 'uppercase', fontWeight: 600 }}>Avg Transit to Scene</div>
                <div style={{ fontSize: '28px', fontWeight: 800, color: '#55A77B', margin: '4px 0' }}>
                  {stats.avg_travel_seconds > 0 ? `${stats.avg_travel_seconds}s` : 'Data pending'}
                </div>
                <div style={{ fontSize: '11px', color: '#A3ACA5' }}>From accept to geofence trigger</div>
              </Card>
            </div>
          </div>
        )}
      </main>

      {/* Incident Detail Modal */}
      {selectedIncident && (
        <IncidentDetailModal
          incident={selectedIncident}
          onClose={() => setSelectedIncident(null)}
          onRefresh={loadAllData}
        />
      )}
    </div>
  );
};
