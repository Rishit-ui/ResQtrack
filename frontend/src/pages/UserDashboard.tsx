import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { useRealtime } from '../context/RealtimeContext';
import { Card, Button, Input, Textarea, Badge } from '../components/ui';
import {
  apiSubmitReport,
  apiUploadReportImage,
  apiGetMyReports,
  apiGetReport,
  apiUpdateProfile,
  apiChangePassword,
  apiGetNotifications,
} from '../api';
import { ManualReport, NotificationItem } from '../types';
import {
  AlertTriangle,
  FileText,
  MapPin,
  Camera,
  Clock,
  CheckCircle,
  Bell,
  Settings,
  Send,
  Navigation,
  Eye,
  X,
} from 'lucide-react';

export const UserDashboard: React.FC = () => {
  const { user, refreshUser } = useAuth();
  const { lastEvent } = useRealtime();

  const [activeTab, setActiveTab] = useState<'report' | 'history' | 'notifications' | 'settings'>('report');
  const [reports, setReports] = useState<ManualReport[]>([]);
  const [notifications, setNotifications] = useState<NotificationItem[]>([]);
  const [selectedReport, setSelectedReport] = useState<ManualReport | null>(null);
  const [loadingReports, setLoadingReports] = useState(false);

  // Form State
  const [category, setCategory] = useState('COLLISION');
  const [description, setDescription] = useState('');
  const [locationName, setLocationName] = useState('');
  const [latitude, setLatitude] = useState<number | undefined>(undefined);
  const [longitude, setLongitude] = useState<number | undefined>(undefined);
  const [contactPhone, setContactPhone] = useState(user?.phone || '');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [formSuccess, setFormSuccess] = useState<string | null>(null);
  const [formError, setFormError] = useState<string | null>(null);
  const [gpsLoading, setGpsLoading] = useState(false);

  // Settings State
  const [nameInput, setNameInput] = useState(user?.full_name || '');
  const [phoneInput, setPhoneInput] = useState(user?.phone || '');
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [settingsMessage, setSettingsMessage] = useState<string | null>(null);

  const loadData = async () => {
    try {
      setLoadingReports(true);
      const myReps = await apiGetMyReports();
      setReports(myReps);
      if (user?.user_id) {
        const notifs = await apiGetNotifications(user.user_id);
        setNotifications(notifs);
      }
    } catch {
      // ignore
    } finally {
      setLoadingReports(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [user?.user_id]);

  useEffect(() => {
    if (lastEvent?.type === 'report.reviewed' || lastEvent?.type === 'report.created') {
      loadData();
    }
  }, [lastEvent]);

  const handleGetLocation = () => {
    if (!navigator.geolocation) {
      alert('Geolocation is not supported by your browser.');
      return;
    }
    setGpsLoading(true);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLatitude(Number(pos.coords.latitude.toFixed(6)));
        setLongitude(Number(pos.coords.longitude.toFixed(6)));
        setGpsLoading(false);
      },
      () => {
        // Fallback default coordinates (Bengaluru city center)
        setLatitude(12.9716);
        setLongitude(77.5946);
        setGpsLoading(false);
      },
      { timeout: 8000 }
    );
  };

  const handleReportSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError(null);
    setFormSuccess(null);
    setSubmitting(true);

    try {
      const report = await apiSubmitReport({
        category,
        description,
        location_name: locationName,
        latitude,
        longitude,
        contact_phone: contactPhone,
      });

      if (imageFile) {
        await apiUploadReportImage(report.report_id, imageFile);
      }

      setFormSuccess(`Incident report submitted successfully! Report ID: ${report.report_id}. Control Room will review and dispatch responders.`);
      setDescription('');
      setLocationName('');
      setImageFile(null);
      loadData();
    } catch (err: any) {
      setFormError(err.message || 'Error submitting report.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleUpdateProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await apiUpdateProfile(nameInput, phoneInput);
      await refreshUser();
      setSettingsMessage('Profile updated successfully.');
    } catch (err: any) {
      alert(`Error: ${err.message}`);
    }
  };

  const handleChangePassword = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await apiChangePassword(currentPassword, newPassword);
      setCurrentPassword('');
      setNewPassword('');
      setSettingsMessage('Password updated successfully.');
    } catch (err: any) {
      alert(`Error: ${err.message}`);
    }
  };

  const stats = {
    total: reports.length,
    underReview: reports.filter((r) => r.status === 'SUBMITTED' || r.status === 'UNDER_REVIEW').length,
    escalated: reports.filter((r) => r.status === 'ESCALATED' || r.status === 'APPROVED').length,
    rejected: reports.filter((r) => r.status === 'REJECTED').length,
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '30px 20px', color: '#F2F0E9' }}>
      {/* Welcome Banner */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '16px',
          marginBottom: '28px',
          paddingBottom: '20px',
          borderBottom: '1px solid #303832',
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <h1 style={{ fontSize: '26px', fontWeight: 800, margin: 0, color: '#F2F0E9' }}>
              Citizen Safety Dashboard
            </h1>
            <Badge variant="amber">CITIZEN PORTAL</Badge>
          </div>
          <p style={{ fontSize: '13px', color: '#A3ACA5', margin: '4px 0 0' }}>
            Welcome back, <b>{user?.full_name}</b> ({user?.email}) • Report hazards & track emergency responses
          </p>
        </div>

        <Button variant="primary" onClick={() => setActiveTab('report')}>
          <AlertTriangle size={16} /> Report an Accident
        </Button>
      </div>

      {/* Metrics Row */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: '16px',
          marginBottom: '28px',
        }}
      >
        <Card>
          <div style={{ fontSize: '11px', color: '#A3ACA5', textTransform: 'uppercase', fontWeight: 600 }}>
            Total Reports Filed
          </div>
          <div style={{ fontSize: '28px', fontWeight: 800, color: '#F2F0E9', margin: '4px 0' }}>
            {stats.total}
          </div>
          <div style={{ fontSize: '11px', color: '#A3ACA5' }}>Recorded in database</div>
        </Card>

        <Card>
          <div style={{ fontSize: '11px', color: '#A3ACA5', textTransform: 'uppercase', fontWeight: 600 }}>
            Under Review
          </div>
          <div style={{ fontSize: '28px', fontWeight: 800, color: '#F2A23A', margin: '4px 0' }}>
            {stats.underReview}
          </div>
          <div style={{ fontSize: '11px', color: '#A3ACA5' }}>Awaiting control room action</div>
        </Card>

        <Card>
          <div style={{ fontSize: '11px', color: '#A3ACA5', textTransform: 'uppercase', fontWeight: 600 }}>
            Dispatched / Escalated
          </div>
          <div style={{ fontSize: '28px', fontWeight: 800, color: '#55A77B', margin: '4px 0' }}>
            {stats.escalated}
          </div>
          <div style={{ fontSize: '11px', color: '#55A77B' }}>Emergency crews deployed</div>
        </Card>

        <Card>
          <div style={{ fontSize: '11px', color: '#A3ACA5', textTransform: 'uppercase', fontWeight: 600 }}>
            Dismissed / Rejected
          </div>
          <div style={{ fontSize: '28px', fontWeight: 800, color: '#A3ACA5', margin: '4px 0' }}>
            {stats.rejected}
          </div>
          <div style={{ fontSize: '11px', color: '#A3ACA5' }}>False alarm / unconfirmed</div>
        </Card>
      </div>

      {/* Navigation Tabs */}
      <div
        style={{
          display: 'flex',
          gap: '10px',
          borderBottom: '1px solid #303832',
          paddingBottom: '12px',
          marginBottom: '24px',
        }}
      >
        <button
          onClick={() => setActiveTab('report')}
          style={{
            background: 'none',
            border: 'none',
            padding: '8px 16px',
            borderRadius: '6px',
            fontSize: '13px',
            fontWeight: 600,
            cursor: 'pointer',
            backgroundColor: activeTab === 'report' ? '#1D2420' : 'transparent',
            color: activeTab === 'report' ? '#F2A23A' : '#A3ACA5',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <Send size={15} /> Submit Report
        </button>

        <button
          onClick={() => setActiveTab('history')}
          style={{
            background: 'none',
            border: 'none',
            padding: '8px 16px',
            borderRadius: '6px',
            fontSize: '13px',
            fontWeight: 600,
            cursor: 'pointer',
            backgroundColor: activeTab === 'history' ? '#1D2420' : 'transparent',
            color: activeTab === 'history' ? '#F2A23A' : '#A3ACA5',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <FileText size={15} /> My Reports ({reports.length})
        </button>

        <button
          onClick={() => setActiveTab('notifications')}
          style={{
            background: 'none',
            border: 'none',
            padding: '8px 16px',
            borderRadius: '6px',
            fontSize: '13px',
            fontWeight: 600,
            cursor: 'pointer',
            backgroundColor: activeTab === 'notifications' ? '#1D2420' : 'transparent',
            color: activeTab === 'notifications' ? '#F2A23A' : '#A3ACA5',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <Bell size={15} /> Notifications ({notifications.length})
        </button>

        <button
          onClick={() => setActiveTab('settings')}
          style={{
            background: 'none',
            border: 'none',
            padding: '8px 16px',
            borderRadius: '6px',
            fontSize: '13px',
            fontWeight: 600,
            cursor: 'pointer',
            backgroundColor: activeTab === 'settings' ? '#1D2420' : 'transparent',
            color: activeTab === 'settings' ? '#F2A23A' : '#A3ACA5',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <Settings size={15} /> Profile & Settings
        </button>
      </div>

      {/* TAB 1: SUBMIT REPORT */}
      {activeTab === 'report' && (
        <div style={{ maxWidth: '750px', margin: '0 auto' }}>
          <Card style={{ padding: '28px' }}>
            <h2 style={{ fontSize: '18px', fontWeight: 700, margin: '0 0 8px', color: '#F2F0E9' }}>
              Report Road Incident or Accident
            </h2>
            <p style={{ fontSize: '13px', color: '#A3ACA5', margin: '0 0 20px', lineHeight: 1.5 }}>
              Your report enters an unverified queue immediately visible to the Emergency Control Room. Please provide clear details and location.
            </p>

            {formSuccess && (
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '12px 16px',
                  borderRadius: '8px',
                  backgroundColor: 'rgba(85, 167, 123, 0.15)',
                  border: '1px solid #55A77B',
                  color: '#55A77B',
                  fontSize: '13px',
                  marginBottom: '20px',
                }}
              >
                <CheckCircle size={18} />
                <span>{formSuccess}</span>
              </div>
            )}

            {formError && (
              <div
                style={{
                  padding: '12px 16px',
                  borderRadius: '8px',
                  backgroundColor: 'rgba(217, 87, 87, 0.15)',
                  border: '1px solid #D95757',
                  color: '#D95757',
                  fontSize: '13px',
                  marginBottom: '20px',
                }}
              >
                {formError}
              </div>
            )}

            <form onSubmit={handleReportSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#A3ACA5', marginBottom: '6px' }}>
                  INCIDENT CATEGORY
                </label>
                <select
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  style={{
                    width: '100%',
                    backgroundColor: '#111512',
                    border: '1px solid #303832',
                    borderRadius: '8px',
                    padding: '10px 12px',
                    color: '#F2F0E9',
                    fontSize: '13px',
                  }}
                >
                  <option value="COLLISION">Vehicle Collision / Crash</option>
                  <option value="PEDESTRIAN">Pedestrian Struck / Down</option>
                  <option value="ROLLOVER">Vehicle Rollover</option>
                  <option value="FIRE">Vehicle Fire / Hazard</option>
                  <option value="HIT_AND_RUN">Hit and Run</option>
                  <option value="MEDICAL">Medical Emergency on Road</option>
                </select>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#A3ACA5', marginBottom: '6px' }}>
                  DESCRIPTION & SITUATION
                </label>
                <Textarea
                  required
                  placeholder="Describe what occurred, number of vehicles or victims, road blockages..."
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#A3ACA5', marginBottom: '6px' }}>
                  LOCATION NAME OR LANDMARK
                </label>
                <Input
                  type="text"
                  placeholder="e.g. MG Road / Residency Road Junction, near Metro Pillar 140"
                  value={locationName}
                  onChange={(e) => setLocationName(e.target.value)}
                />
              </div>

              {/* Coordinates & GPS */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                  <label style={{ fontSize: '12px', fontWeight: 600, color: '#A3ACA5' }}>
                    GPS COORDINATES (LATITUDE, LONGITUDE)
                  </label>
                  <button
                    type="button"
                    onClick={handleGetLocation}
                    disabled={gpsLoading}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: '#F2A23A',
                      fontSize: '12px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px',
                    }}
                  >
                    <Navigation size={13} /> {gpsLoading ? 'Acquiring GPS...' : 'Use My Current GPS'}
                  </button>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                  <Input
                    type="number"
                    step="any"
                    placeholder="Latitude (e.g. 12.9716)"
                    value={latitude !== undefined ? latitude : ''}
                    onChange={(e) => setLatitude(e.target.value ? parseFloat(e.target.value) : undefined)}
                  />
                  <Input
                    type="number"
                    step="any"
                    placeholder="Longitude (e.g. 77.5946)"
                    value={longitude !== undefined ? longitude : ''}
                    onChange={(e) => setLongitude(e.target.value ? parseFloat(e.target.value) : undefined)}
                  />
                </div>
              </div>

              {/* Optional Photo Upload */}
              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#A3ACA5', marginBottom: '6px' }}>
                  OPTIONAL SCENE PHOTO
                </label>
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      setImageFile(e.target.files[0]);
                    }
                  }}
                  style={{
                    width: '100%',
                    backgroundColor: '#111512',
                    border: '1px solid #303832',
                    borderRadius: '8px',
                    padding: '8px',
                    color: '#A3ACA5',
                    fontSize: '12px',
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#A3ACA5', marginBottom: '6px' }}>
                  CONTACT PHONE
                </label>
                <Input
                  type="tel"
                  placeholder="+91-98765-43210"
                  value={contactPhone}
                  onChange={(e) => setContactPhone(e.target.value)}
                />
              </div>

              <Button
                type="submit"
                variant="primary"
                disabled={submitting}
                style={{ width: '100%', padding: '12px', fontSize: '14px', marginTop: '6px' }}
              >
                {submitting ? 'Transmitting to Control Room...' : 'Submit Incident Report'} <Send size={16} />
              </Button>
            </form>
          </Card>
        </div>
      )}

      {/* TAB 2: REPORT HISTORY */}
      {activeTab === 'history' && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h2 style={{ fontSize: '18px', fontWeight: 700, margin: 0, color: '#F2F0E9' }}>
              Your Submitted Incident Reports
            </h2>
            <Button variant="outline" onClick={loadData}>
              Refresh
            </Button>
          </div>

          {loadingReports ? (
            <div style={{ textAlign: 'center', padding: '40px', color: '#A3ACA5' }}>Loading reports...</div>
          ) : reports.length === 0 ? (
            <Card style={{ textAlign: 'center', padding: '40px' }}>
              <FileText size={36} color="#A3ACA5" style={{ opacity: 0.4, marginBottom: '10px' }} />
              <div style={{ fontSize: '15px', fontWeight: 600, color: '#F2F0E9', marginBottom: '6px' }}>
                No reports submitted yet
              </div>
              <p style={{ fontSize: '13px', color: '#A3ACA5', maxWidth: '400px', margin: '0 auto 16px' }}>
                You have not filed any incident reports. Click below to file your first report.
              </p>
              <Button variant="primary" onClick={() => setActiveTab('report')}>
                File Incident Report
              </Button>
            </Card>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {reports.map((rep) => (
                <Card
                  key={rep.report_id}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    flexWrap: 'wrap',
                    gap: '12px',
                    padding: '16px 20px',
                    cursor: 'pointer',
                  }}
                  onClick={() => setSelectedReport(rep)}
                >
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px' }}>
                      <span style={{ fontFamily: 'monospace', fontWeight: 700, color: '#F2A23A', fontSize: '13px' }}>
                        {rep.report_id}
                      </span>
                      <span style={{ fontSize: '14px', fontWeight: 600, color: '#F2F0E9' }}>
                        {rep.category}
                      </span>
                      <Badge
                        variant={
                          rep.status === 'ESCALATED' || rep.status === 'APPROVED'
                            ? 'success'
                            : rep.status === 'SUBMITTED' || rep.status === 'UNDER_REVIEW'
                            ? 'amber'
                            : 'neutral'
                        }
                      >
                        {rep.status}
                      </Badge>
                    </div>

                    <div style={{ fontSize: '13px', color: '#A3ACA5', marginBottom: '6px', maxWidth: '600px' }}>
                      {rep.description}
                    </div>

                    <div style={{ display: 'flex', gap: '16px', fontSize: '11px', color: '#A3ACA5' }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <MapPin size={12} color="#F2A23A" />
                        {rep.location_name || (rep.latitude ? `${rep.latitude}, ${rep.longitude}` : 'Unknown')}
                      </span>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Clock size={12} />
                        {rep.created_at}
                      </span>
                      {rep.linked_incident_id && (
                        <span style={{ color: '#55A77B', fontWeight: 600 }}>
                          Linked Incident: {rep.linked_incident_id}
                        </span>
                      )}
                    </div>
                  </div>

                  <Button variant="outline" style={{ padding: '6px 12px', fontSize: '12px' }}>
                    <Eye size={14} /> View Details
                  </Button>
                </Card>
              ))}
            </div>
          )}
        </div>
      )}

      {/* TAB 3: NOTIFICATIONS */}
      {activeTab === 'notifications' && (
        <div style={{ maxWidth: '750px', margin: '0 auto' }}>
          <h2 style={{ fontSize: '18px', fontWeight: 700, margin: '0 0 16px', color: '#F2F0E9' }}>
            System Notifications
          </h2>
          {notifications.length === 0 ? (
            <Card style={{ textAlign: 'center', padding: '40px', color: '#A3ACA5' }}>
              <Bell size={32} style={{ opacity: 0.4, marginBottom: '8px' }} />
              <div>No new notifications.</div>
            </Card>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {notifications.map((n) => (
                <Card key={n.id} style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                  <Bell size={16} color="#F2A23A" style={{ marginTop: '2px' }} />
                  <div>
                    <div style={{ fontSize: '14px', fontWeight: 600, color: '#F2F0E9' }}>{n.title}</div>
                    <div style={{ fontSize: '13px', color: '#A3ACA5', margin: '2px 0 4px' }}>{n.body}</div>
                    <div style={{ fontSize: '11px', color: '#7E8880' }}>{n.created_at}</div>
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>
      )}

      {/* TAB 4: PROFILE SETTINGS */}
      {activeTab === 'settings' && (
        <div style={{ maxWidth: '650px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <Card style={{ padding: '24px' }}>
            <h2 style={{ fontSize: '16px', fontWeight: 700, margin: '0 0 16px', color: '#F2F0E9' }}>
              Profile Information
            </h2>
            {settingsMessage && (
              <div style={{ padding: '10px 14px', borderRadius: '6px', backgroundColor: 'rgba(85, 167, 123, 0.15)', color: '#55A77B', fontSize: '12px', marginBottom: '16px' }}>
                {settingsMessage}
              </div>
            )}
            <form onSubmit={handleUpdateProfile} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#A3ACA5', marginBottom: '6px' }}>
                  FULL NAME
                </label>
                <Input value={nameInput} onChange={(e) => setNameInput(e.target.value)} required />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#A3ACA5', marginBottom: '6px' }}>
                  PHONE NUMBER
                </label>
                <Input value={phoneInput} onChange={(e) => setPhoneInput(e.target.value)} />
              </div>

              <Button type="submit" variant="primary" style={{ alignSelf: 'flex-start' }}>
                Save Profile
              </Button>
            </form>
          </Card>

          <Card style={{ padding: '24px' }}>
            <h2 style={{ fontSize: '16px', fontWeight: 700, margin: '0 0 16px', color: '#F2F0E9' }}>
              Change Password
            </h2>
            <form onSubmit={handleChangePassword} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#A3ACA5', marginBottom: '6px' }}>
                  CURRENT PASSWORD
                </label>
                <Input type="password" value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} required />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#A3ACA5', marginBottom: '6px' }}>
                  NEW PASSWORD
                </label>
                <Input type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} required />
              </div>

              <Button type="submit" variant="secondary" style={{ alignSelf: 'flex-start' }}>
                Update Password
              </Button>
            </form>
          </Card>
        </div>
      )}

      {/* Report Detail Modal */}
      {selectedReport && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.75)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 100,
            padding: '20px',
          }}
          onClick={() => setSelectedReport(null)}
        >
          <div
            style={{
              width: '100%',
              maxWidth: '600px',
              backgroundColor: '#151917',
              border: '1px solid #303832',
              borderRadius: '16px',
              padding: '24px',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontFamily: 'monospace', fontWeight: 700, color: '#F2A23A', fontSize: '16px' }}>
                  {selectedReport.report_id}
                </span>
                <Badge variant={selectedReport.status === 'ESCALATED' ? 'success' : 'amber'}>
                  {selectedReport.status}
                </Badge>
              </div>
              <button
                onClick={() => setSelectedReport(null)}
                style={{ background: 'none', border: 'none', color: '#A3ACA5', cursor: 'pointer' }}
              >
                <X size={20} />
              </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '13px' }}>
              <div>
                <b style={{ color: '#A3ACA5' }}>Category:</b> {selectedReport.category}
              </div>
              <div>
                <b style={{ color: '#A3ACA5' }}>Description:</b>
                <p style={{ margin: '4px 0', color: '#F2F0E9' }}>{selectedReport.description}</p>
              </div>
              <div>
                <b style={{ color: '#A3ACA5' }}>Location:</b> {selectedReport.location_name || 'N/A'}
              </div>
              {selectedReport.latitude && (
                <div>
                  <b style={{ color: '#A3ACA5' }}>Coordinates:</b> {selectedReport.latitude}, {selectedReport.longitude}
                </div>
              )}
              {selectedReport.image_path && (
                <div>
                  <b style={{ color: '#A3ACA5' }}>Attached Photo:</b>
                  <div style={{ marginTop: '6px', borderRadius: '8px', overflow: 'hidden', maxHeight: '200px' }}>
                    <img src={selectedReport.image_path} alt="Report attachment" style={{ width: '100%', objectFit: 'cover' }} />
                  </div>
                </div>
              )}
              {selectedReport.admin_notes && (
                <div style={{ backgroundColor: '#1D2420', padding: '10px', borderRadius: '8px', border: '1px solid #303832' }}>
                  <b style={{ color: '#F2A23A' }}>Control Room Notes:</b> {selectedReport.admin_notes}
                </div>
              )}
              {selectedReport.linked_incident_id && (
                <div style={{ backgroundColor: 'rgba(85, 167, 123, 0.1)', padding: '10px', borderRadius: '8px', border: '1px solid #55A77B', color: '#55A77B' }}>
                  <b>Active Emergency Dispatch Created:</b> Incident ID <b>{selectedReport.linked_incident_id}</b>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
