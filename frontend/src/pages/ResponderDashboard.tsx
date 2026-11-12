import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../context/AuthContext';
import { useRealtime } from '../context/RealtimeContext';
import { Card, Button, Badge, ElevatedCard } from '../components/ui';
import { MapView } from '../components/MapView';
import {
  apiGetResponder,
  apiGetResponderAlerts,
  apiAcceptIncident,
  apiDeclineIncident,
  apiMarkArrived,
  apiGetIncidentHospitals,
  apiChooseHospital,
  apiCloseIncident,
  apiUpdateResponderLocation,
  apiUpdateResponderStatus,
} from '../api';
import { Incident, Responder, Facility } from '../types';
import {
  Radio,
  MapPin,
  Navigation,
  Hospital,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Clock,
  Compass,
  ArrowRight,
  ShieldCheck,
  Power,
} from 'lucide-react';

export const ResponderDashboard: React.FC = () => {
  const { user } = useAuth();
  const { lastEvent, playAlertTone } = useRealtime();

  const responderId = user?.responder_id || 'RESP-0001';
  const [responder, setResponder] = useState<Responder | null>(null);
  const [activeCall, setActiveCall] = useState<Incident | null>(null);
  const [incomingAlert, setIncomingAlert] = useState<any | null>(null);
  const [hospitals, setHospitals] = useState<Facility[]>([]);
  const [routeGeometry, setRouteGeometry] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);

  // GPS streaming state
  const [gpsStreaming, setGpsStreaming] = useState(false);
  const [gpsCoords, setGpsCoords] = useState<{ lat: number; lon: number }>({ lat: 12.9716, lon: 77.5946 });
  const [distanceToScene, setDistanceToScene] = useState<number | null>(null);
  const gpsIntervalRef = useRef<number | null>(null);

  const loadResponderData = async () => {
    try {
      const resp = await apiGetResponder(responderId);
      setResponder(resp);

      // Check alerts
      const alerts = await apiGetResponderAlerts(responderId);
      if (alerts && alerts.length > 0) {
        const primary = alerts[0];
        if (primary.dispatch_status === 'SENT') {
          setIncomingAlert(primary);
        } else if (primary.dispatch_status === 'ACCEPTED') {
          setActiveCall(primary);
        }
      } else {
        setIncomingAlert(null);
      }
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    loadResponderData();
  }, [responderId]);

  useEffect(() => {
    if (!lastEvent) return;
    if (lastEvent.type === 'incident.alert') {
      loadResponderData();
      playAlertTone();
    } else if (lastEvent.type === 'incident.stand_down') {
      alert('Another emergency crew accepted this case. You may stand down.');
      setIncomingAlert(null);
      loadResponderData();
    }
  }, [lastEvent]);

  // GPS Streaming loop
  useEffect(() => {
    if (gpsStreaming) {
      const streamFix = () => {
        if (navigator.geolocation) {
          navigator.geolocation.getCurrentPosition(
            async (pos) => {
              const lat = pos.coords.latitude;
              const lon = pos.coords.longitude;
              setGpsCoords({ lat, lon });
              try {
                const res = await apiUpdateResponderLocation(responderId, lat, lon, pos.coords.speed || 0, pos.coords.heading || 0);
                if (res.distance_to_scene_m !== undefined) {
                  setDistanceToScene(res.distance_to_scene_m);
                }
                if (res.arrived) {
                  alert('Arrival geofence triggered! You have reached the accident scene.');
                  loadResponderData();
                }
              } catch {
                // ignore
              }
            },
            async () => {
              // Simulated nudge towards scene for demo/testing
              if (activeCall) {
                const nudgeLat = gpsCoords.lat + (activeCall.latitude - gpsCoords.lat) * 0.15;
                const nudgeLon = gpsCoords.lon + (activeCall.longitude - gpsCoords.lon) * 0.15;
                setGpsCoords({ lat: nudgeLat, lon: nudgeLon });
                const res = await apiUpdateResponderLocation(responderId, nudgeLat, nudgeLon, 45, 90);
                if (res.arrived) {
                  alert('Arrival geofence triggered! You have reached the scene.');
                  loadResponderData();
                }
              }
            },
            { enableHighAccuracy: true, timeout: 5000 }
          );
        }
      };

      streamFix();
      gpsIntervalRef.current = window.setInterval(streamFix, 4000);
    } else {
      if (gpsIntervalRef.current) clearInterval(gpsIntervalRef.current);
    }

    return () => {
      if (gpsIntervalRef.current) clearInterval(gpsIntervalRef.current);
    };
  }, [gpsStreaming, activeCall?.incident_id, responderId]);

  const handleToggleAvailability = async () => {
    if (!responder) return;
    const newStatus = responder.status === 'AVAILABLE' ? 'OFFLINE' : 'AVAILABLE';
    try {
      const updated = await apiUpdateResponderStatus(responderId, newStatus);
      setResponder(updated);
    } catch (err: any) {
      alert(`Error updating status: ${err.message}`);
    }
  };

  const handleAcceptCall = async (incidentId: string) => {
    try {
      setLoading(true);
      const res = await apiAcceptIncident(incidentId, responderId);
      setActiveCall(res.incident);
      setIncomingAlert(null);
      if (res.route) {
        setRouteGeometry(res.route.geometry);
      }
      setGpsStreaming(true);
      await loadResponderData();
    } catch (err: any) {
      alert(`Error accepting call: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleDeclineCall = async (incidentId: string) => {
    try {
      setLoading(true);
      await apiDeclineIncident(incidentId, responderId);
      setIncomingAlert(null);
      await loadResponderData();
    } catch (err: any) {
      alert(`Error declining: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleManualArrived = async () => {
    if (!activeCall) return;
    try {
      const updated = await apiMarkArrived(activeCall.incident_id, responderId);
      setActiveCall(updated);
      // Fetch nearby hospitals
      const hospRes = await apiGetIncidentHospitals(activeCall.incident_id);
      setHospitals(hospRes.facilities);
    } catch (err: any) {
      alert(`Error: ${err.message}`);
    }
  };

  const handleSelectHospital = async (fac: Facility) => {
    if (!activeCall) return;
    try {
      const res = await apiChooseHospital(activeCall.incident_id, responderId, {
        name: fac.name,
        latitude: fac.latitude,
        longitude: fac.longitude,
        phone: fac.phone,
      });
      setActiveCall(res.incident);
      if (res.route) {
        setRouteGeometry(res.route.geometry);
      }
      alert(`Hospital locked: ${fac.name}. Least-traffic route plotted!`);
    } catch (err: any) {
      alert(`Error selecting hospital: ${err.message}`);
    }
  };

  const handleCloseCase = async () => {
    if (!activeCall) return;
    const notes = prompt('Enter handover / patient delivery notes:') || 'Handover completed';
    try {
      await apiCloseIncident(activeCall.incident_id, 'RESOLVED', notes, responderId);
      setActiveCall(null);
      setHospitals([]);
      setRouteGeometry(null);
      setGpsStreaming(false);
      await loadResponderData();
      alert('Case closed. Unit status set to AVAILABLE.');
    } catch (err: any) {
      alert(`Error closing case: ${err.message}`);
    }
  };

  return (
    <div style={{ maxWidth: '640px', margin: '0 auto', padding: '16px', color: '#F2F0E9' }}>
      {/* Unit Status Card */}
      <Card style={{ marginBottom: '16px', padding: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Radio size={18} color="#F2A23A" />
              <span style={{ fontSize: '16px', fontWeight: 800, color: '#F2F0E9' }}>
                {responder?.name || responderId}
              </span>
            </div>
            <div style={{ fontSize: '11px', color: '#A3ACA5', marginTop: '2px' }}>
              Unit: <b>{responder?.vehicle_number || 'KA-01-EA-1108'}</b> • Type: <b>{responder?.responder_type || 'AMBULANCE'}</b>
            </div>
          </div>

          <button
            onClick={handleToggleAvailability}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 14px',
              borderRadius: '20px',
              border: 'none',
              backgroundColor: responder?.status === 'AVAILABLE' ? 'rgba(85, 167, 123, 0.2)' : 'rgba(217, 87, 87, 0.2)',
              color: responder?.status === 'AVAILABLE' ? '#55A77B' : '#D95757',
              fontWeight: 700,
              fontSize: '12px',
              cursor: 'pointer',
            }}
          >
            <Power size={13} />
            {responder?.status || 'OFFLINE'}
          </button>
        </div>
      </Card>

      {/* INCOMING EMERGENCY ALERT BANNER */}
      {incomingAlert && (
        <div
          style={{
            backgroundColor: '#1D2420',
            border: '2px solid #D95757',
            borderRadius: '14px',
            padding: '20px',
            marginBottom: '18px',
            animation: 'pulse 1.5s infinite',
            boxShadow: '0 0 25px rgba(217, 87, 87, 0.4)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
            <AlertTriangle size={24} color="#D95757" />
            <div>
              <div style={{ fontSize: '17px', fontWeight: 800, color: '#D95757' }}>
                NEW EMERGENCY DISPATCH
              </div>
              <div style={{ fontSize: '11px', color: '#A3ACA5' }}>
                Immediate crew response requested
              </div>
            </div>
          </div>

          <div style={{ fontSize: '15px', fontWeight: 700, color: '#F2F0E9', margin: '8px 0 4px' }}>
            {incomingAlert.label}
          </div>
          <div style={{ fontSize: '12px', color: '#A3ACA5', marginBottom: '14px' }}>
            Location: <b>{incomingAlert.location_name || 'Near camera junction'}</b> • Distance:{' '}
            <b>{incomingAlert.distance_km || '1.2'} km</b> • ETA: <b>{incomingAlert.eta_minutes || '4'} min</b>
          </div>

          <div style={{ display: 'flex', gap: '10px' }}>
            <Button
              variant="primary"
              disabled={loading}
              onClick={() => handleAcceptCall(incomingAlert.incident_id)}
              style={{ flex: 2, padding: '12px', fontSize: '14px', backgroundColor: '#55A77B' }}
            >
              <CheckCircle2 size={16} /> ACCEPT CALL
            </Button>
            <Button
              variant="secondary"
              disabled={loading}
              onClick={() => handleDeclineCall(incomingAlert.incident_id)}
              style={{ flex: 1, padding: '12px', fontSize: '14px' }}
            >
              <XCircle size={16} /> Decline
            </Button>
          </div>
        </div>
      )}

      {/* ACTIVE CALL WORKFLOW */}
      {activeCall ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <Card style={{ borderLeft: '4px solid #F2A23A', padding: '18px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontFamily: 'monospace', fontWeight: 700, color: '#F2A23A', fontSize: '14px' }}>
                {activeCall.incident_id}
              </span>
              <Badge variant="amber">{activeCall.status}</Badge>
            </div>

            <h2 style={{ fontSize: '18px', fontWeight: 800, margin: '0 0 6px', color: '#F2F0E9' }}>
              {activeCall.label}
            </h2>
            <div style={{ fontSize: '12px', color: '#A3ACA5', marginBottom: '14px' }}>
              Location: <b>{activeCall.location_name || `${activeCall.latitude}, ${activeCall.longitude}`}</b>
            </div>

            {/* GPS Telemetry row */}
            <div
              style={{
                backgroundColor: '#111512',
                border: '1px solid #303832',
                borderRadius: '8px',
                padding: '10px 14px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '14px',
                fontSize: '12px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Navigation size={14} color="#55A77B" />
                <span>GPS Fix: <b>{gpsCoords.lat.toFixed(4)}, {gpsCoords.lon.toFixed(4)}</b></span>
              </div>
              <button
                onClick={() => setGpsStreaming(!gpsStreaming)}
                style={{
                  background: 'none',
                  border: 'none',
                  color: gpsStreaming ? '#55A77B' : '#A3ACA5',
                  fontSize: '11px',
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                {gpsStreaming ? '● STREAMING' : 'START GPS'}
              </button>
            </div>

            {/* Tactical Route Map */}
            <MapView
              incidents={[activeCall]}
              responders={responder ? [responder] : []}
              facilities={hospitals}
              routeGeometry={routeGeometry}
              center={[activeCall.latitude, activeCall.longitude]}
              zoom={14}
              height="260px"
            />

            {/* Status Steps Action Buttons */}
            <div style={{ marginTop: '16px' }}>
              {!activeCall.arrived_at ? (
                <Button
                  variant="primary"
                  onClick={handleManualArrived}
                  style={{ width: '100%', padding: '13px', fontSize: '15px' }}
                >
                  <MapPin size={18} /> CONFIRM ARRIVAL AT SCENE
                </Button>
              ) : activeCall.status !== 'TRANSPORTING' ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <div style={{ fontSize: '13px', fontWeight: 700, color: '#55A77B', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <CheckCircle2 size={16} /> ON SCENE • SELECT RECEIVING HOSPITAL
                  </div>

                  {hospitals.map((hosp) => (
                    <div
                      key={hosp.ref}
                      onClick={() => handleSelectHospital(hosp)}
                      style={{
                        padding: '12px 14px',
                        backgroundColor: '#111512',
                        border: '1px solid #303832',
                        borderRadius: '8px',
                        cursor: 'pointer',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                      }}
                    >
                      <div>
                        <div style={{ fontSize: '14px', fontWeight: 700, color: '#F2F0E9' }}>{hosp.name}</div>
                        <div style={{ fontSize: '11px', color: '#A3ACA5' }}>{hosp.address || 'Hospital'} • Phone: {hosp.phone || 'N/A'}</div>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: '14px', fontWeight: 700, color: '#55A77B' }}>
                          {hosp.travel_time_minutes || 6} min
                        </div>
                        <div style={{ fontSize: '10px', color: '#A3ACA5' }}>by road</div>
                      </div>
                    </div>
                  ))}
                  {hospitals.length === 0 && (
                    <Button variant="secondary" onClick={handleManualArrived}>
                      Load Nearby Hospitals
                    </Button>
                  )}
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <div style={{ padding: '12px', backgroundColor: 'rgba(85, 167, 123, 0.1)', border: '1px solid #55A77B', borderRadius: '8px', color: '#55A77B', fontSize: '13px' }}>
                    <b>Transporting Patient:</b> {activeCall.hospital_name}
                  </div>
                  <Button
                    variant="success"
                    onClick={handleCloseCase}
                    style={{ width: '100%', padding: '13px', fontSize: '15px' }}
                  >
                    <CheckCircle2 size={18} /> COMPLETE HANDOVER & CLOSE CASE
                  </Button>
                </div>
              )}
            </div>
          </Card>
        </div>
      ) : (
        <Card style={{ textAlign: 'center', padding: '40px 20px', color: '#A3ACA5' }}>
          <ShieldCheck size={42} color="#55A77B" style={{ opacity: 0.8, marginBottom: '12px' }} />
          <div style={{ fontSize: '17px', fontWeight: 700, color: '#F2F0E9', marginBottom: '6px' }}>
            Unit on Standby
          </div>
          <p style={{ fontSize: '13px', maxWidth: '360px', margin: '0 auto', lineHeight: 1.5 }}>
            Your crew is connected to the emergency dispatch hub. When an accident is confirmed in your jurisdiction, audio & visual alerts will trigger here.
          </p>
        </Card>
      )}
    </div>
  );
};
