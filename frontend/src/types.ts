export type Role = 'CITIZEN' | 'ADMIN' | 'RESPONDER';

export interface User {
  user_id: string;
  email: string;
  full_name: string;
  role: Role;
  phone?: string;
  status: 'ACTIVE' | 'SUSPENDED';
  responder_id?: string;
  created_at?: string;
  updated_at?: string;
}

export interface Incident {
  id?: number;
  incident_id: string;
  status: 'DETECTED' | 'DISPATCHED' | 'ACCEPTED' | 'ON_SCENE' | 'TRANSPORTING' | 'CLOSED' | 'CANCELLED' | 'FALSE_ALARM';
  kind: string;
  label: string;
  severity: 'CRITICAL' | 'HIGH' | 'MODERATE' | 'LOW';
  confidence: number;
  reason?: string;
  evidence_types?: string | string[];
  signals?: string | any[];
  involved?: string | any[];
  vehicles?: string | any[];
  ml_probability?: number;
  camera_id: string;
  location_name?: string;
  latitude: number;
  longitude: number;
  frame?: number;
  snapshot_path?: string;
  detected_at: string;
  created_at: string;
  assigned_responder?: string;
  accepted_at?: string;
  arrived_at?: string;
  hospital_name?: string;
  hospital_latitude?: number;
  hospital_longitude?: number;
  transport_started_at?: string;
  closed_at?: string;
  outcome?: string;
  dispatches?: Dispatch[];
  timeline?: IncidentEvent[];
}

export interface Responder {
  id?: number;
  responder_id: string;
  name: string;
  responder_type: 'AMBULANCE' | 'PARAMEDIC' | 'POLICE' | 'FIRE';
  phone?: string;
  vehicle_number?: string;
  latitude: number;
  longitude: number;
  status: 'AVAILABLE' | 'ALERTED' | 'EN_ROUTE' | 'ON_SCENE' | 'TRANSPORTING' | 'OFFLINE';
  active_incident?: string | null;
  registered_at?: string;
  last_seen?: string;
  assignments?: any[];
}

export interface Dispatch {
  id?: number;
  incident_id: string;
  responder_id: string;
  status: 'SENT' | 'ACCEPTED' | 'DECLINED' | 'SUPERSEDED';
  distance_km?: number;
  eta_minutes?: number;
  sent_at: string;
  responded_at?: string;
  name?: string;
  responder_type?: string;
  phone?: string;
  latitude?: number;
  longitude?: number;
}

export interface IncidentEvent {
  id: number;
  incident_id: string;
  event_type: string;
  message: string;
  actor?: string;
  payload: string | Record<string, any>;
  created_at: string;
}

export interface ManualReport {
  id?: number;
  report_id: string;
  user_id: string;
  category: string;
  description: string;
  location_name?: string;
  latitude?: number;
  longitude?: number;
  image_path?: string;
  contact_phone?: string;
  status: 'SUBMITTED' | 'UNDER_REVIEW' | 'APPROVED' | 'REJECTED' | 'ESCALATED';
  admin_notes?: string;
  reviewed_by?: string;
  reviewed_at?: string;
  linked_incident_id?: string;
  created_at: string;
  updated_at: string;
  reporter_name?: string;
  reporter_email?: string;
}

export interface Facility {
  id?: number;
  ref: string;
  name: string;
  kind: string;
  latitude: number;
  longitude: number;
  phone?: string;
  address?: string;
  emergency: number;
  source?: string;
  travel_time_seconds?: number;
  travel_time_minutes?: number;
  distance_km?: number;
}

export interface AuditLog {
  id?: number;
  log_id: string;
  user_id?: string;
  action: string;
  target_type?: string;
  target_id?: string;
  details: string | Record<string, any>;
  ip_address?: string;
  created_at: string;
  user_email?: string;
  user_name?: string;
  user_role?: string;
}

export interface NotificationItem {
  id: number;
  recipient: string;
  incident_id?: string;
  title: string;
  body: string;
  level: string;
  created_at: string;
  read_at?: string;
}

export interface SystemStats {
  totals: {
    total: number;
    closed: number;
    false_alarms: number;
  };
  by_kind: Array<{ kind: string; count: number }>;
  by_severity: Array<{ severity: string; count: number }>;
  avg_acceptance_seconds: number;
  avg_travel_seconds: number;
}
