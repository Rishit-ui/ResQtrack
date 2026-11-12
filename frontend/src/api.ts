import { User, Incident, Responder, ManualReport, Facility, AuditLog, SystemStats, NotificationItem } from './types';

// In production on Vercel, requests can target VITE_API_URL or fall back to local origin /api
const BACKEND_URL = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');
const API_BASE = BACKEND_URL ? `${BACKEND_URL}/api` : '/api';

export function getToken(): string | null {
  return localStorage.getItem('resqtrack_token');
}

export function setToken(token: string): void {
  localStorage.setItem('resqtrack_token', token);
}

export function clearToken(): void {
  localStorage.removeItem('resqtrack_token');
  localStorage.removeItem('resqtrack_user');
}

export function getStoredUser(): User | null {
  const raw = localStorage.getItem('resqtrack_user');
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

export function setStoredUser(user: User): void {
  localStorage.setItem('resqtrack_user', JSON.stringify(user));
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string> || {}),
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers,
    });
  } catch (networkErr: any) {
    throw new Error(
      `Network Error: Unable to reach emergency backend service at ${API_BASE}. Please verify your internet connection or backend deployment status.`
    );
  }

  if (!response.ok) {
    let errMessage = '';
    let parsedDetail: any = null;
    try {
      const errData = await response.json();
      if (errData && errData.detail) {
        parsedDetail = errData.detail;
        errMessage = typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail);
      }
    } catch {
      // response body was not JSON (e.g. 404 HTML from Vercel)
    }

    if (!errMessage) {
      if (response.status === 404) {
        errMessage = `API endpoint not found (404) at ${API_BASE}${endpoint}. The frontend is either pointing to an incorrect backend URL or the route does not exist.`;
      } else if (response.status === 401) {
        errMessage = 'Invalid credentials. Please verify your email and password.';
      } else if (response.status === 403) {
        errMessage = 'Access forbidden (403): You do not have permission to perform this action.';
      } else if (response.status === 429) {
        errMessage = 'Rate limit exceeded: Too many requests. Please wait a minute before trying again.';
      } else if (response.status >= 500) {
        errMessage = `Backend server error (${response.status}). Emergency service is encountering an internal error.`;
      } else {
        errMessage = `Request failed (${response.status}: ${response.statusText || 'Unknown error'})`;
      }
    }
    throw new Error(errMessage);
  }

  return response.json();
}

// ============================================================
// AUTH API
// ============================================================

export async function apiLogin(email: string, password: string): Promise<{ access_token: string; user: User }> {
  const data = await request<{ access_token: string; user: User }>('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });
  setToken(data.access_token);
  setStoredUser(data.user);
  return data;
}

export async function apiRegister(email: string, password: string, fullName: string, phone?: string): Promise<{ access_token: string; user: User }> {
  const data = await request<{ access_token: string; user: User }>('/auth/register', {
    method: 'POST',
    body: JSON.stringify({ email, password, full_name: fullName, phone }),
  });
  setToken(data.access_token);
  setStoredUser(data.user);
  return data;
}

export async function apiGetMe(): Promise<User> {
  const user = await request<User>('/auth/me');
  setStoredUser(user);
  return user;
}

export async function apiUpdateProfile(fullName?: string, phone?: string): Promise<User> {
  const user = await request<User>('/auth/profile', {
    method: 'PUT',
    body: JSON.stringify({ full_name: fullName, phone }),
  });
  setStoredUser(user);
  return user;
}

export async function apiChangePassword(currentPassword: string, newPassword: string): Promise<{ message: string }> {
  return request<{ message: string }>('/auth/change-password', {
    method: 'POST',
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  });
}

export async function apiForgotPassword(email: string): Promise<{ message: string; reset_token: string }> {
  return request<{ message: string; reset_token: string }>('/auth/forgot-password', {
    method: 'POST',
    body: JSON.stringify({ email }),
  });
}

export async function apiResetPassword(token: string, newPassword: string): Promise<{ message: string }> {
  return request<{ message: string }>('/auth/reset-password', {
    method: 'POST',
    body: JSON.stringify({ token, new_password: newPassword }),
  });
}

// ============================================================
// INCIDENTS API
// ============================================================

export async function apiListIncidents(status?: string, activeOnly = false, limit = 50): Promise<Incident[]> {
  const params = new URLSearchParams();
  if (status) params.append('status', status);
  if (activeOnly) params.append('active_only', 'true');
  params.append('limit', limit.toString());
  return request<Incident[]>(`/incidents?${params.toString()}`);
}

export async function apiGetIncident(incidentId: string): Promise<Incident> {
  return request<Incident>(`/incidents/${incidentId}`);
}

export async function apiAcceptIncident(incidentId: string, responderId: string): Promise<{ incident: Incident; responder: Responder; route?: any }> {
  return request<{ incident: Incident; responder: Responder; route?: any }>(`/incidents/${incidentId}/accept`, {
    method: 'POST',
    body: JSON.stringify({ responder_id: responderId }),
  });
}

export async function apiDeclineIncident(incidentId: string, responderId: string): Promise<any> {
  return request<any>(`/incidents/${incidentId}/decline`, {
    method: 'POST',
    body: JSON.stringify({ responder_id: responderId }),
  });
}

export async function apiMarkArrived(incidentId: string, responderId: string): Promise<Incident> {
  return request<Incident>(`/incidents/${incidentId}/arrive`, {
    method: 'POST',
    body: JSON.stringify({ responder_id: responderId }),
  });
}

export async function apiGetIncidentHospitals(incidentId: string, radiusKm = 15.0): Promise<{ incident_id: string; facilities: Facility[]; count: number; source: string }> {
  return request<{ incident_id: string; facilities: Facility[]; count: number; source: string }>(`/incidents/${incidentId}/hospitals?radius_km=${radiusKm}`);
}

export async function apiChooseHospital(incidentId: string, responderId: string, hospital: { name: string; latitude: number; longitude: number; phone?: string }): Promise<{ incident: Incident; route?: any }> {
  return request<{ incident: Incident; route?: any }>(`/incidents/${incidentId}/hospital`, {
    method: 'POST',
    body: JSON.stringify({ responder_id: responderId, ...hospital }),
  });
}

export async function apiCloseIncident(incidentId: string, outcome: string, notes?: string, responderId?: string): Promise<Incident> {
  return request<Incident>(`/incidents/${incidentId}/close`, {
    method: 'POST',
    body: JSON.stringify({ outcome, notes, responder_id: responderId }),
  });
}

// ============================================================
// RESPONDERS API
// ============================================================

export async function apiListResponders(status?: string): Promise<Responder[]> {
  const url = status ? `/responders?status=${status}` : '/responders';
  return request<Responder[]>(url);
}

export async function apiGetResponder(responderId: string): Promise<Responder> {
  return request<Responder>(`/responders/${responderId}`);
}

export async function apiUpdateResponderLocation(responderId: string, lat: number, lon: number, speed?: number, heading?: number): Promise<{ responder: Responder; arrived: boolean; distance_to_scene_m?: number }> {
  return request<{ responder: Responder; arrived: boolean; distance_to_scene_m?: number }>(`/responders/${responderId}/location`, {
    method: 'POST',
    body: JSON.stringify({ latitude: lat, longitude: lon, speed_kmh: speed, heading }),
  });
}

export async function apiUpdateResponderStatus(responderId: string, status: string): Promise<Responder> {
  return request<Responder>(`/responders/${responderId}/status`, {
    method: 'POST',
    body: JSON.stringify({ status }),
  });
}

export async function apiGetResponderAlerts(responderId: string): Promise<any[]> {
  return request<any[]>(`/responders/${responderId}/alerts`);
}

// ============================================================
// MANUAL CITIZEN REPORTS
// ============================================================

export async function apiSubmitReport(report: {
  category: string;
  description: string;
  location_name?: string;
  latitude?: number;
  longitude?: number;
  contact_phone?: string;
}): Promise<ManualReport> {
  return request<ManualReport>('/reports', {
    method: 'POST',
    body: JSON.stringify(report),
  });
}

export async function apiUploadReportImage(reportId: string, file: File): Promise<{ report_id: string; image_path: string }> {
  const token = getToken();
  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch(`${API_BASE}/reports/${reportId}/image`, {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: formData,
  });
  if (!response.ok) {
    throw new Error(`Upload failed: ${response.statusText}`);
  }
  return response.json();
}

export async function apiGetMyReports(): Promise<ManualReport[]> {
  return request<ManualReport[]>('/reports/my');
}

export async function apiGetReport(reportId: string): Promise<ManualReport> {
  return request<ManualReport>(`/reports/${reportId}`);
}

export async function apiAdminListReports(status?: string): Promise<ManualReport[]> {
  const url = status ? `/admin/reports?status=${status}` : '/admin/reports';
  return request<ManualReport[]>(url);
}

export async function apiAdminReviewReport(reportId: string, status: 'APPROVED' | 'REJECTED' | 'ESCALATED', adminNotes?: string, severity = 'HIGH'): Promise<{ report: ManualReport; incident?: Incident }> {
  return request<{ report: ManualReport; incident?: Incident }>(`/admin/reports/${reportId}/review`, {
    method: 'POST',
    body: JSON.stringify({ status, admin_notes: adminNotes, severity }),
  });
}

// ============================================================
// ADMIN MANAGEMENT & SYSTEM
// ============================================================

export async function apiAdminListUsers(role?: string, status?: string): Promise<User[]> {
  const params = new URLSearchParams();
  if (role) params.append('role', role);
  if (status) params.append('status', status);
  return request<User[]>(`/admin/users?${params.toString()}`);
}

export async function apiAdminProvisionUser(data: { email: string; password: string; full_name: string; role: string; phone?: string; responder_id?: string }): Promise<User> {
  return request<User>('/admin/users', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function apiAdminUpdateUserStatus(userId: string, status: 'ACTIVE' | 'SUSPENDED'): Promise<User> {
  return request<User>(`/admin/users/${userId}/status`, {
    method: 'PATCH',
    body: JSON.stringify({ status }),
  });
}

export async function apiAdminAuditLogs(limit = 100): Promise<AuditLog[]> {
  return request<AuditLog[]>(`/admin/audit-logs?limit=${limit}`);
}

export async function apiGetStats(): Promise<SystemStats> {
  return request<SystemStats>('/stats');
}

export async function apiGetSystemHealth(): Promise<any> {
  return request<any>('/system/health');
}

export async function apiGetNotifications(recipient: string): Promise<NotificationItem[]> {
  return request<NotificationItem[]>(`/notifications/${recipient}`);
}

export async function apiGetRoute(fromLat: number, fromLon: number, toLat: number, toLon: number): Promise<any> {
  return request<any>(`/route?from_lat=${fromLat}&from_lon=${fromLon}&to_lat=${toLat}&to_lon=${toLon}`);
}
