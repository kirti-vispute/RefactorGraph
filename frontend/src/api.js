const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

// Thrown for both HTTP-error responses and network failures so callers can
// branch on a single error type; `.status` is null for network errors.
export class ApiError extends Error {
  constructor(message, status = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export async function analyzeCode(source, filename) {
  let res
  try {
    res = await fetch(`${API_BASE}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source, filename }),
    })
  } catch {
    throw new ApiError('Could not reach the analysis server. Is the backend running?')
  }

  if (!res.ok) {
    let detail = `Analysis failed (HTTP ${res.status}).`
    try {
      const body = await res.json()
      if (body?.detail) detail = body.detail
    } catch {
      // response body wasn't JSON; keep the generic message
    }
    throw new ApiError(detail, res.status)
  }

  return res.json()
}
