const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ForecastPoint {
  timestamp: string;
  forecast: number;
  lower_bound: number;
  upper_bound: number;
  uncertainty: number;
}

export interface ForecastResponse {
  forecasts: ForecastPoint[];
  current_price: number;
  last_update: string;
  metadata: {
    horizon_hours: number;
    confidence_level: number;
    data_points: number;
    model_features: number;
  };
}

export interface ForecastRequest {
  horizon_hours: number;
  confidence_level: number;
}

export async function fetchForecast(
  horizonHours: number = 24,
  confidenceLevel: number = 0.9,
): Promise<ForecastResponse> {
  const response = await fetch(`${API_BASE_URL}/forecast`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      horizon_hours: horizonHours,
      confidence_level: confidenceLevel,
    }),
  });

  if (!response.ok) {
    throw new Error(`API Error: ${response.statusText}`);
  }

  return response.json();
}

export async function fetchCurrentPrice(): Promise<{
  price: number;
  timestamp: string;
}> {
  const response = await fetch(`${API_BASE_URL}/current`);

  if (!response.ok) {
    throw new Error(`API Error: ${response.statusText}`);
  }

  return response.json();
}

export async function checkHealth(): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/`);

  if (!response.ok) {
    throw new Error(`API Error: ${response.statusText}`);
  }

  return response.json();
}
