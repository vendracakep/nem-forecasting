"use client";

import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
  Area,
  AreaChart,
} from "recharts";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ForecastPoint } from "@/lib/api";
import { formatPrice, formatDateTime } from "@/lib/utils";

interface ForecastChartProps {
  forecasts: ForecastPoint[];
  showConfidence: boolean;
  highThreshold: number;
  lowThreshold: number;
}

export function ForecastChart({
  forecasts,
  showConfidence,
  highThreshold,
  lowThreshold,
}: ForecastChartProps) {
  const chartData = forecasts.map((f) => ({
    time: new Date(f.timestamp).getTime(),
    timeLabel: formatDateTime(f.timestamp),
    forecast: f.forecast,
    lower: f.lower_bound,
    upper: f.upper_bound,
    high: highThreshold,
    low: lowThreshold,
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Price Forecast Trend</CardTitle>
        <CardDescription>Forecast with confidence intervals</CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={400}>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="time"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(time) => {
                const date = new Date(time);
                return `${date.getHours()}:00`;
              }}
              stroke="#64748b"
            />
            <YAxis
              label={{
                value: "Price ($/MWh)",
                angle: -90,
                position: "insideLeft",
              }}
              stroke="#64748b"
            />
            <Tooltip
              labelFormatter={(time) =>
                formatDateTime(new Date(time as number).toISOString())
              }
              formatter={(value) =>
                typeof value === "number" ? formatPrice(value) : value
              }
            />

            <Legend />

            {showConfidence && (
              <>
                <Area
                  type="monotone"
                  dataKey="upper"
                  stroke="none"
                  fill="#3b82f6"
                  fillOpacity={0.1}
                  name="Upper Bound"
                />
                <Area
                  type="monotone"
                  dataKey="lower"
                  stroke="none"
                  fill="#3b82f6"
                  fillOpacity={0.1}
                  name="Lower Bound"
                />
              </>
            )}

            <Line
              type="monotone"
              dataKey="forecast"
              stroke="#3b82f6"
              strokeWidth={3}
              dot={false}
              name="Forecast"
            />

            <Line
              type="monotone"
              dataKey="high"
              stroke="#ef4444"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              name="High Alert"
            />

            <Line
              type="monotone"
              dataKey="low"
              stroke="#10b981"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              name="Low Alert"
            />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
