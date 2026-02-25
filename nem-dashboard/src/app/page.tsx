"use client";

import { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, AlertTriangle, RefreshCw } from "lucide-react";
import { fetchForecast, ForecastResponse } from "@/lib/api";
import { ForecastChart } from "@/components/ui/ForecastChart";
import { MetricsCards } from "@/components/ui/MetricsCards";
import { formatDateTime } from "@/lib/utils";

export default function Dashboard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ForecastResponse | null>(null);

  const [horizonHours, setHorizonHours] = useState(24);
  const [confidenceLevel, setConfidenceLevel] = useState(0.9);
  const [showConfidence, setShowConfidence] = useState(true);
  const [highThreshold, setHighThreshold] = useState(300);
  const [lowThreshold, setLowThreshold] = useState(50);

  const loadForecast = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetchForecast(horizonHours, confidenceLevel);
      setData(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load forecast");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadForecast();
  }, [horizonHours, confidenceLevel]);

  if (loading && !data) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto p-8">
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
        <Button onClick={loadForecast} className="mt-4">
          <RefreshCw className="mr-2 h-4 w-4" />
          Retry
        </Button>
      </div>
    );
  }

  if (!data) return null;

  const forecasts = data.forecasts;
  const avgForecast =
    forecasts.reduce((sum, f) => sum + f.forecast, 0) / forecasts.length;
  const maxForecast = Math.max(...forecasts.map((f) => f.forecast));
  const minForecast = Math.min(...forecasts.map((f) => f.forecast));
  const peakTime =
    forecasts.find((f) => f.forecast === maxForecast)?.timestamp || "";

  const highPriceCount = forecasts.filter(
    (f) => f.forecast > highThreshold,
  ).length;

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-slate-900 to-slate-800 text-white">
        <div className="container mx-auto px-8 py-6">
          <h1 className="text-3xl font-bold">NEM Pre-Dispatch Dashboard</h1>
          <p className="text-slate-300 mt-2">
            National Electricity Market - Real-time Forecasting & Analytics
          </p>
          <p className="text-sm text-slate-400 mt-1">
            Last Update: {formatDateTime(data.last_update)}
          </p>
        </div>
      </header>

      <div className="container mx-auto px-8 py-8">
        {/* Alert if high prices expected */}
        {highPriceCount > 0 && (
          <Alert className="mb-6 border-amber-500 bg-amber-50">
            <AlertTriangle className="h-4 w-4 text-amber-600" />
            <AlertDescription className="text-amber-900">
              <strong>High Price Warning:</strong> {highPriceCount} intervals
              forecast to exceed ${highThreshold}/MWh threshold.
            </AlertDescription>
          </Alert>
        )}

        {/* Metrics */}
        <MetricsCards
          currentPrice={data.current_price}
          avgForecast={avgForecast}
          maxForecast={maxForecast}
          minForecast={minForecast}
          peakTime={peakTime}
        />

        {/* Controls */}
        <Card className="my-6">
          <CardHeader>
            <CardTitle>Forecast Settings</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div>
              <label className="text-sm font-medium">
                Forecast Horizon: {horizonHours} hours
              </label>
              <Slider
                value={[horizonHours]}
                onValueChange={(value) => setHorizonHours(value[0])}
                min={12}
                max={72}
                step={6}
                className="mt-2"
              />
            </div>

            <div>
              <label className="text-sm font-medium">
                Confidence Level: {(confidenceLevel * 100).toFixed(0)}%
              </label>
              <Slider
                value={[confidenceLevel * 100]}
                onValueChange={(value) => setConfidenceLevel(value[0] / 100)}
                min={80}
                max={95}
                step={5}
                className="mt-2"
              />
            </div>

            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="showConfidence"
                checked={showConfidence}
                onChange={(e) => setShowConfidence(e.target.checked)}
                className="rounded"
              />
              <label htmlFor="showConfidence" className="text-sm">
                Show Confidence Intervals
              </label>
            </div>
          </CardContent>
        </Card>

        {/* Main Chart */}
        <ForecastChart
          forecasts={forecasts}
          showConfidence={showConfidence}
          highThreshold={highThreshold}
          lowThreshold={lowThreshold}
        />

        {/* Tabs for additional analysis */}
        <Tabs defaultValue="statistics" className="mt-6">
          <TabsList>
            <TabsTrigger value="statistics">Statistics</TabsTrigger>
            <TabsTrigger value="data">Data Table</TabsTrigger>
            <TabsTrigger value="insights">Insights</TabsTrigger>
          </TabsList>

          <TabsContent value="statistics">
            <Card>
              <CardHeader>
                <CardTitle>Forecast Statistics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Mean</p>
                    <p className="text-2xl font-bold">
                      ${avgForecast.toFixed(2)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Max</p>
                    <p className="text-2xl font-bold">
                      ${maxForecast.toFixed(2)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Min</p>
                    <p className="text-2xl font-bold">
                      ${minForecast.toFixed(2)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Range</p>
                    <p className="text-2xl font-bold">
                      ${(maxForecast - minForecast).toFixed(2)}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="data">
            <Card>
              <CardHeader>
                <CardTitle>Forecast Data</CardTitle>
                <CardDescription>
                  Detailed forecast with confidence intervals
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-auto max-h-96">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-100 sticky top-0">
                      <tr>
                        <th className="p-2 text-left">Timestamp</th>
                        <th className="p-2 text-right">Forecast</th>
                        <th className="p-2 text-right">Lower Bound</th>
                        <th className="p-2 text-right">Upper Bound</th>
                      </tr>
                    </thead>
                    <tbody>
                      {forecasts.map((f, i) => (
                        <tr key={i} className="border-b hover:bg-slate-50">
                          <td className="p-2">{formatDateTime(f.timestamp)}</td>
                          <td className="p-2 text-right font-medium">
                            ${f.forecast.toFixed(2)}
                          </td>
                          <td className="p-2 text-right text-muted-foreground">
                            ${f.lower_bound.toFixed(2)}
                          </td>
                          <td className="p-2 text-right text-muted-foreground">
                            ${f.upper_bound.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="insights">
            <Card>
              <CardHeader>
                <CardTitle>Key Insights</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                  <h4 className="font-semibold text-blue-900">Price Trend</h4>
                  <p className="text-sm text-blue-800 mt-2">
                    {avgForecast > data.current_price
                      ? `Prices expected to increase by ${(((avgForecast - data.current_price) / data.current_price) * 100).toFixed(1)}% on average`
                      : `Prices expected to decrease by ${(((data.current_price - avgForecast) / data.current_price) * 100).toFixed(1)}% on average`}
                  </p>
                </div>

                {highPriceCount > 0 && (
                  <div className="p-4 bg-amber-50 rounded-lg border border-amber-200">
                    <h4 className="font-semibold text-amber-900">
                      High Price Alert
                    </h4>
                    <p className="text-sm text-amber-800 mt-2">
                      {highPriceCount} time intervals are forecast to exceed the
                      ${highThreshold}/MWh threshold. Consider demand response
                      strategies during these periods.
                    </p>
                  </div>
                )}

                <div className="p-4 bg-slate-50 rounded-lg border border-slate-200">
                  <h4 className="font-semibold text-slate-900">
                    Model Information
                  </h4>
                  <p className="text-sm text-slate-700 mt-2">
                    Forecast generated using XGBoost with{" "}
                    {data.metadata.model_features} features based on{" "}
                    {data.metadata.data_points} historical data points.
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Footer */}
      <footer className="bg-slate-900 text-slate-400 py-6 mt-12">
        <div className="container mx-auto px-8 text-center text-sm">
          <p>Dashboard powered by Next.js | Data source: AEMO NEM | 2026</p>
          <p className="mt-2">
            Forecasts are estimates only - Not financial advice
          </p>
        </div>
      </footer>
    </div>
  );
}
