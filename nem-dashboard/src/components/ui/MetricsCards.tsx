"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { formatPrice } from "@/lib/utils";

interface MetricCardProps {
  title: string;
  value: string;
  change?: number;
  changeLabel?: string;
  status?: string;
}

function MetricCard({
  title,
  value,
  change,
  changeLabel,
  status,
}: MetricCardProps) {
  const getTrendIcon = () => {
    if (change === undefined) return null;
    if (change > 0) return <TrendingUp className="h-4 w-4 text-red-500" />;
    if (change < 0) return <TrendingDown className="h-4 w-4 text-green-500" />;
    return <Minus className="h-4 w-4 text-gray-500" />;
  };

  const getTrendColor = () => {
    if (change === undefined) return "text-gray-500";
    if (change > 0) return "text-red-500";
    if (change < 0) return "text-green-500";
    return "text-gray-500";
  };

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-baseline justify-between">
          <p className="text-2xl font-bold">{value}</p>
          {status && <Badge variant={status as any}>{status}</Badge>}
        </div>
        {(change !== undefined || changeLabel) && (
          <div
            className={`flex items-center gap-1 mt-2 text-sm ${getTrendColor()}`}
          >
            {getTrendIcon()}
            <span>
              {change !== undefined && `${Math.abs(change).toFixed(1)}%`}
              {changeLabel && ` ${changeLabel}`}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

interface MetricsCardsProps {
  currentPrice: number;
  avgForecast: number;
  maxForecast: number;
  minForecast: number;
  peakTime: string;
}

export function MetricsCards({
  currentPrice,
  avgForecast,
  maxForecast,
  minForecast,
  peakTime,
}: MetricsCardsProps) {
  const changePercent = ((avgForecast - currentPrice) / currentPrice) * 100;

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      <MetricCard
        title="Current Spot Price"
        value={formatPrice(currentPrice)}
        change={changePercent}
        changeLabel="vs forecast avg"
      />
      <MetricCard title="Average Forecast" value={formatPrice(avgForecast)} />
      <MetricCard
        title="Peak Forecast"
        value={formatPrice(maxForecast)}
        changeLabel={new Date(peakTime).getHours() + ":00"}
      />
      <MetricCard title="Low Forecast" value={formatPrice(minForecast)} />
    </div>
  );
}
