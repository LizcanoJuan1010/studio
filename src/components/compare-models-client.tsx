'use client';
import { useState } from 'react';
import { ComparisonChart } from '@/components/charts/comparison-chart';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import type { AllGlobalMetrics, Model } from '@/lib/types';

type MetricOption = {
  value: keyof AllGlobalMetrics['cnn_simple'];
  label: string;
};

const metricOptions: MetricOption[] = [
  { value: 'accuracy', label: 'Accuracy' },
  { value: 'f1_macro', label: 'F1 Score (Macro)' },
];

const modelNames: Record<Model['id'], string> = {
    cnn_simple: "CNN (Simple)",
    cnn_transfer: "CNN (Transfer Learning)",
    svm: "SVM",
    boosting: "XGBoost",
}

export function CompareModelsClient({ metrics }: { metrics: AllGlobalMetrics }) {
  const [selectedMetric, setSelectedMetric] = useState<MetricOption>(metricOptions[0]);

  const handleMetricChange = (value: string) => {
    const metric = metricOptions.find((m) => m.value === value) || metricOptions[0];
    setSelectedMetric(metric);
  };
  
  return (
    <div className="flex flex-col gap-8 animate-fade-in-up">
      <div>
        <h1 className="text-3xl font-bold tracking-tight font-headline">
          Model Comparison
        </h1>
        <p className="text-muted-foreground mt-2">
          Compare the performance of all models using different metrics.
        </p>
      </div>

      <div className="grid gap-6">
        <Card className="animate-fade-in" style={{ animationDelay: '100ms' }}>
          <CardHeader>
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                <CardTitle className="font-headline text-xl">Metric Visualization</CardTitle>
                <div className="w-full md:w-64">
                    <Select onValueChange={handleMetricChange} defaultValue={selectedMetric.value}>
                    <SelectTrigger>
                        <SelectValue placeholder="Select a metric" />
                    </SelectTrigger>
                    <SelectContent>
                        {metricOptions.map((option) => (
                        <SelectItem key={option.value} value={option.value}>
                            {option.label}
                        </SelectItem>
                        ))}
                    </SelectContent>
                    </Select>
                </div>
            </div>
          </CardHeader>
          <CardContent>
            <ComparisonChart
              data={metrics}
              metric={selectedMetric.value}
              metricLabel={selectedMetric.label}
            />
          </CardContent>
        </Card>

        <Card className="animate-fade-in" style={{ animationDelay: '200ms' }}>
          <CardHeader>
            <CardTitle className="font-headline text-xl">Global Metrics Summary</CardTitle>
            <CardDescription>A detailed look at the key performance indicators for each model.</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Model</TableHead>
                  <TableHead className="text-right">Accuracy</TableHead>
                  <TableHead className="text-right">F1 (Macro)</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {Object.entries(metrics).map(([modelId, modelMetrics]) => (
                  <TableRow key={modelId}>
                    <TableCell className="font-medium">{modelNames[modelId as Model['id']]}</TableCell>
                    <TableCell className="text-right">{(modelMetrics.accuracy * 100).toFixed(1)}%</TableCell>
                    <TableCell className="text-right">{modelMetrics.f1_macro.toFixed(3)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
