import { Component, OnInit, OnDestroy } from '@angular/core';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration } from 'chart.js';
import { Chart, registerables } from 'chart.js';
import { ApiService } from '../../services/api';

Chart.register(...registerables);

@Component({
  selector: 'app-dashboard',
  imports: [BaseChartDirective],
  templateUrl: './dashboard.html',
  styleUrl: './dashboard.css',
})
export class Dashboard implements OnInit, OnDestroy {
  metrics: any = null;
  loading = true;
  error = '';
  private refreshInterval: any;

  // confidence line chart
  confidenceChartData: ChartConfiguration<'line'>['data'] = {
    labels: [],
    datasets: []
  };
  confidenceChartOptions: ChartConfiguration<'line'>['options'] = {
    responsive: true,
    plugins: { legend: { display: true } },
    scales: {
      y: { min: 0, max: 1, title: { display: true, text: 'Confidence' } },
      x: { title: { display: true, text: 'Prediction #' } }
    }
  };

  // pass/fail bar chart
  barChartData: ChartConfiguration<'bar'>['data'] = {
    labels: ['Pass', 'Fail'],
    datasets: []
  };
  barChartOptions: ChartConfiguration<'bar'>['options'] = {
    responsive: true,
    plugins: { legend: { display: false } },
    scales: {
      y: { title: { display: true, text: 'Count' } }
    }
  };

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.fetchMetrics();
    this.fetchPredictions();
    this.refreshInterval = setInterval(() => {
      this.fetchMetrics();
      this.fetchPredictions();
    }, 30000);
  }

  ngOnDestroy() {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
    }
  }

  fetchMetrics() {
    this.api.getMetrics().subscribe({
      next: (data) => {
        this.metrics = data;
        this.loading = false;
        this.error = '';

        this.barChartData = {
          labels: ['Pass', 'Fail'],
          datasets: [{
            data: [data.pass_count, data.fail_count],
            backgroundColor: ['#4caf50', '#ef5350']
          }]
        };
      },
      error: () => {
        this.error = 'Could not reach the API';
        this.loading = false;
      }
    });
  }

  fetchPredictions() {
    this.api.getPredictions(50).subscribe({
      next: (rows) => {
        // reverse so oldest is first (they come in desc order)
        const sorted = [...rows].reverse();
        const labels = sorted.map((_, i) => `#${i + 1}`);
        const passConf = sorted.map(r => r.prediction === 'pass' ? r.confidence : null);
        const failConf = sorted.map(r => r.prediction === 'fail' ? r.confidence : null);

        this.confidenceChartData = {
          labels,
          datasets: [
            {
              label: 'Pass',
              data: passConf,
              borderColor: '#4caf50',
              backgroundColor: 'rgba(76, 175, 80, 0.1)',
              pointRadius: 4,
              spanGaps: false
            },
            {
              label: 'Fail',
              data: failConf,
              borderColor: '#ef5350',
              backgroundColor: 'rgba(239, 83, 80, 0.1)',
              pointRadius: 4,
              spanGaps: false
            }
          ]
        };
      }
    });
  }
}
