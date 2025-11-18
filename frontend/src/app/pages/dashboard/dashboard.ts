import { Component, OnInit, OnDestroy } from '@angular/core';
import { ApiService } from '../../services/api';

@Component({
  selector: 'app-dashboard',
  imports: [],
  templateUrl: './dashboard.html',
  styleUrl: './dashboard.css',
})
export class Dashboard implements OnInit, OnDestroy {
  metrics: any = null;
  loading = true;
  error = '';
  private refreshInterval: any;

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.fetchMetrics();
    this.refreshInterval = setInterval(() => this.fetchMetrics(), 30000);
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
      },
      error: () => {
        this.error = 'Could not reach the API';
        this.loading = false;
      }
    });
  }
}
