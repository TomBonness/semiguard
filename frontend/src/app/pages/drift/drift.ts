import { Component, OnInit } from '@angular/core';
import { ApiService } from '../../services/api';

@Component({
  selector: 'app-drift',
  imports: [],
  templateUrl: './drift.html',
  styleUrl: './drift.css',
})
export class Drift implements OnInit {
  driftData: any = null;
  loading = true;
  error = '';

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.fetchDrift();
  }

  fetchDrift() {
    this.loading = true;
    this.api.getDrift().subscribe({
      next: (data) => {
        this.driftData = data;
        this.loading = false;
        this.error = '';
      },
      error: (err) => {
        this.error = err.error?.error || 'Could not reach the API';
        this.loading = false;
      }
    });
  }

  get driftPct(): string {
    return ((this.driftData?.drift_score || 0) * 100).toFixed(1);
  }
}
