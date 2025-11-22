import { Component, OnInit } from '@angular/core';
import { DatePipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api';

@Component({
  selector: 'app-history',
  imports: [FormsModule, DatePipe],
  templateUrl: './history.html',
  styleUrl: './history.css',
})
export class History implements OnInit {
  predictions: any[] = [];
  loading = true;
  error = '';

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.fetchPredictions();
  }

  fetchPredictions() {
    this.loading = true;
    this.api.getPredictions(100).subscribe({
      next: (rows) => {
        this.predictions = rows;
        this.loading = false;
      },
      error: () => {
        this.error = 'Could not load predictions';
        this.loading = false;
      }
    });
  }

  onFeedbackChange(prediction: any, value: string) {
    if (value === 'unknown') return;

    this.api.submitFeedback(prediction.id, value).subscribe({
      next: () => {
        prediction.actual_label = value;
      },
      error: () => {
        // reset on failure
        prediction.actual_label = null;
      }
    });
  }
}
