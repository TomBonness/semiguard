import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api';

@Component({
  selector: 'app-predict',
  imports: [FormsModule],
  templateUrl: './predict.html',
  styleUrl: './predict.css',
})
export class Predict {
  featuresJson = '';
  result: any = null;
  loading = false;
  error = '';
  loadingSample = false;

  constructor(private api: ApiService) {}

  loadSample() {
    this.loadingSample = true;
    this.api.getSample().subscribe({
      next: (data) => {
        this.featuresJson = JSON.stringify(data.features);
        this.loadingSample = false;
        this.error = '';
      },
      error: () => {
        this.error = 'Could not load sample data';
        this.loadingSample = false;
      }
    });
  }

  submitPrediction() {
    this.error = '';
    this.result = null;

    let features: number[];
    try {
      features = JSON.parse(this.featuresJson);
      if (!Array.isArray(features)) throw new Error();
    } catch {
      this.error = 'Invalid JSON - paste an array of numbers';
      return;
    }

    this.loading = true;
    this.api.predict(features).subscribe({
      next: (data) => {
        this.result = data;
        this.loading = false;
      },
      error: (err) => {
        this.error = err.error?.error || 'Prediction failed';
        this.loading = false;
      }
    });
  }
}
