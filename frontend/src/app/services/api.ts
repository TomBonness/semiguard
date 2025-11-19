import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root',
})
export class ApiService {
  private baseUrl = environment.apiUrl;

  constructor(private http: HttpClient) {}

  getHealth() {
    return this.http.get<any>(`${this.baseUrl}/health`);
  }

  predict(features: number[]) {
    return this.http.post<any>(`${this.baseUrl}/predict`, { features });
  }

  getPredictions(n: number = 50) {
    return this.http.get<any[]>(`${this.baseUrl}/predictions?n=${n}`);
  }

  getMetrics() {
    return this.http.get<any>(`${this.baseUrl}/metrics`);
  }

  getDrift() {
    return this.http.get<any>(`${this.baseUrl}/drift`);
  }

  submitFeedback(predictionId: number, actualLabel: string) {
    return this.http.post<any>(`${this.baseUrl}/feedback`, {
      prediction_id: predictionId,
      actual_label: actualLabel
    });
  }
}
