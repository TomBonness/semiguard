import { Component, OnInit } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { ApiService } from './services/api';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit {
  apiStatus: 'connected' | 'disconnected' | 'checking' = 'checking';

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.checkConnection();
  }

  checkConnection() {
    this.apiStatus = 'checking';
    this.api.getHealth().subscribe({
      next: () => this.apiStatus = 'connected',
      error: () => this.apiStatus = 'disconnected'
    });
  }
}
