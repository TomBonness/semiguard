import { Routes } from '@angular/router';
import { Dashboard } from './pages/dashboard/dashboard';
import { Drift } from './pages/drift/drift';

export const routes: Routes = [
  { path: '', component: Dashboard },
  { path: 'drift', component: Drift },
];
