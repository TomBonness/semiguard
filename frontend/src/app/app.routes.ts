import { Routes } from '@angular/router';
import { Dashboard } from './pages/dashboard/dashboard';
import { Drift } from './pages/drift/drift';
import { Predict } from './pages/predict/predict';

export const routes: Routes = [
  { path: '', component: Dashboard },
  { path: 'predict', component: Predict },
  { path: 'drift', component: Drift },
];
