import { Routes } from '@angular/router';
import { SubmitComponent } from './submit/submit';
import { AboutComponent } from './about/about';
import { ResultsComponent } from './results/results';

export const routes: Routes = [
  { path: 'about', component: AboutComponent },
  { path: 'results', component: ResultsComponent },

  { path: '', component: SubmitComponent },

  { path: '**', redirectTo: '' },
];