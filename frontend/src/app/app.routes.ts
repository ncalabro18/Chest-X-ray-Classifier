import { Routes } from '@angular/router';
import { SubmitComponent } from './submit/submit';
import { AboutComponent } from './about/about';

export const routes: Routes = [
  { path: 'about', component: AboutComponent },
  { path: '', component: SubmitComponent },

  { path: '**', redirectTo: '' },
];