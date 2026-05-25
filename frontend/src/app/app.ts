import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';

import { MastheadComponent } from './masthead/masthead';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, MastheadComponent],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {
  protected readonly title = signal('frontend');

}
