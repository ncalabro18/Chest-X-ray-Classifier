import { Component, inject } from '@angular/core';
import { Router, RouterLink } from '@angular/router';
import { CommonModule } from '@angular/common';
import { ClassifierStateService } from '../classifier-state';

@Component({
  selector: 'app-masthead',
  templateUrl: './masthead.html',
  styleUrl: './masthead.scss',
  imports: [
    CommonModule, RouterLink
  ]
})
export class MastheadComponent {

  constructor(
    public router: Router,
    
    public state: ClassifierStateService
  ) {}

  get currentUrl(): string {
    return this.router.url;
  }
}