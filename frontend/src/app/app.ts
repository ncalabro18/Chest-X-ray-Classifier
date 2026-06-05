import { Component, OnInit, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';

import { MastheadComponent } from './masthead/masthead';
import { PerClassRow, PerEpochRow, ModelMetadata } from './results/results';
import { HttpClient } from '@angular/common/http';
@Component({
  selector: 'app-root',
  imports: [RouterOutlet, MastheadComponent],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {
  protected readonly title = signal('frontend');

  metadata: ModelMetadata = {
    runName: 'swin_v2_base_run_01',
    architecture: 'Swin Transformer V2 Base',
    pretraining: 'ImageNet-22k → NIH ChestX-ray14',
    dataset: 'NIH ChestX-ray14',
    bestEpoch: 42,
    checkpointFile: 'best_epoch_042.pth',
    trainedAt: '2026-05-15',
    totalImages: 112120,
    trainSize: 86524,
    valSize: 25596,
  };

  perClassRows: PerClassRow[] = [];
  perEpochRows: PerEpochRow[] = [];

  constructor(private http: HttpClient) {}

}
