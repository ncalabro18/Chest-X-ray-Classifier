import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

export interface ArchStage {
  label: string;
  depth: number;
  heads: number;
  resolution: string;
  channels: number;
  note?: string;
}

export interface Pipeline {
  step: string;
  detail: string;
  tag?: 'train' | 'inference' | 'both';
}

@Component({
  selector: 'app-about',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './about.html',
  styleUrl: './about.scss',
})
export class AboutComponent {

  readonly backbone: ArchStage[] = [
    { label: 'Stage 1', depth: 2,  heads: 3,  resolution: '96 × 96', channels: 96,  note: 'Patch embedding + early texture features' },
    { label: 'Stage 2', depth: 2,  heads: 6,  resolution: '48 × 48', channels: 192, note: 'Mid-level structural patterns' },
    { label: 'Stage 3', depth: 18, heads: 12, resolution: '24 × 24', channels: 384, note: 'Deep semantic representations — 18 transformer blocks' },
    { label: 'Stage 4', depth: 2,  heads: 24, resolution: '12 × 12', channels: 768, note: 'Global context, final feature map' },
  ];

  readonly pipeline: Pipeline[] = [
    { step: 'Patch embedding',          detail: '4 × 4 non-overlapping patches, projected to 96-dim tokens',                          tag: 'both' },
    { step: 'Multi-scale fusion',       detail: 'Stages 1–3 projected to 768-dim via Conv2d bottleneck, gated and pooled to 12 × 12', tag: 'both' },
    { step: 'Class query attention',    detail: '15 learnable class queries attend over all concatenated scale tokens simultaneously',  tag: 'both' },
    { step: 'View conditioning',        detail: 'PA/AP embedding passed through MLP to produce per-class scale and shift (FiLM)',      tag: 'both' },
    { step: 'Classification head',      detail: 'LayerNorm → Dropout(0.2) → Linear(768→256) → GELU → Dropout(0.1) → Linear(256→1)',   tag: 'both' },
    { step: 'Asymmetric loss',          detail: 'γ⁺=1.0  γ⁻=5.0  clip=0.10  label_smooth=0.05 — penalises false negatives harder',   tag: 'train' },
    { step: 'EMA model',                detail: 'Exponential moving average of weights (decay=0.999) used for all evaluation and inference', tag: 'train' },
    { step: 'Gradual unfreezing',       detail: 'Backbone frozen initially; stages unlocked at epochs 5, 10, 15, 20 with LR warmup',   tag: 'train' },
    { step: 'Test-time augmentation',   detail: 'Original · horizontal flip · CLAHE · CLAHE + flip — logits averaged before sigmoid', tag: 'inference' },
    { step: 'Temperature calibration',  detail: 'Per-class temperature scaling fitted on held-out calibration set post-training',      tag: 'inference' },
    { step: 'Threshold optimisation',   detail: 'Per-class thresholds fitted to maximise 2·TPR − FPR (weighted Youden\'s J)',          tag: 'inference' },
  ];

  readonly classes: string[] = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Effusion', 'Emphysema', 'Fibrosis', 'Infiltration',
    'Mass', 'No Finding', 'Nodule', 'Pleural Thickening',
    'Pneumonia', 'Pneumothorax',
  ];

  tagLabel(tag?: string): string {
    if (tag === 'train')     return 'Training';
    if (tag === 'inference') return 'Inference';
    return 'Both';
  }
}