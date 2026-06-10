/*
© 2026 Nicholas J. Calabro. All rights reserved.
*/
import {
  Component, signal, computed, ElementRef,
  ViewChild, OnInit, OnDestroy,
} from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { Subscription, interval } from 'rxjs';
import { ClassifierStateService } from '../classifier-state';


// Types

// export type ServerStatus = 'starting' | 'ready' | 'busy' | 'unknown' | 'error';
export type ViewPosition = 'PA' | 'AP';
// export type SubmitState = 'idle' | 'loading' | 'success' | 'error';

export interface Prediction {
  probability: number;
  threshold: number;
  spec_threshold: number;
  positive: boolean;
}

export interface ClassifierPayload {
  predictions: Record<string, Prediction>;
  view: ViewPosition;
  attention_maps?: Record<string, string>;
  saliency_maps?:  Record<string, string>;
}

export interface SubmitResponse {
  status: string;
  view: ViewPosition;
  classifier_status: number;
  classifier_response: ClassifierPayload;
}


// Component

@Component({
  selector: 'app-submit',
  standalone: true,
  imports: [CommonModule, HttpClientModule],
  templateUrl: './submit.html',
  styleUrl: './submit.scss',
})
export class SubmitComponent implements OnInit, OnDestroy {
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;

  private readonly SUBMIT_URL = '/submit';

  private statusPoll?: Subscription;

  imagePreviewUrl = signal<string | null>(null);


  highConfidenceFindings = computed(() =>
    this.positiveFindings().filter(
      ([, p]) => p.probability >= p.spec_threshold
    )
  );
  isHighConfidence(p: Prediction): boolean {
    return p.probability >= p.spec_threshold;
  }

  // State
  errorMessage = signal<string | null>(null);
  result       = signal<ClassifierPayload | null>(null);
  dragOver     = signal(false);
  selectedFile = signal<File | null>(null);
  selectedView = signal<ViewPosition>('PA');


  attention_maps = computed(() => this.result()?.attention_maps ?? {});

  selectedAttentionClass = signal<string | null>(null);

  saliency_maps = computed(() => this.result()?.saliency_maps ?? {});

  // 'attention' | 'saliency' - which map type is shown
  activeMapType = signal<'attention' | 'saliency'>('attention');

  setMapType(t: 'attention' | 'saliency'): void {
    this.activeMapType.set(t);
  }

  // Replace the existing activeAttentionMap computed:
  activeMap = computed(() => {
    const cls  = this.selectedAttentionClass() ?? this.positiveClassesWithMaps()[0];
    const maps = this.activeMapType() === 'saliency'
      ? this.saliency_maps()
      : this.attention_maps();
    return cls ? (maps[cls] ?? null) : null;
  });

  // Update positiveClassesWithMaps to union both map sets:
  positiveClassesWithMaps = computed(() =>
    this.positiveFindings()
      .map(([name]) => name)
      .filter(name => !!this.attention_maps()[name] || !!this.saliency_maps()[name])
  );

  setAttentionClass(cls: string): void {
    this.selectedAttentionClass.set(cls);
  }


  // Derived
  positiveFindings = computed(() =>
    Object.entries(this.result()?.predictions ?? {})
      .filter(([, p]) => p.positive)
      .sort(([, a], [, b]) => b.probability - a.probability)
  );

  negativeFindings = computed(() =>
    Object.entries(this.result()?.predictions ?? {})
      .filter(([, p]) => !p.positive)
      .sort(([, a], [, b]) => b.probability - a.probability)
  );

  positiveCount = computed(() => this.positiveFindings().length);

  // Excludes "No Finding" - used for the count shown to the clinician
  pathologyCount = computed(() =>
    this.positiveFindings().filter(([name]) => name !== 'No Finding').length
  );

  // True only when at least one pathology (not just No Finding) is positive
  hasPathologyPositives = computed(() => this.pathologyCount() > 0);

  constructor(
    private http: HttpClient,
    public stateService: ClassifierStateService
  ) {}
  // File handling
  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.dragOver.set(true);
  }

  onDragLeave(): void {
    this.dragOver.set(false);
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragOver.set(false);
    const file = event.dataTransfer?.files?.[0];
    if (file) this.setFile(file);
  }

  toggleSidebar(): void {
    this.stateService.sidebarOpen.update(v => !v);
  }

  onFileChange(event: Event): void {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (file) this.setFile(file);
  }

  setView(v: ViewPosition): void {
    this.selectedView.set(v);
  }

  // Submit
  submit(): void {
    const file = this.selectedFile();
    if (!file || this.stateService.submitState() === 'loading') return;

    const form = new FormData();
    form.append('file', file, file.name);
    form.append('view', this.selectedView());

    this.errorMessage.set(null);
    this.result.set(null);

    this.stateService.submitState.set("loading")
    this.http.post<SubmitResponse>(this.SUBMIT_URL, form).subscribe({
      next: (res) => {
        this.result.set(res.classifier_response);
        this.stateService.sidebarOpen.set(false);
        this.stateService.submitState.set("success")
      },
      error: (err: HttpErrorResponse) => {
        this.stateService.submitState.set("error")

        this.errorMessage.set(
          err.error?.detail ?? 'Server error - check classifier logs.');
      },
    });
  }


  ngOnInit(): void {
    this.stateService.pollStatus();
    this.statusPoll = interval(
      8000).subscribe(() => this.stateService.pollStatus());
  }



  // Add signal

  // Update setFile()
  private setFile(file: File): void {
    // Revoke the old object URL to avoid memory leaks
    const prev = this.imagePreviewUrl();
    if (prev) URL.revokeObjectURL(prev);

    this.selectedFile.set(file);
    this.imagePreviewUrl.set(URL.createObjectURL(file));
    this.result.set(null);
    this.errorMessage.set(null);
    this.stateService.submitState.set('idle');
  }

  comparisonOverlay = signal<'attention' | 'gradcam' | null>(null);

  setComparison(mode: 'attention' | 'gradcam' | null): void {
    this.comparisonOverlay.set(mode);
  }

  reset(): void {
    const prev = this.imagePreviewUrl();
    if (prev) URL.revokeObjectURL(prev);
    this.imagePreviewUrl.set(null);

    this.stateService.submitState.set('idle');
    this.selectedFile.set(null);
    this.activeMapType.set('attention');
    this.selectedAttentionClass.set(null);
    this.result.set(null);
    this.errorMessage.set(null);
    this.selectedView.set('PA');
    if (this.fileInput?.nativeElement) {
      this.fileInput.nativeElement.value = '';
    }
     this.stateService.sidebarOpen.set(true);  
  }

  // Revoke on destroy too
  ngOnDestroy(): void {
    this.statusPoll?.unsubscribe();
    const prev = this.imagePreviewUrl();
    if (prev) URL.revokeObjectURL(prev);
  }

  // Helpers
  formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  }

  pct(p: number): string {
    return (p * 100).toFixed(1) + '%';
  }

  barWidth(p: number): string {
    return Math.round(p * 100) + '%';
  }

  thresholdLeft(t: number): string {
    return Math.round(t * 100) + '%';
  }

  //attentionMap = computed(() => this.result()?.attention_map ?? null);
}