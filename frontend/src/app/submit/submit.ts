import {
  Component, signal, computed, ElementRef,
  ViewChild, OnInit, OnDestroy,
} from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { Subscription, interval } from 'rxjs';
import { RouterLink } from '@angular/router';


// Types

export type ServerStatus = 'starting' | 'ready' | 'busy' | 'unknown' | 'error';
export type ViewPosition = 'PA' | 'AP';
export type SubmitState = 'idle' | 'loading' | 'success' | 'error';

export interface Prediction {
  probability: number;
  threshold: number;
  positive: boolean;
}

export interface ClassifierPayload {
  predictions: Record<string, Prediction>;
  view: ViewPosition;
  attention_map?: string;
  grad_cam?: string;
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
  imports: [CommonModule, HttpClientModule, RouterLink],
  templateUrl: './submit.html',
  styleUrl: './submit.scss',
})
export class SubmitComponent implements OnInit, OnDestroy {
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;

  private readonly SUBMIT_URL = '/submit';
  private readonly STATUS_URL = '/status';

  private statusPoll?: Subscription;

  imagePreviewUrl = signal<string | null>(null);
  sidebarOpen = signal(true);


  // State
  state        = signal<SubmitState>('idle');
  errorMessage = signal<string | null>(null);
  result       = signal<ClassifierPayload | null>(null);
  dragOver     = signal(false);
  selectedFile = signal<File | null>(null);
  selectedView = signal<ViewPosition>('PA');
  serverStatus = signal<ServerStatus>('unknown');


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

  constructor(private http: HttpClient) {}

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
    this.sidebarOpen.update(v => !v);
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
    if (!file || this.state() === 'loading') return;

    const form = new FormData();
    form.append('file', file, file.name);
    form.append('view', this.selectedView());

    this.state.set('loading');
    this.errorMessage.set(null);
    this.result.set(null);

    this.http.post<SubmitResponse>(this.SUBMIT_URL, form).subscribe({
      next: (res) => {
        this.result.set(res.classifier_response);
        this.state.set('success');
        this.sidebarOpen.set(false);
      },
      error: (err: HttpErrorResponse) => {
        this.state.set('error');
        this.errorMessage.set(err.error?.detail ?? 'Server error — check classifier logs.');
      },
    });
  }


  ngOnInit(): void {
    this.pollStatus();
    this.statusPoll = interval(8000).subscribe(() => this.pollStatus());
  }

  private pollStatus(): void {
    this.http.get<{ state: ServerStatus }>(this.STATUS_URL).subscribe({
      next: (res) => {
        this.serverStatus.set(res.state);
      },
      error: () => {
        this.serverStatus.set('unknown');
      },
    });
  }

  serverIndicatorLabel(): string {
    switch (this.serverStatus()) {
      case 'starting': return 'Server starting';
      case 'busy':     return 'Server busy';
      case 'ready':    return 'Server ready';
      case 'error':    return 'Server error';
      default:         return 'Server offline';
    }
  }

  submissionIndicatorLabel(): string {
    switch (this.state()) {
      case 'loading': return 'Processing';
      case 'success': return 'Complete';
      case 'error':   return 'Submission failed';
      default:        return 'Idle';
    }
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
    this.state.set('idle');
  }

  comparisonOverlay = signal<'attention' | 'gradcam' | null>(null);

  setComparison(mode: 'attention' | 'gradcam' | null): void {
    this.comparisonOverlay.set(mode);
  }

  reset(): void {
    const prev = this.imagePreviewUrl();
    if (prev) URL.revokeObjectURL(prev);
    this.imagePreviewUrl.set(null);

    this.state.set('idle');
    this.selectedFile.set(null);
    this.result.set(null);
    this.errorMessage.set(null);
    this.selectedView.set('PA');
    if (this.fileInput?.nativeElement) {
      this.fileInput.nativeElement.value = '';
    }
     this.sidebarOpen.set(true);  
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

  attentionMap = computed(() => this.result()?.attention_map ?? null);
  gradCam      = computed(() => this.result()?.grad_cam      ?? null);  
}