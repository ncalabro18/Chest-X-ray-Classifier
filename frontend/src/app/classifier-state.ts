import { Injectable, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { ModelMetadata } from './csv-loader.service';

export type ServerStatus = 'starting' | 'ready' | 'busy' | 'unknown' | 'error';
export type SubmitState  = 'idle' | 'loading' | 'success' | 'error';

@Injectable({ providedIn: 'root' })
export class ClassifierStateService {
  private readonly STATUS_URL   = '/status';
  private readonly METADATA_URL = 'metadata.json';

  public serverStatus = signal<ServerStatus>('unknown');
  public submitState  = signal<SubmitState>('idle');
  public sidebarOpen  = signal(true);
  public metadata     = signal<ModelMetadata | null>(null);

  constructor(private http: HttpClient) {
    this.loadMetadata();
  }

  private async loadMetadata(): Promise<void> {
    try {
      const meta = await fetch(this.METADATA_URL).then(r => r.json());
      this.metadata.set(meta);
    } catch {
      // tooltip epoch-filtering degrades gracefully — getClosestRow returns null
    }
  }

  toggleSidebar(): void { this.sidebarOpen.update(v => !v); }

  pollStatus(): void {
    this.http.get<{ state: ServerStatus }>(this.STATUS_URL).subscribe({
      next:  (res) => this.serverStatus.set(res.state),
      error: ()    => this.serverStatus.set('unknown'),
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
    switch (this.submitState()) {
      case 'loading': return 'Processing';
      case 'success': return 'Complete';
      case 'error':   return 'Submission failed';
      default:        return 'Idle';
    }
  }
}