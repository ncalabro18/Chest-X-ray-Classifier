import { Injectable, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';

export type ServerStatus =
  | 'starting'
  | 'ready'
  | 'busy'
  | 'unknown'
  | 'error';

export type SubmitState =
  | 'idle'
  | 'loading'
  | 'success'
  | 'error';

@Injectable({
  providedIn: 'root'
})
export class ClassifierStateService {

  private readonly STATUS_URL = '/status';

  serverStatus = signal<ServerStatus>('unknown');

  submitState = signal<SubmitState>('idle');

  sidebarOpen = signal(true);

  constructor(private http: HttpClient) {}

  toggleSidebar(): void {
    this.sidebarOpen.update(v => !v);
  }


  pollStatus(): void {
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
    switch (this.submitState()) {
      case 'loading': return 'Processing';
      case 'success': return 'Complete';
      case 'error':   return 'Submission failed';
      default:        return 'Idle';
    }
  }
}