import { Component, Input } from '@angular/core';
import { PerClassRow } from '../csv-loader.service';

@Component({
  selector: 'app-operating-point-tooltip',
  imports: [],
  templateUrl: './operating-point-tooltip.html',
  styleUrl: './operating-point-tooltip.scss',
})
export class OperatingPointTooltip {

  @Input() row: PerClassRow | null = null;

  pct(v: number | undefined): string {
    return v == null || isNaN(v) ? '—' : (v * 100).toFixed(1) + '%';
  }
}
