import {
  Component,
  ChangeDetectionStrategy,
  signal,
  computed,
  AfterViewInit,
  Input,
  effect,
} from '@angular/core';
import * as d3 from 'd3';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { CsvLoaderService, PerClassRow, PerEpochRow, ModelMetadata } from '../csv-loader.service';

export type PriorityMetric = 'auc' | 'f1' | 'sens' | 'spec' | 'ppv' | 'npv' | 'alert_rate';

export const CLASS_ORDER: string[] = [
  'Pneumothorax', 'Consolidation', 'Pneumonia', 'Edema', 'Effusion',
  'Cardiomegaly', 'Atelectasis', 'Infiltration', 'Mass', 'Nodule',
  'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding',
];

export type ClassKey = 'TP' | 'FP' | 'TN' | 'FN';
export const CLASS_LABELS: Record<ClassKey, string> = {
  TP: 'True Positive',
  FP: 'False Positive',
  TN: 'True Negative',
  FN: 'False Negative',
};

export const CLINICAL_GROUPS: { label: string; classes: string[] }[] = [
  { label: 'Critical / Acute', classes: ['Pneumothorax', 'Consolidation', 'Pneumonia', 'Edema', 'Effusion'] },
  { label: 'Significant / Subacute', classes: ['Cardiomegaly', 'Atelectasis', 'Infiltration', 'Mass', 'Nodule'] },
  { label: 'Chronic / Incidental', classes: ['Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding'] },
];

function scalarToGrayHex(t: number): string {
  // clamp 0–1
  const v = Math.max(0, Math.min(1, t));
  // 0 -> 255 (white), 1 -> 0 (black)
  const c = Math.round(255 * (1 - v));
  const hex = c.toString(16).padStart(2, '0');
  return `#${hex}${hex}${hex}`; // #RRGGBB
}

function hexToRgb(hex: string) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!m) {
    return { r: 0, g: 0, b: 0 };
  }
  return {
    r: parseInt(m[1], 16),
    g: parseInt(m[2], 16),
    b: parseInt(m[3], 16),
  };
}

// Returns '#000000' or '#ffffff' depending on background brightness
function getContrastingTextColor(bgHex: string): string {
  const { r, g, b } = hexToRgb(bgHex);
  // W3C / YIQ-like brightness formula
  const brightness = (r * 299 + g * 587 + b * 114) / 1000; // 0–255 scale [web:208][web:214]
  return brightness > 128 ? '#000000' : '#ffffff';
}


@Component({
  selector: 'app-results',
  standalone: true,
  imports: [CommonModule, FormsModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './results.html',
  styleUrls: ['./results.scss'],
})
export class ResultsComponent implements AfterViewInit {
  @Input() metadata?: ModelMetadata;

  perClassRows = signal<PerClassRow[]>([]);
  perEpochRows = signal<PerEpochRow[]>([]);
  isLoading = signal(true);
  error = signal<string | null>(null);

  readonly priorityOptions: { value: PriorityMetric; label: string }[] = [
    { value: 'auc', label: 'AUC' },
    { value: 'f1', label: 'F1' },
    { value: 'sens', label: 'Sensitivity' },
    { value: 'spec', label: 'Specificity' },
    { value: 'ppv', label: 'PPV' },
    { value: 'npv', label: 'NPV' },
    { value: 'alert_rate', label: 'Lowest Alert Rate' },
  ];

  readonly selectedPriority = signal<PriorityMetric>('auc');
  readonly selectedClass = signal<string>(CLASS_ORDER[0]);
  readonly visibleGroups = signal<Set<string>>(
    new Set([
      'discrimination', 'calibration',
      'threshold', 'spec_threshold',
      'counts', 'spec_thresh_alert_rate',
      'spec_thresh_sens', 'spec_thresh_spec',
      ])
  );

  readonly clinicalGroups = CLINICAL_GROUPS;
  CLASS_ORDER = CLASS_ORDER;


  constructor(private csvLoader: CsvLoaderService) {
    effect(() => {
      if (this.isLoading()) return;
      this.globalConfusion();   // subscribe
      setTimeout(() => this.drawGlobalConfusionMatrix(), 0);
    });

    effect(() => {
      if (this.isLoading()) return;
      this.selectedClassRow();  // subscribe — fires on class change
      setTimeout(() => this.drawPerClassConfusionMatrix(), 0);
    });

  }
 
  async ngAfterViewInit(): Promise<void> {
    try {
      const [perClass, perEpoch] = await Promise.all([
        this.csvLoader.loadPerClass('per_class.csv'),
        this.csvLoader.loadPerEpoch('per_epoch.csv'),
      ]);

      this.perClassRows.set(perClass);
      this.perEpochRows.set(perEpoch);
      this.isLoading.set(false); // effect fires here automatically
    } catch (e) {
      this.error.set(String(e));
      this.isLoading.set(false);
    }
  }

  selectPriority(metric: PriorityMetric): void {
    this.selectedPriority.set(metric);
  }

  selectClass(cls: string): void {
    this.selectedClass.set(cls);
  }

  toggleGroup(key: string): void {
    const next = new Set(this.visibleGroups());
    next.has(key) ? next.delete(key) : next.add(key);
    this.visibleGroups.set(next);
  }

  isGroupVisible(key: string): boolean {
    return this.visibleGroups().has(key);
  }

  sortedEpochRows = computed(() =>
    [...this.perEpochRows()].sort((a, b) => a.epoch - b.epoch)
  );

  classRowsByKey = computed(() => {
    const map = new Map<string, PerClassRow>();
    for (const row of this.perClassRows()) {
      const key = row.class.trim().replace(/ /g, '_');
      map.set(key, row);
    }
    return map;
  });

  selectedClassRow = computed(() => {
    const raw = this.selectedClass();
    const key = raw.trim().replace(/ /g, '_');
    return this.classRowsByKey().get(key) ?? null;
  });

  bestEpochRow = computed(() => {
    const metric = this.selectedPriority();
    const rows = this.sortedEpochRows();
    if (!rows.length) return null;

    const scoreKey = this.epochScoreKey(metric);
    const usable = rows.filter((r) => typeof r[scoreKey] === 'number');
    if (!usable.length) return rows[0];

    return [...usable].sort((a, b) => {
      const av = Number(a[scoreKey]);
      const bv = Number(b[scoreKey]);
      return metric === 'alert_rate' ? av - bv : bv - av;
    })[0];
  });

  macroAuc = computed(() => {
    const row = this.bestEpochRow();
    if (!row) return 0;
    const keys = CLASS_ORDER.filter((c) => c !== 'No Finding').map((c) => `${c}_auc`);
    const vals = keys
      .map((k) => this.toNum(row[k]))
      .filter((v): v is number => typeof v === 'number' && !isNaN(v));
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
  });

  private epochScoreKey(metric: PriorityMetric): string {
    switch (metric) {
      case 'auc': return 'val_auc';
      case 'f1': return 'val_f1';
      case 'sens': return 'val_thresh_sens';
      case 'spec': return 'val_thresh_spec';
      case 'ppv': return 'val_thresh_ppv';
      case 'npv': return 'val_thresh_npv';
      case 'alert_rate': return 'val_thresh_alert_rate';
    }
  }

  displayLabel(cls: string): string {
    const key = cls as ClassKey;
    if (key in CLASS_LABELS) {
      return CLASS_LABELS[key];
    }
    return cls.replace(/_/g, ' ');  
  }

  groupForClass(cls: string): string {
    for (const g of CLINICAL_GROUPS) {
      if (g.classes.includes(cls)) return g.label;
    }
    return 'Other';
  }

  pct(v: number | undefined): string {
    return v == null || isNaN(v) ? '—' : (v * 100).toFixed(1) + '%';
  }

  fmt3(v: number | undefined): string {
    return v == null || isNaN(v) ? '—' : v.toFixed(3);
  }

  fmt4(v: number | undefined): string {
    return v == null || isNaN(v) ? '—' : v.toFixed(4);
  }

  balancedAcc(row: PerClassRow | null | undefined): number {
    if (!row) return 0;
    return (row.sens + row.spec) / 2;
  }

  f1Score(row: PerClassRow): number {
    const denom = 2 * row.tp + row.fp + row.fn;
    return denom > 0 ? (2 * row.tp) / denom : 0;
  }

  alertRateWarning(rate: number): boolean {
    return rate > 0.3;
  }

  globalConfusion = computed(() => {
    const rows = this.perClassRows();
    let tp = 0, fp = 0, tn = 0, fn = 0;

    for (const r of rows) {
      tp += r.tp || 0;
      fp += r.fp || 0;
      tn += r.tn || 0;
      fn += r.fn || 0;
    }

    return { tp, fp, tn, fn };
  });

  private toNum(v: unknown): number | undefined {
    return typeof v === 'number' && isFinite(v) ? v : undefined;
  }


 


  private drawConfusionMatrix(
    selector: string,
    tp: number, fn: number,
    fp: number, tn: number
  ): void {
    const data = [
      { row: 'True +', col: 'Predicted +', value: tp, key: 'TP' },
      { row: 'True +', col: 'Predicted -', value: fn, key: 'FN' },
      { row: 'True -', col: 'Predicted +', value: fp, key: 'FP' },
      { row: 'True -', col: 'Predicted -', value: tn, key: 'TN' },
    ];

    const rows = ['True +', 'True -'];
    const cols = ['Predicted +', 'Predicted -'];

    const maxCorrect = Math.max(tp, tn, 1);
    const maxError   = Math.max(fp, fn, 1);

    const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

    const values = data.map(d => d.value);
    const max = Math.max(...values, 1);  // avoid 0

    const colorScale = d3.scaleLinear()
      .domain([0, max])
      .range([0, 1]);

    const greenScale = d3.scaleLinear<string>()
      .domain([0, maxCorrect])
      .range(isDark ? ['#064e3b', '#6ee7b7'] : ['#d1fae5', '#065f46']);

    const redScale = d3.scaleLinear<string>()
      .domain([0, maxError])
      .range(isDark ? ['#450a0a', '#fca5a5'] : ['#fee2e2', '#991b1b']);

    const container = d3.select(selector);
    container.selectAll('*').remove();

    const cellSize = 120;
    const margin = { top: 44, right: 0, bottom: 16, left: 0 };
    const width  = margin.left + margin.right + cellSize * 2;
    const height = margin.top  + margin.bottom + cellSize * 2;

    const svg = container
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('class', 'confusion-svg');

    svg.selectAll<SVGTextElement, string>('text.col-label')
      .data(cols).enter()
      .append('text')
      .attr('class', 'col-label')
      .attr('x', (_, j) => margin.left + j * cellSize + cellSize / 2)
      .attr('y', margin.top - 12)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--c-text-mid)')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .text(d => d);

    svg.selectAll<SVGTextElement, string>('text.row-label')
      .data(rows).enter()
      .append('text')
      .attr('class', 'row-label')
      .attr('x', margin.left - 10)
      .attr('y', (_, i) => margin.top + i * cellSize + cellSize / 2)
      .attr('text-anchor', 'end')
      .attr('dominant-baseline', 'middle')
      .attr('fill', 'var(--c-text-mid)')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .text(d => d);

    const cells = svg
      .selectAll('g.cm-cell')
      .data(data)
      .enter()
      .append('g')
      .attr('class', 'cm-cell')
      .attr('transform', d => {
        const rowIndex = rows.indexOf(d.row);
        const colIndex = cols.indexOf(d.col);
        const x = margin.left + colIndex * cellSize;
        const y = margin.top + rowIndex * cellSize;
        return `translate(${x},${y})`;
      });

    // Rects
    cells
      .append('rect')
      .attr('width', cellSize)
      .attr('height', cellSize)
      .attr('rx', 8)
      .attr('ry', 8)
      .attr('fill', d => {
        const t = colorScale(d.value);           // 0–1
        return scalarToGrayHex(t);              // hex bg
      })
      .attr('stroke', '#000000')
      .attr('stroke-width', 0.8);

    // Labels (TP/FP/TN/FN)
    cells
      .append('text')
      .attr('x', cellSize / 2)
      .attr('y', cellSize / 2 - 8)
      .attr('text-anchor', 'middle')
      .attr('fill', d => {
        const t = colorScale(d.value);
        const bg = scalarToGrayHex(t);
        return getContrastingTextColor(bg);    // black or white
      })
      .style('font-size', '12px')
      .style('font-weight', '600')
      .text(d => d.key);

    // Counts
    cells
      .append('text')
      .attr('x', cellSize / 2)
      .attr('y', cellSize / 2 + 10)
      .attr('text-anchor', 'middle')
      .attr('fill', d => {
        const t = colorScale(d.value);
        const bg = scalarToGrayHex(t);
        return getContrastingTextColor(bg);
      })
      .style('font-size', '12px')
      .text(d => d.value.toLocaleString());

  }

  // Thin wrappers so effect() calls stay readable
  private drawGlobalConfusionMatrix(): void {
    const g = this.globalConfusion();
    this.drawConfusionMatrix('#confusion-d3-container', g.tp, g.fn, g.fp, g.tn);
  }

  private drawPerClassConfusionMatrix(): void {
    const row = this.selectedClassRow();
    if (!row) return;
    this.drawConfusionMatrix('#class-confusion-d3-container', row.tp, row.fn, row.fp, row.tn);
  }



}
export type { PerClassRow, PerEpochRow, ModelMetadata };