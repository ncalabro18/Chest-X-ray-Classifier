import { Injectable } from '@angular/core';
import Papa from 'papaparse';

export interface PerClassRow {
  epoch: number;
  class: string;
  threshold: number;
  spec_threshold: number;
  auc: number;
  sens: number;
  spec: number;
  ppv: number;
  npv: number;
  alert_rate: number;
  ece: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  spec_thresh_sens:       number;
  spec_thresh_spec:       number;
  spec_thresh_ppv:        number;
  spec_thresh_npv:        number;
  spec_thresh_alert_rate: number;
}

export interface PerEpochRow {
  epoch: number;
  tr_loss: number;
  tr_auc: number;
  tr_f1: number;
  val_loss: number;
  val_auc: number;
  val_f1: number;
  val_thresh_sens: number;
  val_thresh_spec: number;
  val_thresh_ppv: number;
  val_thresh_npv: number;
  val_thresh_alert_rate: number;
  [key: string]: number | string | undefined;
}

export interface ModelMetadata {
  runName: string;
  architecture: string;
  pretraining: string;
  dataset: string;
  totalImages?: number;
  trainSize?: number;
  valSize?: number;
  testSize?: number;
  bestEpoch: number;
  checkpointFile?: string;
  trainedAt?: string;
  notes?: string;
}

@Injectable({
  providedIn: 'root',
})
export class CsvLoaderService {

  async loadPerClass(path: string): Promise<PerClassRow[]> {
    const text = await fetch(path).then((res) => res.text());

    return new Promise((resolve, reject) => {
      Papa.parse(text, {
        header: true,
        skipEmptyLines: true,
        delimiter: ',',
        transformHeader: (h) => h.trim(),
        complete: (results) => {
          const rows: PerClassRow[] = results.data.map((r: any) => ({
            epoch: Number(r['epoch']),
            class: String(r['class']),
            threshold: Number(r['threshold']),
            spec_threshold: Number(r['spec_threshold']),
            auc: Number(r['auc']),
            sens: Number(r['sens']),
            spec: Number(r['spec']),
            ppv: Number(r['ppv']),
            npv: Number(r['npv']),
            alert_rate: Number(r['alert_rate']),
            ece: Number(r['ece']),
            tp: Number(r['tp']),
            fp: Number(r['fp']),
            tn: Number(r['tn']),
            fn: Number(r['fn']),
            spec_thresh_sens: Number(r['spec_thresh_sens']),
            spec_thresh_spec: Number(r['spec_thresh_spec']),
            spec_thresh_ppv: Number(r['spec_thresh_ppv']),
            spec_thresh_npv: Number(r['spec_thresh_npv']),
            spec_thresh_alert_rate: Number(r['spec_thresh_alert_rate'])
            
          }));
          resolve(rows);
        },
        error: (err: any) => reject(err),
      });
    });
  }

  async loadPerEpoch(path: string): Promise<PerEpochRow[]> {
    const text = await fetch(path).then((res) => res.text());

    return new Promise((resolve, reject) => {
      Papa.parse(text, {
        header: true,
        skipEmptyLines: true,
        delimiter: ',',
        transformHeader: (h) => h.trim(),
        complete: (resultsRaw) => {
          const rows: PerEpochRow[] = resultsRaw.data.map((r: any) => {
            const row: PerEpochRow = {
              epoch: Number(r['epoch']),
              tr_loss: Number(r['tr_loss']),
              tr_auc: Number(r['tr_auc']),
              tr_f1: Number(r['tr_f1']),
              val_loss: Number(r['val_loss']),
              val_auc: Number(r['val_auc']),
              val_f1: Number(r['val_f1']),
              val_thresh_sens: Number(r['val_thresh_sens']),
              val_thresh_spec: Number(r['val_thresh_spec']),
              val_thresh_ppv: Number(r['val_thresh_ppv']),
              val_thresh_npv: Number(r['val_thresh_npv']),
              val_thresh_alert_rate: Number(r['val_thresh_alert_rate']),
            };

            for (const key in r) {
              if (!Object.prototype.hasOwnProperty.call(row, key)) {
                const val = r[key];
                row[key] = typeof val === 'string' ? (Number(val) || 0) : val;
              }
            }

            return row;
          });
          resolve(rows);
        },
        error: (err: any) => reject(err),
      });
    });
  }
}