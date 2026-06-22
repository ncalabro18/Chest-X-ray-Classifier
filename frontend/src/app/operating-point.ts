import { inject, Injectable } from '@angular/core';
import { CsvLoaderService, PerClassRow } from './csv-loader.service';

@Injectable({
  providedIn: 'root',
})
export class OperatingPoint {

  private csvLoader = inject(CsvLoaderService);
  private cache: Promise<Map<string, PerClassRow[]>> | null = null;

  private load(): Promise<Map<string, PerClassRow[]>> {
    if (!this.cache) {
      this.cache = this.csvLoader.loadPerClass('per_class.csv').then((rows) => {
        const map = new Map<string, PerClassRow[]>();
        for (const row of rows) {
          const key = row.class.trim().replace(/ /g, '_');
          if (!map.has(key)) map.set(key, []);
          map.get(key)!.push(row);
        }
        return map;
      });
    }
    return this.cache;
  }

  private bestEpochCache: Promise<number> | null = null;
  private getBestEpoch(): Promise<number> {
    if (!this.bestEpochCache) {
      this.bestEpochCache = this.load().then((map) => {
        const epochAuc = new Map<number, number[]>();

        for (const rows of map.values()) {
          for (const row of rows) {
            if (isNaN(row.auc)) continue;
            if (!epochAuc.has(row.epoch)) epochAuc.set(row.epoch, []);
            epochAuc.get(row.epoch)!.push(row.auc);
          }
        }

        if (!epochAuc.size) throw new Error('no valid auc values in per_class.csv');

        let bestEpoch = -1;
        let bestAuc = -Infinity;
        for (const [epoch, aucs] of epochAuc) {
          const mean = aucs.reduce((a, b) => a + b, 0) / aucs.length;
          if (mean > bestAuc) { bestAuc = mean; bestEpoch = epoch; }
        }

        console.log('bestEpoch:', bestEpoch, 'mean AUC:', bestAuc.toFixed(4));
        return bestEpoch;
      });
    }
    return this.bestEpochCache;
  }
  
  async getClosestRow(
    className: string,
    deployedThreshold: number,
  ): Promise<PerClassRow | null> {
    const [map, bestEpoch] = await Promise.all([this.load(), this.getBestEpoch()]);
    const key = className.trim().replace(/ /g, '_');
    const rows = (map.get(key) ?? []).filter((r) => r.epoch === bestEpoch);
    if (!rows.length) return null;

    // Round to 4 decimal places to handle floating point issues
    const deployed = Math.round(deployedThreshold * 10000) / 10000;

    const below = rows.filter(r => {
      const threshold = Math.round(r.threshold_value * 10000) / 10000;
      return threshold <= deployed;
    });

    console.log('deployedThreshold:', deployedThreshold);
    console.log('below count:', below.length);
    console.log('below thresholds:', below.map(r => r.threshold_value));
    console.log('all thresholds:', rows.map(r => r.threshold_value));

    if (below.length) {
      return below.reduce((best, row) => {
        const bestThreshold = Math.round(best.threshold_value * 10000) / 10000;
        const rowThreshold = Math.round(row.threshold_value * 10000) / 10000;
        return rowThreshold > bestThreshold ? row : best;
      });
    }

    return rows.reduce((best, row) =>
      row.threshold_value < best.threshold_value ? row : best
    );
  }

  async getSpecRow(className: string): Promise<PerClassRow | null> {
    const [map, bestEpoch] = await Promise.all([this.load(), this.getBestEpoch()]);
    const key = className.trim().replace(/ /g, '_');
    const rows = (map.get(key) ?? []).filter(r => r.epoch === bestEpoch);
    if (!rows.length) return null;
    return rows.reduce((best, row) =>
      row.threshold_id > best.threshold_id ? row : best
    );
  }

}
