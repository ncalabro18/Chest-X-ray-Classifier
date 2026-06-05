import { TestBed } from '@angular/core/testing';

import { CsvLoaderService } from './csv-loader.service';

describe('CsvLoaderService', () => {
  let service: CsvLoaderService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(CsvLoaderService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
