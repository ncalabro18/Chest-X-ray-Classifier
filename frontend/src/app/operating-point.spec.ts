import { TestBed } from '@angular/core/testing';

import { OperatingPoint } from './operating-point';

describe('OperatingPoint', () => {
  let service: OperatingPoint;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(OperatingPoint);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
