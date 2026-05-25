import { TestBed } from '@angular/core/testing';

import { ClassifierState } from './classifier-state';

describe('ClassifierState', () => {
  let service: ClassifierState;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ClassifierState);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
