import { ComponentFixture, TestBed } from '@angular/core/testing';

import { OperatingPointTooltip } from './operating-point-tooltip';

describe('OperatingPointTooltip', () => {
  let component: OperatingPointTooltip;
  let fixture: ComponentFixture<OperatingPointTooltip>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [OperatingPointTooltip],
    }).compileComponents();

    fixture = TestBed.createComponent(OperatingPointTooltip);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
