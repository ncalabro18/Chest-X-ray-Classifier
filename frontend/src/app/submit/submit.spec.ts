import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Submit } from './submit';

describe('Submit', () => {
  let component: Submit;
  let fixture: ComponentFixture<Submit>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Submit],
    }).compileComponents();

    fixture = TestBed.createComponent(Submit);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
