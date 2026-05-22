"""
© 2026 Nicholas J. Calabro. All rights reserved.


"""
import cv2


from torch.utils.data import DataLoader, WeightedRandomSampler


### Data Loader Parameters ###
# Ran without locking when workers were 2 and 2
# CPU will bottleneck less with higher workers, may deadlock (immediately)

BATCH_SIZE_VAL   = 16
BATCH_SIZE_TRAIN = 16
PREFETECH_FACTOR = 2

LOADER_WORKERS_TRAIN = 10
LOADER_WORKERS_VALUE = 2
PERSISTENT_WORKERS   = True

# Unlikely this should change
BASE_BATCH_SIZE  = 16

# Sample extra from low appearing catagories
SAMPLER_POWER = 0.17

# Worker Init; keep for memory safety
def worker_init_fn(worker_id):
    cv2.setNumThreads(0)



def init_train_dataloader(train_ds, train_idx, label_matrix):
    
    # Sampler
    class_counts = label_matrix[train_idx].sum(axis=0)
    class_weights = 1.0 / (class_counts + 1e-6) ** SAMPLER_POWER
    sample_weights = (label_matrix[train_idx] * class_weights).sum(axis=1)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return DataLoader(
        train_ds,
        batch_size=BATCH_SIZE_TRAIN,
        sampler=sampler,
        num_workers=LOADER_WORKERS_TRAIN,
        worker_init_fn=worker_init_fn,
        persistent_workers=PERSISTENT_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETECH_FACTOR,
    )

def init_value_dataloader(value_ds):
    return DataLoader(
        value_ds,
        batch_size=BATCH_SIZE_VAL,
        shuffle=False,
        num_workers=LOADER_WORKERS_VALUE,
        worker_init_fn=worker_init_fn,
        persistent_workers=PERSISTENT_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETECH_FACTOR,
    )

def init_thresh_dataloader(thresh_ds):
    return DataLoader(
        thresh_ds,
        batch_size=BATCH_SIZE_VAL,
        shuffle=False,
        num_workers=LOADER_WORKERS_VALUE,
        worker_init_fn=worker_init_fn,
        persistent_workers=PERSISTENT_WORKERS,
        pin_memory=True,
    )

def print_dataloader_parameters():
    print("  BATCH_SIZE_VAL", BATCH_SIZE_VAL)
    print("  BATCH_SIZE_TRAIN", BATCH_SIZE_TRAIN)
    print("  TRAIN_LOADER_WORKERS", LOADER_WORKERS_TRAIN )
    print("  VALUE_LOADER_WORKERS", LOADER_WORKERS_VALUE)
    print("  PREFETECH_FACTOR", PREFETECH_FACTOR)
    print("  PERSISTENT_WORKERS", PERSISTENT_WORKERS)
    print("  SAMPLER_POWER", SAMPLER_POWER)
